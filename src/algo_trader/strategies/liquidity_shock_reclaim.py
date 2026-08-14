"""Liquidity-shock exhaustion reclaim signal generation.

Version 1.1 narrows the original exhaustion-reclaim hypothesis to the most
structurally important liquidity pools: previous-day high/low only. A qualifying
shock must also carry exceptional same-time-of-day volume (RVOL >= 12x). LONG and
SHORT rules remain exact mirrors: shock, prior-day-level sweep, same-bar reclaim,
then one-bar confirmation. The confirmation candle must complete before a Signal
exists, so its availability time is the decision timestamp. Historical
normalization excludes the current session. The strategy creates no orders,
allocations, ML decisions, or execution plans. Initial-stop evidence and the
R-multiple trailing mechanics are preserved from v1.0, while the hard target is
reduced modestly from 1.5R to 1.25R. Exit metadata is stored for downstream
composition after an actual entry fill is known.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from itertools import pairwise
from statistics import median
from types import MappingProxyType
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from algo_trader.data import bar_available_at
from algo_trader.domain import Side, Signal
from algo_trader.indicators import atr
from algo_trader.strategies.validation import validate_strategy_input

MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")
@dataclass(frozen=True, slots=True)
class LiquidityShockReclaimConfig:
    """Single immutable source for Strategy 1 v1.1 behavior and metadata."""

    timeframe_minutes: int = 5
    signal_time_start: time = time(9, 40)
    signal_time_end: time = time(14, 35)
    shock_horizon_bars: int = 2
    shock_history_sessions: int = 60
    shock_robust_z_threshold: float = 3.0
    mad_consistency_scale: float = 1.4826
    volume_history_sessions: int = 20
    relative_volume_threshold: float = 12.0
    liquidity_history_sessions: int = 20
    minimum_median_daily_turnover_rupees: int = 200_000_000
    atr_period: int = 14
    level_policy: str = "PRIOR_DAY_EXTREME_ONLY"
    minimum_penetration_atr: float = 0.10
    maximum_penetration_atr: float = 0.75
    minimum_reclaim_atr: float = 0.05
    confirmation_bars: int = 1
    stop_buffer_atr: float = 0.10
    hard_target_r: float = 1.25
    maximum_hold_minutes: int = 30
    latest_exit_time: time = time(15, 10)
    trailing_breakeven_trigger_r: float = 0.75
    trailing_breakeven_stop_r: float = 0.0
    trailing_profit_lock_trigger_r: float = 1.0
    trailing_profit_lock_stop_r: float = 0.25
    trailing_distance_r: float = 0.50
    max_signals_per_symbol_per_day: int = 1

    @property
    def warmup_bars(self) -> int:
        # NSE cash has 75 five-minute bars/session; 60 sessions preserves v1.1's 4500.
        return self.shock_history_sessions * 75

    def as_parameters(self) -> Mapping[str, Any]:
        values = {
            "timeframe_minutes": self.timeframe_minutes,
            "signal_time_start": self.signal_time_start.strftime("%H:%M"),
            "signal_time_end": self.signal_time_end.strftime("%H:%M"),
            "shock_horizon_bars": self.shock_horizon_bars,
            "shock_history_sessions": self.shock_history_sessions,
            "shock_robust_z_threshold": self.shock_robust_z_threshold,
            "mad_consistency_scale": self.mad_consistency_scale,
            "volume_history_sessions": self.volume_history_sessions,
            "relative_volume_threshold": self.relative_volume_threshold,
            "liquidity_history_sessions": self.liquidity_history_sessions,
            "minimum_median_daily_turnover_rupees": (
                self.minimum_median_daily_turnover_rupees
            ),
            "atr_period": self.atr_period,
            "level_policy": self.level_policy,
            "minimum_penetration_atr": self.minimum_penetration_atr,
            "maximum_penetration_atr": self.maximum_penetration_atr,
            "minimum_reclaim_atr": self.minimum_reclaim_atr,
            "confirmation_bars": self.confirmation_bars,
            "stop_buffer_atr": self.stop_buffer_atr,
            # Historical metadata aliases share one behavioral driver.
            "reward_r_multiple": self.hard_target_r,
            "maximum_hold_minutes": self.maximum_hold_minutes,
            "latest_exit_time": self.latest_exit_time.strftime("%H:%M"),
            "trailing_breakeven_trigger_r": self.trailing_breakeven_trigger_r,
            "trailing_breakeven_stop_r": self.trailing_breakeven_stop_r,
            "trailing_profit_lock_trigger_r": self.trailing_profit_lock_trigger_r,
            "trailing_profit_lock_stop_r": self.trailing_profit_lock_stop_r,
            "trailing_distance_r": self.trailing_distance_r,
            "trailing_hard_target_r": self.hard_target_r,
            "max_signals_per_symbol_per_day": self.max_signals_per_symbol_per_day,
        }
        return MappingProxyType(values)


@dataclass(frozen=True, slots=True)
class _Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str

    @property
    def trading_date(self) -> date:
        return self.timestamp.date()

    @property
    def slot(self) -> time:
        return self.timestamp.time().replace(tzinfo=None)


@dataclass(frozen=True, slots=True)
class _Level:
    price: float
    level_type: str
    known_at: datetime


@dataclass(frozen=True, slots=True)
class _Normalization:
    shock_return: float
    shock_median: float
    shock_mad: float
    robust_z: float
    volume_median: float
    relative_volume: float
    median_turnover: float


class LiquidityShockReclaimStrategy:
    """Causal symmetric prior-day liquidity-shock exhaustion-reclaim strategy."""

    strategy_id = "liquidity-shock-exhaustion-reclaim"
    strategy_version = "1.1.0"
    def __init__(self, config: LiquidityShockReclaimConfig | None = None) -> None:
        self.config = config or LiquidityShockReclaimConfig()
        self.parameters = self.config.as_parameters()
        self.warmup_bars = self.config.warmup_bars

    def generate_signals(self, candles: pl.DataFrame) -> list[Signal]:
        """Return cumulative signals available from completed candle history."""
        validate_strategy_input(candles)
        bars = _bars_in_market_timezone(candles)
        if len(bars) < 4:
            return []

        config = self.config
        atr_values = atr(candles, period=config.atr_period).to_list()
        dates, indices_by_date, slot_index, date_position = _session_index(bars)
        signals: list[Signal] = []
        emitted_by_date: dict[date, int] = {}

        for event_index in range(
            config.shock_horizon_bars,
            len(bars) - config.confirmation_bars,
        ):
            event = bars[event_index]
            confirmation = bars[event_index + config.confirmation_bars]
            if (
                emitted_by_date.get(event.trading_date, 0)
                >= config.max_signals_per_symbol_per_day
            ):
                continue
            if not _consecutive_event_and_confirmation(
                bars, event_index, confirmation, config
            ):
                continue
            market_values = (
                event.open,
                event.high,
                event.low,
                event.close,
                event.volume,
                confirmation.open,
                confirmation.high,
                confirmation.low,
                confirmation.close,
            )
            if not all(math.isfinite(value) for value in market_values):
                continue

            signal_timestamp = bar_available_at(
                confirmation.timestamp, config.timeframe_minutes
            )
            if not _within_signal_window(signal_timestamp, config):
                continue
            event_atr = float(atr_values[event_index])
            if not _positive_finite(event_atr):
                continue

            prior_dates = dates[: date_position[event.trading_date]]
            normalization = _normalization(
                bars,
                event_index,
                prior_dates,
                indices_by_date,
                slot_index,
                config,
            )
            if normalization is None:
                continue
            if normalization.robust_z <= -config.shock_robust_z_threshold:
                side = Side.LONG
            elif normalization.robust_z >= config.shock_robust_z_threshold:
                side = Side.SHORT
            else:
                continue

            qualifying = _qualifying_level(
                bars,
                event_index,
                side,
                event_atr,
                prior_dates,
                indices_by_date,
                config,
            )
            if qualifying is None:
                continue
            level, penetration, reclaim_depth = qualifying
            if not _confirmation_passes(confirmation, event, level.price, side):
                continue

            stop_reference = (
                event.low - config.stop_buffer_atr * event_atr
                if side is Side.LONG
                else event.high + config.stop_buffer_atr * event_atr
            )
            feature_snapshot = _feature_snapshot(
                event,
                confirmation,
                level,
                normalization,
                event_atr,
                penetration,
                reclaim_depth,
                stop_reference,
                config,
            )
            signals.append(
                Signal(
                    strategy_id=self.strategy_id,
                    strategy_version=self.strategy_version,
                    symbol=event.symbol,
                    timestamp=signal_timestamp,
                    side=side,
                    strategy_parameters=self.parameters,
                    feature_snapshot=feature_snapshot,
                )
            )
            emitted_by_date[event.trading_date] = (
                emitted_by_date.get(event.trading_date, 0) + 1
            )

        return signals


def _bars_in_market_timezone(candles: pl.DataFrame) -> tuple[_Bar, ...]:
    bars = []
    for row in candles.rows(named=True):
        timestamp = row["timestamp"].astimezone(MARKET_TIMEZONE)
        bars.append(
            _Bar(
                timestamp=timestamp,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                symbol=str(row["symbol"]),
            )
        )
    return tuple(bars)


def _session_index(
    bars: tuple[_Bar, ...],
) -> tuple[
    list[date],
    dict[date, list[int]],
    dict[tuple[date, time], int],
    dict[date, int],
]:
    dates: list[date] = []
    indices_by_date: dict[date, list[int]] = {}
    slot_index: dict[tuple[date, time], int] = {}
    date_position: dict[date, int] = {}
    for index, bar in enumerate(bars):
        if bar.trading_date not in indices_by_date:
            date_position[bar.trading_date] = len(dates)
            dates.append(bar.trading_date)
            indices_by_date[bar.trading_date] = []
        indices_by_date[bar.trading_date].append(index)
        slot_index[(bar.trading_date, bar.slot)] = index
    return dates, indices_by_date, slot_index, date_position


def _consecutive_event_and_confirmation(
    bars: tuple[_Bar, ...],
    event_index: int,
    confirmation: _Bar,
    config: LiquidityShockReclaimConfig,
) -> bool:
    event = bars[event_index]
    first_index = event_index - config.shock_horizon_bars
    last_index = event_index + config.confirmation_bars
    sequence = bars[first_index : last_index + 1]
    delta = timedelta(minutes=config.timeframe_minutes)
    return all(item.trading_date == event.trading_date for item in sequence) and all(
        current.timestamp - previous.timestamp == delta
        for previous, current in pairwise(sequence)
    ) and confirmation is sequence[-1]


def _normalization(
    bars: tuple[_Bar, ...],
    event_index: int,
    prior_dates: list[date],
    indices_by_date: dict[date, list[int]],
    slot_index: dict[tuple[date, time], int],
    config: LiquidityShockReclaimConfig,
) -> _Normalization | None:
    event = bars[event_index]
    shock_return = _shock_return(bars, event_index, config)
    if shock_return is None:
        return None

    # Same-time-of-day normalization avoids mixing structurally different slots.
    shock_history: list[float] = []
    volume_history: list[float] = []
    for prior_date in reversed(prior_dates):
        equivalent_index = slot_index.get((prior_date, event.slot))
        if equivalent_index is not None:
            if len(shock_history) < config.shock_history_sessions:
                historical_return = _shock_return(bars, equivalent_index, config)
                if historical_return is not None:
                    shock_history.append(historical_return)
            if len(volume_history) < config.volume_history_sessions:
                historical_volume = bars[equivalent_index].volume
                if _positive_finite(historical_volume):
                    volume_history.append(historical_volume)
        if (
            len(shock_history) == config.shock_history_sessions
            and len(volume_history) == config.volume_history_sessions
        ):
            break
    if (
        len(shock_history) != config.shock_history_sessions
        or len(volume_history) != config.volume_history_sessions
    ):
        return None

    shock_median = float(median(shock_history))
    shock_mad = float(median(abs(value - shock_median) for value in shock_history))
    # 1.4826 makes MAD comparable to standard deviation under a normal baseline.
    scale = config.mad_consistency_scale * shock_mad
    if not _positive_finite(scale):
        return None
    robust_z = (shock_return - shock_median) / scale
    volume_median = float(median(volume_history))
    if not _positive_finite(volume_median):
        return None
    relative_volume = event.volume / volume_median

    if len(prior_dates) < config.liquidity_history_sessions:
        return None
    turnovers = [
        _daily_turnover(bars, indices_by_date[prior_date])
        for prior_date in prior_dates[-config.liquidity_history_sessions :]
    ]
    if any(value is None for value in turnovers):
        return None
    median_turnover = float(median(value for value in turnovers if value is not None))
    values = (
        shock_return,
        shock_median,
        shock_mad,
        robust_z,
        volume_median,
        relative_volume,
        median_turnover,
    )
    if not all(math.isfinite(value) for value in values):
        return None
    if (
        relative_volume < config.relative_volume_threshold
        or median_turnover < config.minimum_median_daily_turnover_rupees
    ):
        return None
    return _Normalization(*values)


def _shock_return(
    bars: tuple[_Bar, ...],
    index: int,
    config: LiquidityShockReclaimConfig,
) -> float | None:
    if index < config.shock_horizon_bars:
        return None
    start = bars[index - config.shock_horizon_bars]
    end = bars[index]
    window = bars[index - config.shock_horizon_bars : index + 1]
    delta = timedelta(minutes=config.timeframe_minutes)
    if (
        any(item.trading_date != end.trading_date for item in window)
        or any(
            current.timestamp - previous.timestamp != delta
            for previous, current in pairwise(window)
        )
        or not _positive_finite(start.close)
        or not math.isfinite(end.close)
    ):
        return None
    result = end.close / start.close - 1.0
    return result if math.isfinite(result) else None


def _daily_turnover(bars: tuple[_Bar, ...], indices: list[int]) -> float | None:
    values = [bars[index].close * bars[index].volume for index in indices]
    if not values or not all(math.isfinite(value) and value >= 0 for value in values):
        return None
    return sum(values)


def _qualifying_level(
    bars: tuple[_Bar, ...],
    event_index: int,
    side: Side,
    event_atr: float,
    prior_dates: list[date],
    indices_by_date: dict[date, list[int]],
    config: LiquidityShockReclaimConfig,
) -> tuple[_Level, float, float] | None:
    event = bars[event_index]
    levels = _candidate_levels(bars, side, prior_dates, indices_by_date, config)
    for level in levels:
        penetration = (
            level.price - event.low
            if side is Side.LONG
            else event.high - level.price
        )
        reclaim = (
            event.close - level.price
            if side is Side.LONG
            else level.price - event.close
        )
        if (
            _at_least(penetration, config.minimum_penetration_atr * event_atr)
            and _at_most(penetration, config.maximum_penetration_atr * event_atr)
            and _at_least(reclaim, config.minimum_reclaim_atr * event_atr)
        ):
            return level, penetration, reclaim
    return None


def _candidate_levels(
    bars: tuple[_Bar, ...],
    side: Side,
    prior_dates: list[date],
    indices_by_date: dict[date, list[int]],
    config: LiquidityShockReclaimConfig,
) -> tuple[_Level, ...]:
    """Return the single previous-day extreme eligible in v1.1.

    Session-so-far and confirmed swing levels were valid v1.0 research features
    but are intentionally excluded from v1.1. The previous-day extreme is known
    before the current session opens, preserving the original causal contract.
    """
    if config.level_policy != "PRIOR_DAY_EXTREME_ONLY":
        raise ValueError(f"unsupported level_policy: {config.level_policy}")
    if not prior_dates:
        return ()

    prior_indices = indices_by_date[prior_dates[-1]]
    if not prior_indices:
        return ()

    price = (
        min(bars[index].low for index in prior_indices)
        if side is Side.LONG
        else max(bars[index].high for index in prior_indices)
    )
    return (
        _Level(
            price=price,
            level_type="PDL" if side is Side.LONG else "PDH",
            known_at=bar_available_at(
                bars[prior_indices[-1]].timestamp,
                config.timeframe_minutes,
            ),
        ),
    )


def _confirmation_passes(
    confirmation: _Bar, event: _Bar, level: float, side: Side
) -> bool:
    if side is Side.LONG:
        return confirmation.low >= level and confirmation.close > event.close
    return confirmation.high <= level and confirmation.close < event.close


def _within_signal_window(
    timestamp: datetime,
    config: LiquidityShockReclaimConfig,
) -> bool:
    local_time = timestamp.astimezone(MARKET_TIMEZONE).time().replace(tzinfo=None)
    return config.signal_time_start <= local_time <= config.signal_time_end


def _feature_snapshot(
    event: _Bar,
    confirmation: _Bar,
    level: _Level,
    normalization: _Normalization,
    event_atr: float,
    penetration: float,
    reclaim_depth: float,
    stop_reference: float,
    config: LiquidityShockReclaimConfig,
) -> dict[str, Any]:
    # R metadata is stored instead of moving a stop before an actual fill exists.
    values: dict[str, Any] = {
        "event_bar_start": event.timestamp,
        "confirmation_bar_start": confirmation.timestamp,
        "level_type": level.level_type,
        "level_known_at": level.known_at,
        "shock_return": normalization.shock_return,
        "shock_history_median": normalization.shock_median,
        "shock_history_mad": normalization.shock_mad,
        "shock_robust_z": normalization.robust_z,
        "event_volume": event.volume,
        "historical_slot_volume_median": normalization.volume_median,
        "relative_volume": normalization.relative_volume,
        "median_daily_turnover": normalization.median_turnover,
        "atr": event_atr,
        "level_price": level.price,
        "penetration": penetration,
        "penetration_atr": penetration / event_atr,
        "reclaim_depth": reclaim_depth,
        "reclaim_atr": reclaim_depth / event_atr,
        "event_open": event.open,
        "event_high": event.high,
        "event_low": event.low,
        "event_close": event.close,
        "confirmation_open": confirmation.open,
        "confirmation_high": confirmation.high,
        "confirmation_low": confirmation.low,
        "confirmation_close": confirmation.close,
        "confirmation_return_from_event_close": confirmation.close / event.close - 1.0,
        "stop_reference_price": stop_reference,
        "stop_buffer_atr": config.stop_buffer_atr,
        "reward_r_multiple": config.hard_target_r,
        "maximum_hold_minutes": config.maximum_hold_minutes,
        "trailing_breakeven_trigger_r": config.trailing_breakeven_trigger_r,
        "trailing_breakeven_stop_r": config.trailing_breakeven_stop_r,
        "trailing_profit_lock_trigger_r": config.trailing_profit_lock_trigger_r,
        "trailing_profit_lock_stop_r": config.trailing_profit_lock_stop_r,
        "trailing_distance_r": config.trailing_distance_r,
        "trailing_hard_target_r": config.hard_target_r,
    }
    numeric = [value for value in values.values() if isinstance(value, int | float)]
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("signal feature snapshot contains a non-finite numeric value")
    return values


def _positive_finite(value: float) -> bool:
    return math.isfinite(value) and value > 0


def _at_least(value: float, boundary: float) -> bool:
    return value >= boundary or math.isclose(value, boundary, rel_tol=1e-12, abs_tol=1e-12)


def _at_most(value: float, boundary: float) -> bool:
    return value <= boundary or math.isclose(value, boundary, rel_tol=1e-12, abs_tol=1e-12)
