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
_RELATIVE_VOLUME_THRESHOLD = 12.0
_LEVEL_POLICY = "PRIOR_DAY_EXTREME_ONLY"

_PARAMETERS: Mapping[str, Any] = MappingProxyType(
    {
        "timeframe_minutes": 5,
        "signal_time_start": "09:40",
        "signal_time_end": "14:35",
        "shock_horizon_bars": 2,
        "shock_history_sessions": 60,
        "shock_robust_z_threshold": 3.0,
        "mad_consistency_scale": 1.4826,
        "volume_history_sessions": 20,
        "relative_volume_threshold": _RELATIVE_VOLUME_THRESHOLD,
        "liquidity_history_sessions": 20,
        "minimum_median_daily_turnover_rupees": 200_000_000,
        "atr_period": 14,
        "level_policy": _LEVEL_POLICY,
        "minimum_penetration_atr": 0.10,
        "maximum_penetration_atr": 0.75,
        "minimum_reclaim_atr": 0.05,
        "confirmation_bars": 1,
        "stop_buffer_atr": 0.10,
        "reward_r_multiple": 1.25,
        "maximum_hold_minutes": 30,
        "latest_exit_time": "15:10",
        "trailing_breakeven_trigger_r": 0.75,
        "trailing_breakeven_stop_r": 0.0,
        "trailing_profit_lock_trigger_r": 1.0,
        "trailing_profit_lock_stop_r": 0.25,
        "trailing_distance_r": 0.50,
        "trailing_hard_target_r": 1.25,
        "max_signals_per_symbol_per_day": 1,
    }
)

_SIGNAL_TIME_START = time(9, 40)
_SIGNAL_TIME_END = time(14, 35)
_BAR_DELTA = timedelta(minutes=5)


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
    parameters = _PARAMETERS
    warmup_bars = 4500

    def generate_signals(self, candles: pl.DataFrame) -> list[Signal]:
        """Return cumulative signals available from completed candle history."""
        validate_strategy_input(candles)
        bars = _bars_in_market_timezone(candles)
        if len(bars) < 4:
            return []

        atr_values = atr(candles, period=14).to_list()
        dates, indices_by_date, slot_index, date_position = _session_index(bars)
        signals: list[Signal] = []
        emitted_dates: set[date] = set()

        for event_index in range(2, len(bars) - 1):
            event = bars[event_index]
            confirmation = bars[event_index + 1]
            if event.trading_date in emitted_dates:
                continue  # Frozen one-signal/day gate; later setups cannot replace it.
            if not _consecutive_event_and_confirmation(
                bars, event_index, confirmation
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

            signal_timestamp = bar_available_at(confirmation.timestamp, 5)
            if not _within_signal_window(signal_timestamp):
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
            )
            if normalization is None:
                continue
            if normalization.robust_z <= -3.0:
                side = Side.LONG
            elif normalization.robust_z >= 3.0:
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
            )
            if qualifying is None:
                continue
            level, penetration, reclaim_depth = qualifying
            if not _confirmation_passes(confirmation, event, level.price, side):
                continue

            stop_reference = (
                event.low - 0.10 * event_atr
                if side is Side.LONG
                else event.high + 0.10 * event_atr
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
            emitted_dates.add(event.trading_date)

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
    bars: tuple[_Bar, ...], event_index: int, confirmation: _Bar
) -> bool:
    event = bars[event_index]
    return (
        bars[event_index - 2].trading_date == event.trading_date
        and bars[event_index - 1].trading_date == event.trading_date
        and bars[event_index - 1].timestamp - bars[event_index - 2].timestamp
        == _BAR_DELTA
        and event.timestamp - bars[event_index - 1].timestamp == _BAR_DELTA
        and confirmation.trading_date == event.trading_date
        and confirmation.timestamp - event.timestamp == _BAR_DELTA
    )


def _normalization(
    bars: tuple[_Bar, ...],
    event_index: int,
    prior_dates: list[date],
    indices_by_date: dict[date, list[int]],
    slot_index: dict[tuple[date, time], int],
) -> _Normalization | None:
    event = bars[event_index]
    shock_return = _shock_return(bars, event_index)
    if shock_return is None:
        return None

    # Same-time-of-day normalization avoids mixing structurally different slots.
    shock_history: list[float] = []
    volume_history: list[float] = []
    for prior_date in reversed(prior_dates):
        equivalent_index = slot_index.get((prior_date, event.slot))
        if equivalent_index is not None:
            if len(shock_history) < 60:
                historical_return = _shock_return(bars, equivalent_index)
                if historical_return is not None:
                    shock_history.append(historical_return)
            if len(volume_history) < 20:
                historical_volume = bars[equivalent_index].volume
                if _positive_finite(historical_volume):
                    volume_history.append(historical_volume)
        if len(shock_history) == 60 and len(volume_history) == 20:
            break
    if len(shock_history) != 60 or len(volume_history) != 20:
        return None

    shock_median = float(median(shock_history))
    shock_mad = float(median(abs(value - shock_median) for value in shock_history))
    # 1.4826 makes MAD comparable to standard deviation under a normal baseline.
    scale = 1.4826 * shock_mad
    if not _positive_finite(scale):
        return None
    robust_z = (shock_return - shock_median) / scale
    volume_median = float(median(volume_history))
    if not _positive_finite(volume_median):
        return None
    relative_volume = event.volume / volume_median

    if len(prior_dates) < 20:
        return None
    turnovers = [
        _daily_turnover(bars, indices_by_date[prior_date])
        for prior_date in prior_dates[-20:]
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
    if relative_volume < _RELATIVE_VOLUME_THRESHOLD or median_turnover < 200_000_000:
        return None
    return _Normalization(*values)


def _shock_return(bars: tuple[_Bar, ...], index: int) -> float | None:
    if index < 2:
        return None
    start = bars[index - 2]
    middle = bars[index - 1]
    end = bars[index]
    if (
        start.trading_date != end.trading_date
        or middle.trading_date != end.trading_date
        or middle.timestamp - start.timestamp != _BAR_DELTA
        or end.timestamp - middle.timestamp != _BAR_DELTA
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
) -> tuple[_Level, float, float] | None:
    event = bars[event_index]
    levels = _candidate_levels(bars, side, prior_dates, indices_by_date)
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
            _at_least(penetration, 0.10 * event_atr)
            and _at_most(penetration, 0.75 * event_atr)
            and _at_least(reclaim, 0.05 * event_atr)
        ):
            return level, penetration, reclaim
    return None


def _candidate_levels(
    bars: tuple[_Bar, ...],
    side: Side,
    prior_dates: list[date],
    indices_by_date: dict[date, list[int]],
) -> tuple[_Level, ...]:
    """Return the single previous-day extreme eligible in v1.1.

    Session-so-far and confirmed swing levels were valid v1.0 research features
    but are intentionally excluded from v1.1. The previous-day extreme is known
    before the current session opens, preserving the original causal contract.
    """
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
            known_at=bar_available_at(bars[prior_indices[-1]].timestamp, 5),
        ),
    )


def _confirmation_passes(
    confirmation: _Bar, event: _Bar, level: float, side: Side
) -> bool:
    if side is Side.LONG:
        return confirmation.low >= level and confirmation.close > event.close
    return confirmation.high <= level and confirmation.close < event.close


def _within_signal_window(timestamp: datetime) -> bool:
    local_time = timestamp.astimezone(MARKET_TIMEZONE).time().replace(tzinfo=None)
    return _SIGNAL_TIME_START <= local_time <= _SIGNAL_TIME_END


def _feature_snapshot(
    event: _Bar,
    confirmation: _Bar,
    level: _Level,
    normalization: _Normalization,
    event_atr: float,
    penetration: float,
    reclaim_depth: float,
    stop_reference: float,
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
        "stop_buffer_atr": 0.10,
        "reward_r_multiple": 1.25,
        "maximum_hold_minutes": 30,
        "trailing_breakeven_trigger_r": 0.75,
        "trailing_breakeven_stop_r": 0.0,
        "trailing_profit_lock_trigger_r": 1.0,
        "trailing_profit_lock_stop_r": 0.25,
        "trailing_distance_r": 0.50,
        "trailing_hard_target_r": 1.25,
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
