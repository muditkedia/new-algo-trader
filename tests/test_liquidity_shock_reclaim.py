from __future__ import annotations

from datetime import date, datetime, time, timedelta
from types import MappingProxyType
from zoneinfo import ZoneInfo

import polars as pl
import pytest

import algo_trader.strategies.liquidity_shock_reclaim as strategy_module
from algo_trader import Side, SignalStatus
from algo_trader.data import bar_available_at
from algo_trader.strategies import (
    LiquidityShockReclaimStrategy,
    Strategy,
    assert_strategy_prefix_invariant,
)

IST = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")

EXPECTED_PARAMETERS = {
    "timeframe_minutes": 5,
    "signal_time_start": "09:40",
    "signal_time_end": "14:35",
    "shock_horizon_bars": 2,
    "shock_history_sessions": 60,
    "shock_robust_z_threshold": 3.0,
    "mad_consistency_scale": 1.4826,
    "volume_history_sessions": 20,
    "relative_volume_threshold": 2.0,
    "liquidity_history_sessions": 20,
    "minimum_median_daily_turnover_rupees": 200_000_000,
    "atr_period": 14,
    "swing_left_bars": 2,
    "swing_right_bars": 2,
    "minimum_level_age_minutes": 10,
    "minimum_penetration_atr": 0.10,
    "maximum_penetration_atr": 0.75,
    "minimum_reclaim_atr": 0.05,
    "confirmation_bars": 1,
    "stop_buffer_atr": 0.10,
    "reward_r_multiple": 1.5,
    "maximum_hold_minutes": 30,
    "latest_exit_time": "15:10",
    "trailing_breakeven_trigger_r": 0.75,
    "trailing_breakeven_stop_r": 0.0,
    "trailing_profit_lock_trigger_r": 1.0,
    "trailing_profit_lock_stop_r": 0.25,
    "trailing_distance_r": 0.50,
    "trailing_hard_target_r": 1.5,
    "max_signals_per_symbol_per_day": 1,
}


def _timestamp(day: date, index: int, start: time = time(9, 15)) -> datetime:
    return datetime.combine(day, start, tzinfo=IST) + timedelta(minutes=5 * index)


def _row(
    timestamp: datetime,
    *,
    close: float,
    volume: float,
    high: float | None = None,
    low: float | None = None,
    open_: float | None = None,
) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "open": close if open_ is None else open_,
        "high": close + 1.0 if high is None else high,
        "low": close - 1.0 if low is None else low,
        "close": close,
        "volume": volume,
        "symbol": "TEST",
    }


def make_fixture(
    *,
    side: Side = Side.LONG,
    history_sessions: int = 60,
    event_index: int = 59,
    prior_bars: int = 74,
    level_mode: str = "PREVIOUS_DAY",
    zero_mad: bool = False,
    history_return_step: float = 0.0002,
    valid_volume_sessions: int = 60,
    event_volume_multiple: float = 2.5,
    history_volume: float = 30_000.0,
    confirmation_gap_minutes: int = 5,
    confirmation_low: float | None = None,
    confirmation_high: float | None = None,
    confirmation_close: float | None = None,
    event_low: float | None = None,
    event_high: float | None = None,
    event_close: float | None = None,
    start: time = time(9, 15),
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    first_day = date(2025, 1, 1)
    for session in range(history_sessions):
        day = first_day + timedelta(days=session)
        session_rows = [
            _row(
                _timestamp(day, index, start),
                close=100.0,
                volume=history_volume,
            )
            for index in range(prior_bars)
        ]
        if event_index < prior_bars:
            shock = 0.0 if zero_mad else ((session % 10) - 4.5) * history_return_step
            historical_close = 100.0 * (1.0 + shock)
            session_rows[event_index] = _row(
                _timestamp(day, event_index, start),
                close=historical_close,
                volume=(
                    history_volume
                    if session >= history_sessions - valid_volume_sessions
                    else 0.0
                ),
            )
        rows.extend(session_rows)

    current_day = first_day + timedelta(days=history_sessions)
    current_count = event_index + 2
    base = 105.0 if side is Side.LONG else 95.0
    current = [
        _row(_timestamp(current_day, index, start), close=base, volume=history_volume)
        for index in range(current_count)
    ]

    prior_start = (history_sessions - 1) * prior_bars
    if level_mode in {"PREVIOUS_DAY", "PRIORITY"}:
        if side is Side.LONG:
            rows[prior_start + 10]["low"] = 96.0
        else:
            rows[prior_start + 10]["high"] = 104.0
    else:
        if side is Side.LONG:
            rows[prior_start + 10]["low"] = 90.0
        else:
            rows[prior_start + 10]["high"] = 110.0

    if level_mode in {"SESSION", "PRIORITY"}:
        if side is Side.LONG:
            current[10]["low"] = 96.0
        else:
            current[10]["high"] = 104.0
    elif level_mode == "SESSION_NEW":
        if side is Side.LONG:
            current[event_index - 2]["low"] = 96.0
        else:
            current[event_index - 2]["high"] = 104.0
    if level_mode in {"SWING", "SWING_SINGLE", "SWING_PLATEAU", "PRIORITY"}:
        pivot = min(40, event_index - 8)
        older_pivot = min(20, pivot - 8)
        if side is Side.LONG:
            current[5]["low"] = 95.7
            if level_mode != "SWING_SINGLE":
                for index in (
                    older_pivot - 2,
                    older_pivot - 1,
                    older_pivot + 1,
                    older_pivot + 2,
                ):
                    current[index]["low"] = 97.3
                current[older_pivot]["low"] = 96.2
            for index in (pivot - 2, pivot - 1, pivot + 1, pivot + 2):
                current[index]["low"] = 97.0
            current[pivot]["low"] = 96.0
            if level_mode == "SWING_PLATEAU":
                current[pivot + 1]["low"] = 96.0
        else:
            current[5]["high"] = 104.3
            if level_mode != "SWING_SINGLE":
                for index in (
                    older_pivot - 2,
                    older_pivot - 1,
                    older_pivot + 1,
                    older_pivot + 2,
                ):
                    current[index]["high"] = 102.7
                current[older_pivot]["high"] = 103.8
            for index in (pivot - 2, pivot - 1, pivot + 1, pivot + 2):
                current[index]["high"] = 103.0
            current[pivot]["high"] = 104.0
            if level_mode == "SWING_PLATEAU":
                current[pivot + 1]["high"] = 104.0
    elif level_mode == "SWING_UNCONFIRMED":
        pivot = event_index - 3
        if side is Side.LONG:
            current[5]["low"] = 95.7
            for index in (pivot - 2, pivot - 1, pivot + 1, pivot + 2):
                current[index]["low"] = 97.0
            current[pivot]["low"] = 96.0
        else:
            current[5]["high"] = 104.3
            for index in (pivot - 2, pivot - 1, pivot + 1, pivot + 2):
                current[index]["high"] = 103.0
            current[pivot]["high"] = 104.0

    if side is Side.LONG:
        selected_event_low = 95.5 if event_low is None else event_low
        selected_event_high = 100.0 if event_high is None else event_high
        selected_event_close = 96.5 if event_close is None else event_close
        selected_confirmation_low = 96.1 if confirmation_low is None else confirmation_low
        selected_confirmation_high = 97.5 if confirmation_high is None else confirmation_high
        selected_confirmation_close = (
            97.0 if confirmation_close is None else confirmation_close
        )
    else:
        selected_event_low = 100.0 if event_low is None else event_low
        selected_event_high = 104.5 if event_high is None else event_high
        selected_event_close = 103.5 if event_close is None else event_close
        selected_confirmation_low = 102.5 if confirmation_low is None else confirmation_low
        selected_confirmation_high = 103.9 if confirmation_high is None else confirmation_high
        selected_confirmation_close = (
            103.0 if confirmation_close is None else confirmation_close
        )
    current[event_index] = _row(
        _timestamp(current_day, event_index, start),
        open_=100.0,
        high=selected_event_high,
        low=selected_event_low,
        close=selected_event_close,
        volume=history_volume * event_volume_multiple,
    )
    current[event_index + 1] = _row(
        _timestamp(current_day, event_index, start)
        + timedelta(minutes=confirmation_gap_minutes),
        open_=selected_event_close,
        high=selected_confirmation_high,
        low=selected_confirmation_low,
        close=selected_confirmation_close,
        volume=history_volume,
    )
    rows.extend(current)
    return pl.DataFrame(rows)


def _signals(candles: pl.DataFrame):
    return LiquidityShockReclaimStrategy().generate_signals(candles)


def _constant_atr(monkeypatch: pytest.MonkeyPatch, value: float = 2.0) -> None:
    monkeypatch.setattr(
        "algo_trader.strategies.liquidity_shock_reclaim.atr",
        lambda candles, period: pl.Series("atr", [value] * candles.height),
    )


def test_identity_protocol_and_exact_immutable_parameters() -> None:
    strategy = LiquidityShockReclaimStrategy()

    assert isinstance(strategy, Strategy)
    assert strategy.strategy_id == "liquidity-shock-exhaustion-reclaim"
    assert strategy.strategy_version == "1.0.0"
    assert strategy.warmup_bars == 4500
    assert strategy.parameters == EXPECTED_PARAMETERS
    assert isinstance(strategy.parameters, MappingProxyType)
    with pytest.raises(TypeError):
        strategy.parameters["atr_period"] = 99  # type: ignore[index]


@pytest.mark.parametrize("side", [Side.LONG, Side.SHORT])
def test_canonical_long_and_short_previous_day_examples(side: Side) -> None:
    signals = _signals(make_fixture(side=side))

    assert len(signals) == 1
    assert signals[0].side is side
    assert signals[0].status is SignalStatus.GENERATED
    assert signals[0].feature_snapshot["level_type"] == (
        "PDL" if side is Side.LONG else "PDH"
    )


def test_shock_requires_consecutive_same_session_bars_and_60_samples() -> None:
    insufficient = make_fixture(history_sessions=59)
    assert _signals(insufficient) == []

    gapped = make_fixture().with_columns(
        pl.when(pl.arange(0, pl.len()) == 60 * 74 + 58)
        .then(pl.col("timestamp") - timedelta(minutes=1))
        .otherwise(pl.col("timestamp"))
        .alias("timestamp")
    )
    assert _signals(gapped) == []


def test_robust_z_below_threshold_and_zero_mad_reject() -> None:
    assert _signals(make_fixture(history_return_step=0.01)) == []
    assert _signals(make_fixture(zero_mad=True)) == []


def test_volume_history_rvol_and_liquidity_filters() -> None:
    assert _signals(make_fixture(valid_volume_sessions=19)) == []
    assert _signals(make_fixture(event_volume_multiple=1.99)) == []
    assert _signals(
        make_fixture(history_volume=10_000.0, event_volume_multiple=2.5)
    ) == []


@pytest.mark.parametrize(
    ("side", "mode", "expected_type"),
    [
        (Side.LONG, "SESSION", "SESSION_LOW"),
        (Side.SHORT, "SESSION", "SESSION_HIGH"),
        (Side.LONG, "SWING", "SWING_LOW"),
        (Side.SHORT, "SWING", "SWING_HIGH"),
    ],
)
def test_aged_session_and_most_recent_confirmed_swing_levels(
    side: Side, mode: str, expected_type: str
) -> None:
    signals = _signals(make_fixture(side=side, level_mode=mode))

    assert len(signals) == 1
    assert signals[0].feature_snapshot["level_type"] == expected_type
    if mode == "SWING":
        assert signals[0].feature_snapshot["level_price"] == (
            96.0 if side is Side.LONG else 104.0
        )


@pytest.mark.parametrize("side", [Side.LONG, Side.SHORT])
def test_swing_window_requires_consecutive_five_minute_bars(side: Side) -> None:
    complete = make_fixture(side=side, level_mode="SWING_SINGLE")
    complete_signals = _signals(complete)

    assert len(complete_signals) == 1
    assert complete_signals[0].feature_snapshot["level_type"] == (
        "SWING_LOW" if side is Side.LONG else "SWING_HIGH"
    )

    current_start = 60 * 74
    gapped = (
        complete.with_row_index("row_number")
        .filter(pl.col("row_number") != current_start + 41)
        .drop("row_number")
    )

    assert _signals(gapped) == []


def test_date_ordinal_lookup_preserves_exact_representative_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candles = make_fixture()
    expected = _signals(candles)
    original_session_index = strategy_module._session_index

    def session_index_with_linear_date_lookup(bars):
        dates, indices_by_date, slot_index, _ = original_session_index(bars)
        return (
            dates,
            indices_by_date,
            slot_index,
            {trading_date: dates.index(trading_date) for trading_date in dates},
        )

    monkeypatch.setattr(
        strategy_module, "_session_index", session_index_with_linear_date_lookup
    )

    assert _signals(candles) == expected


def test_new_session_level_and_equal_swing_plateau_are_unavailable() -> None:
    assert _signals(make_fixture(level_mode="SESSION_NEW")) == []
    assert _signals(make_fixture(level_mode="SWING_UNCONFIRMED")) == []
    assert _signals(make_fixture(level_mode="SWING_PLATEAU")) == []


def test_level_priority_is_previous_day_then_session_then_swing() -> None:
    signal = _signals(make_fixture(level_mode="PRIORITY"))[0]
    assert signal.feature_snapshot["level_type"] == "PDL"


@pytest.mark.parametrize(
    ("event_low", "event_close", "accepted"),
    [
        (95.8, 96.1, True),
        (95.800001, 96.1, False),
        (94.5, 96.1, True),
        (94.499999, 96.1, False),
        (95.5, 96.1, True),
        (95.5, 96.099999, False),
    ],
    ids=[
        "min-penetration",
        "below-min",
        "max-penetration",
        "above-max",
        "min-reclaim",
        "below-reclaim",
    ],
)
def test_penetration_and_reclaim_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    event_low: float,
    event_close: float,
    accepted: bool,
) -> None:
    _constant_atr(monkeypatch)
    candles = make_fixture(
        event_low=event_low,
        event_close=event_close,
        confirmation_close=event_close + 0.2,
    )
    assert bool(_signals(candles)) is accepted


def test_delayed_reclaim_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _constant_atr(monkeypatch)
    candles = make_fixture(event_close=95.9, confirmation_close=96.5)
    assert _signals(candles) == []


@pytest.mark.parametrize(
    ("side", "overrides"),
    [
        (Side.LONG, {"confirmation_low": 95.99}),
        (Side.LONG, {"confirmation_close": 96.5}),
        (Side.SHORT, {"confirmation_high": 104.01}),
        (Side.SHORT, {"confirmation_close": 103.5}),
    ],
)
def test_confirmation_geometry_is_strict(side: Side, overrides: dict[str, float]) -> None:
    assert _signals(make_fixture(side=side, **overrides)) == []


def test_nonconsecutive_confirmation_and_signal_window_reject() -> None:
    assert _signals(make_fixture(confirmation_gap_minutes=10)) == []
    assert _signals(make_fixture(event_index=2)) == []
    assert _signals(make_fixture(event_index=63)) == []


def test_signal_window_boundaries_and_confirmation_level_equality_are_inclusive() -> None:
    assert _signals(make_fixture(event_index=3))[0].timestamp.time() == time(9, 40)
    assert _signals(make_fixture(event_index=62))[0].timestamp.time() == time(14, 35)
    assert len(_signals(make_fixture(confirmation_low=96.0))) == 1
    assert len(_signals(make_fixture(side=Side.SHORT, confirmation_high=104.0))) == 1


def test_signal_timestamp_timezone_and_feature_exit_metadata() -> None:
    candles = make_fixture().with_columns(pl.col("timestamp").dt.convert_time_zone("UTC"))
    signal = _signals(candles)[0]
    feature = signal.feature_snapshot

    assert signal.timestamp == bar_available_at(feature["confirmation_bar_start"])
    assert signal.timestamp.tzinfo == IST
    required = {
        "event_bar_start",
        "confirmation_bar_start",
        "level_type",
        "level_known_at",
        "shock_return",
        "shock_history_median",
        "shock_history_mad",
        "shock_robust_z",
        "event_volume",
        "historical_slot_volume_median",
        "relative_volume",
        "median_daily_turnover",
        "atr",
        "level_price",
        "penetration",
        "penetration_atr",
        "reclaim_depth",
        "reclaim_atr",
        "event_open",
        "event_high",
        "event_low",
        "event_close",
        "confirmation_open",
        "confirmation_high",
        "confirmation_low",
        "confirmation_close",
        "confirmation_return_from_event_close",
        "stop_reference_price",
        "stop_buffer_atr",
        "reward_r_multiple",
        "maximum_hold_minutes",
        "trailing_breakeven_trigger_r",
        "trailing_breakeven_stop_r",
        "trailing_profit_lock_trigger_r",
        "trailing_profit_lock_stop_r",
        "trailing_distance_r",
        "trailing_hard_target_r",
    }
    assert required <= feature.keys()
    assert feature["stop_reference_price"] == pytest.approx(
        feature["event_low"] - 0.10 * feature["atr"]
    )
    assert {
        key: feature[key]
        for key in (
            "trailing_breakeven_trigger_r",
            "trailing_breakeven_stop_r",
            "trailing_profit_lock_trigger_r",
            "trailing_profit_lock_stop_r",
            "trailing_distance_r",
            "trailing_hard_target_r",
        )
    } == {
        "trailing_breakeven_trigger_r": 0.75,
        "trailing_breakeven_stop_r": 0.0,
        "trailing_profit_lock_trigger_r": 1.0,
        "trailing_profit_lock_stop_r": 0.25,
        "trailing_distance_r": 0.50,
        "trailing_hard_target_r": 1.5,
    }
    assert "target_price" not in feature
    assert "trailing_stop_price" not in feature
    assert "target_price" not in signal.strategy_parameters


def test_only_first_signal_per_day_input_immutability_and_future_invariance() -> None:
    first = make_fixture()
    rows = first.rows(named=True)
    current_start = 60 * 74
    current_day = rows[current_start]["timestamp"].date()
    for session in range(60):
        historical_close = 100.0 * (
            1.0 + ((session % 10) - 4.5) * 0.0002
        )
        historical_event = rows[session * 74 + 62]
        historical_event.update(
            {
                "open": historical_close,
                "high": historical_close + 1.0,
                "low": historical_close - 1.0,
                "close": historical_close,
            }
        )
    for index in (61,):
        rows.append(
            _row(_timestamp(current_day, index), close=105.0, volume=30_000.0)
        )
    rows.extend(
        [
            _row(
                _timestamp(current_day, 62),
                open_=100.0,
                high=100.0,
                low=95.5,
                close=96.5,
                volume=75_000.0,
            ),
            _row(
                _timestamp(current_day, 63),
                open_=96.5,
                high=97.5,
                low=96.1,
                close=97.0,
                volume=30_000.0,
            ),
        ]
    )
    combined = pl.DataFrame(rows)
    before = combined.clone()
    first_signals = _signals(first)
    combined_signals = _signals(combined)

    assert len(combined_signals) == 1
    assert combined_signals == first_signals
    assert combined.equals(before)
    assert _signals(combined.head(first.height)) == first_signals

    without_first = combined.with_columns(
        pl.when(pl.arange(0, pl.len()) == current_start + 60)
        .then(pl.lit(95.9))
        .otherwise(pl.col("low"))
        .alias("low")
    )
    later_signal = _signals(without_first)
    assert len(later_signal) == 1
    assert later_signal[0].feature_snapshot["event_bar_start"] == _timestamp(
        current_day, 62
    )


def test_representative_fixture_passes_exhaustive_prefix_causality_gate() -> None:
    candles = make_fixture()
    assert candles.height == 4501

    report = assert_strategy_prefix_invariant(
        LiquidityShockReclaimStrategy(), candles
    )

    assert report.full_signal_count == 1
    assert report.first_tested_prefix_length == 4500
    assert report.tested_prefix_count == 2
