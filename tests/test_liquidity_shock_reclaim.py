from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, time, timedelta
from types import MappingProxyType
from zoneinfo import ZoneInfo

import polars as pl
import pytest

import algo_trader.strategies.liquidity_shock_reclaim as strategy_module
from algo_trader import Side, SignalStatus
from algo_trader.data import SymbolCoverage, bar_available_at
from algo_trader.strategies import (
    LiquidityShockReclaimConfig,
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
    "relative_volume_threshold": 12.0,
    "liquidity_history_sessions": 20,
    "minimum_median_daily_turnover_rupees": 200_000_000,
    "atr_period": 14,
    "level_policy": "PRIOR_DAY_EXTREME_ONLY",
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
    event_volume_multiple: float = 15.0,
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
    assert strategy.strategy_version == "1.1.0"
    assert strategy.warmup_bars == 4500
    assert strategy.parameters == EXPECTED_PARAMETERS
    assert isinstance(strategy.parameters, MappingProxyType)
    with pytest.raises(TypeError):
        strategy.parameters["atr_period"] = 99  # type: ignore[index]


@pytest.mark.parametrize(
    "updates",
    [
        {"shock_robust_z_threshold": 1_000_000_000.0},
        {"relative_volume_threshold": 100.0},
        {"minimum_median_daily_turnover_rupees": 10_000_000_000},
        {"minimum_penetration_atr": 10.0},
        {"minimum_reclaim_atr": 10.0},
        {"signal_time_end": time(10, 0)},
    ],
    ids=("shock-z", "rvol", "turnover", "penetration", "reclaim", "signal-window"),
)
def test_optimizer_eligible_config_changes_real_eligibility(
    updates: dict[str, object],
) -> None:
    baseline = LiquidityShockReclaimStrategy().generate_signals(make_fixture())
    configured = LiquidityShockReclaimStrategy(
        replace(LiquidityShockReclaimConfig(), **updates)
    ).generate_signals(make_fixture())
    assert len(baseline) == 1
    assert configured == []


def test_exit_config_changes_behavioral_signal_metadata_from_same_source() -> None:
    config = replace(
        LiquidityShockReclaimConfig(),
        stop_buffer_atr=0.2,
        hard_target_r=1.4,
    )
    signal = LiquidityShockReclaimStrategy(config).generate_signals(make_fixture())[0]
    assert signal.feature_snapshot["stop_buffer_atr"] == 0.2
    assert signal.feature_snapshot["trailing_hard_target_r"] == 1.4
    assert signal.strategy_parameters["reward_r_multiple"] == 1.4
    assert signal.strategy_parameters["trailing_hard_target_r"] == 1.4


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
    assert _signals(make_fixture(event_volume_multiple=11.99)) == []
    assert len(_signals(make_fixture(event_volume_multiple=12.0))) == 1
    assert _signals(
        make_fixture(history_volume=10_000.0, event_volume_multiple=15.0)
    ) == []


@pytest.mark.parametrize(
    ("side", "mode"),
    [
        (Side.LONG, "SESSION"),
        (Side.SHORT, "SESSION"),
        (Side.LONG, "SWING"),
        (Side.SHORT, "SWING"),
        (Side.LONG, "SWING_SINGLE"),
        (Side.SHORT, "SWING_SINGLE"),
    ],
)
def test_v11_rejects_non_prior_day_structural_levels(side: Side, mode: str) -> None:
    assert _signals(make_fixture(side=side, level_mode=mode)) == []


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


def test_prior_day_level_remains_selected_when_other_structures_overlap() -> None:
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
        "trailing_hard_target_r": 1.25,
    }
    assert "target_price" not in feature
    assert "trailing_stop_price" not in feature
    assert "target_price" not in signal.strategy_parameters


def test_only_first_signal_per_day_input_immutability_and_future_invariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This test isolates daily signal gating/future invariance from ATR-path changes.
    _constant_atr(monkeypatch, 2.0)
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
                # v1.1 requires RVOL >= 12x; make this synthetic setup 15x.
                volume=30_000.0 * 15.0,
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


def test_runner_real_causality_preflight_is_non_vacuous() -> None:
    from scripts.run_strategy1_development_backtest import run_real_causality_gate

    candles = make_fixture()
    last = candles.row(-1, named=True)
    future = []
    for offset in range(1, 9):
        timestamp = last["timestamp"] + timedelta(minutes=5 * offset)
        future.append(
            {
                **last,
                "timestamp": timestamp,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 30_000.0,
            }
        )
    extended = pl.concat((candles, pl.DataFrame(future, schema=candles.schema)))

    class Store:
        def load_candles(self, symbol, start, end):
            assert symbol == "TEST"
            return extended.clone()

    report = run_real_causality_gate(
        store=Store(),  # type: ignore[arg-type]
        coverages=(
            SymbolCoverage(
                symbol="TEST",
                first_timestamp=extended["timestamp"].item(0),
                last_timestamp=extended["timestamp"].item(-1),
                row_count=extended.height,
            ),
        ),
        strategy=LiquidityShockReclaimStrategy(),
        allowed_end_exclusive=extended["timestamp"].item(-1).date() + timedelta(days=1),
    )
    assert report.full_signal_count >= 1
    assert report.tested_prefix_count > 2
