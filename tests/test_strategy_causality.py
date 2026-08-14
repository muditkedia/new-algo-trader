from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from algo_trader import Side, Signal, SignalStatus
from algo_trader.data import bar_available_at
from algo_trader.strategies import (
    STRATEGY_CAUSALITY_GATE_VERSION,
    Strategy,
    StrategyCausalityReport,
    StrategyCausalityViolation,
    assert_strategy_prefix_invariant,
)

MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")


def make_candles(*, timezone: ZoneInfo = MARKET_TIMEZONE) -> pl.DataFrame:
    timestamps = [
        datetime(2025, 1, 2, 15, 10, tzinfo=MARKET_TIMEZONE),
        datetime(2025, 1, 2, 15, 15, tzinfo=MARKET_TIMEZONE),
        datetime(2025, 1, 2, 15, 25, tzinfo=MARKET_TIMEZONE),  # deliberate gap
        datetime(2025, 1, 2, 15, 30, tzinfo=MARKET_TIMEZONE),
        datetime(2025, 1, 3, 9, 15, tzinfo=MARKET_TIMEZONE),
        datetime(2025, 1, 3, 9, 20, tzinfo=MARKET_TIMEZONE),
        datetime(2025, 1, 3, 9, 25, tzinfo=MARKET_TIMEZONE),
        datetime(2025, 1, 3, 9, 30, tzinfo=MARKET_TIMEZONE),
        datetime(2025, 1, 3, 9, 35, tzinfo=MARKET_TIMEZONE),
        datetime(2025, 1, 3, 9, 40, tzinfo=MARKET_TIMEZONE),
    ]
    timestamps = [timestamp.astimezone(timezone) for timestamp in timestamps]
    closes = [100.0, 102.0, 101.0, 104.0, 103.0, 105.0, 107.0, 106.0, 108.0, 110.0]
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [close - 0.5 for close in closes],
            "high": [close + 1.0 for close in closes],
            "low": [close - 1.0 for close in closes],
            "close": closes,
            "volume": [1_000.0 + index * 100 for index in range(len(closes))],
            "symbol": ["TEST"] * len(closes),
        }
    )


class CausalRollingStrategy:
    strategy_id = "causal-rolling"
    strategy_version = "1.0.0"
    parameters: Mapping[str, Any] = {"lookback": 2, "comparison": "current-vs-prior-mean"}
    warmup_bars = 2

    def generate_signals(self, candles: pl.DataFrame) -> list[Signal]:
        rows = candles.rows(named=True)
        signals: list[Signal] = []
        for index in range(2, len(rows)):
            prior_mean = (rows[index - 2]["close"] + rows[index - 1]["close"]) / 2
            current = rows[index]
            signals.append(
                Signal(
                    strategy_id=self.strategy_id,
                    strategy_version=self.strategy_version,
                    symbol=current["symbol"],
                    timestamp=bar_available_at(current["timestamp"]),
                    side=Side.LONG if current["close"] >= prior_mean else Side.SHORT,
                    strategy_parameters=self.parameters,
                    feature_snapshot={
                        "source_bar_start": current["timestamp"],
                        "lagged_close": rows[index - 1]["close"],
                        "prior_mean": prior_mean,
                    },
                )
            )
        return signals


class EmptyStrategy:
    strategy_id = "empty"
    strategy_version = "1"
    parameters: Mapping[str, Any] = {}
    warmup_bars = 0

    def generate_signals(self, candles: pl.DataFrame) -> list[Signal]:
        return []


class LateSignalStrategy(EmptyStrategy):
    strategy_id = "late"

    def generate_signals(self, candles: pl.DataFrame) -> list[Signal]:
        if candles.height < 5:
            return []
        return [make_signal(self, candles.row(4, named=True), feature={"branch": "late"})]


def make_signal(
    strategy: Any,
    row: dict[str, Any],
    *,
    timestamp: datetime | None = None,
    side: Side = Side.LONG,
    feature: Mapping[str, Any] | None = None,
    strategy_id: str | None = None,
    strategy_version: str | None = None,
    symbol: str | None = None,
    status: SignalStatus = SignalStatus.GENERATED,
) -> Signal:
    return Signal(
        strategy_id=strategy.strategy_id if strategy_id is None else strategy_id,
        strategy_version=(
            strategy.strategy_version if strategy_version is None else strategy_version
        ),
        symbol=row["symbol"] if symbol is None else symbol,
        timestamp=bar_available_at(row["timestamp"]) if timestamp is None else timestamp,
        side=side,
        strategy_parameters=strategy.parameters,
        feature_snapshot={} if feature is None else feature,
        status=status,
    )


def assert_violation(strategy: Any, category: str, candles: pl.DataFrame | None = None) -> None:
    with pytest.raises(StrategyCausalityViolation, match=rf"\[{category}\].*prefix_length="):
        assert_strategy_prefix_invariant(strategy, make_candles() if candles is None else candles)


def test_causal_fixture_passes_exhaustive_multi_day_gapped_prefixes_without_mutation() -> None:
    strategy = CausalRollingStrategy()
    candles = make_candles()
    source_before = candles.clone()

    report = assert_strategy_prefix_invariant(strategy, candles)

    assert isinstance(strategy, Strategy)
    assert report == StrategyCausalityReport(
        gate_version=STRATEGY_CAUSALITY_GATE_VERSION,
        strategy_id="causal-rolling",
        strategy_version="1.0.0",
        symbol="TEST",
        timeframe_minutes=5,
        warmup_bars=2,
        row_count=10,
        first_tested_prefix_length=2,
        tested_prefix_count=9,
        full_signal_count=8,
        first_candle_timestamp=candles["timestamp"].item(0),
        last_candle_timestamp=candles["timestamp"].item(-1),
        last_information_available_at=bar_available_at(candles["timestamp"].item(-1)),
    )
    assert report.tested_prefix_count == report.row_count - report.first_tested_prefix_length + 1
    assert report.tested_prefix_count > 5
    assert candles.equals(source_before)
    assert report == assert_strategy_prefix_invariant(CausalRollingStrategy(), candles)


def test_next_session_and_extreme_future_rows_do_not_change_prior_signals() -> None:
    strategy = CausalRollingStrategy()
    candles = make_candles()
    first_session = candles.head(4)
    prior_signals = strategy.generate_signals(first_session)
    extreme = pl.DataFrame(
        {
            "timestamp": [datetime(2025, 1, 3, 9, 45, tzinfo=MARKET_TIMEZONE)],
            "open": [1_000_000.0],
            "high": [2_000_000.0],
            "low": [0.01],
            "close": [1_500_000.0],
            "volume": [1_000_000_000.0],
            "symbol": ["TEST"],
        }
    )
    extended = pl.concat([candles, extreme])

    assert strategy.generate_signals(candles)[: len(prior_signals)] == prior_signals
    assert strategy.generate_signals(extended)[: len(prior_signals)] == prior_signals
    assert_strategy_prefix_invariant(CausalRollingStrategy(), extended)


def test_zero_signal_and_signal_only_late_strategies_pass() -> None:
    zero_report = assert_strategy_prefix_invariant(EmptyStrategy(), make_candles())
    late_report = assert_strategy_prefix_invariant(LateSignalStrategy(), make_candles())

    assert zero_report.full_signal_count == 0
    assert late_report.full_signal_count == 1


def test_same_timestamp_distinct_signals_are_permitted_in_stable_order() -> None:
    class SameTimestampStrategy(EmptyStrategy):
        strategy_id = "same-timestamp"

        def generate_signals(self, candles: pl.DataFrame) -> list[Signal]:
            row = candles.row(0, named=True)
            return [
                make_signal(self, row, side=Side.LONG, feature={"sequence": 1}),
                make_signal(self, row, side=Side.SHORT, feature={"sequence": 2}),
            ]

    report = assert_strategy_prefix_invariant(SameTimestampStrategy(), make_candles())

    assert report.full_signal_count == 2


def test_non_kolkata_aware_timezone_and_equal_economic_future_rows_are_tested() -> None:
    candles = make_candles(timezone=UTC)
    final_row = candles.row(-1, named=True)
    repeated_values = pl.DataFrame(
        {
            **{column: [final_row[column]] for column in candles.columns if column != "timestamp"},
            "timestamp": [final_row["timestamp"] + timedelta(minutes=5)],
        }
    ).select(candles.columns)
    extended = pl.concat([candles, repeated_values])

    report = assert_strategy_prefix_invariant(CausalRollingStrategy(), extended)

    assert report.first_candle_timestamp.tzinfo is not None
    assert report.tested_prefix_count == 10


class NextRowLeak(EmptyStrategy):
    strategy_id = "next-row-leak"

    def generate_signals(self, candles: pl.DataFrame) -> list[Signal]:
        rows = candles.rows(named=True)
        return [
            make_signal(
                self,
                rows[index],
                side=Side.LONG if rows[index + 1]["close"] > rows[index]["close"] else Side.SHORT,
                feature={"next_close": rows[index + 1]["close"]},
            )
            for index in range(len(rows) - 1)
        ]


class GlobalStatisticLeak(EmptyStrategy):
    strategy_id = "global-statistic-leak"

    def generate_signals(self, candles: pl.DataFrame) -> list[Signal]:
        row = candles.row(0, named=True)
        return [
            make_signal(
                self,
                row,
                side=Side.LONG,
                feature={"full_frame_mean": sum(candles["close"]) / candles.height},
            )
        ]


class CenteredWindowLeak(EmptyStrategy):
    strategy_id = "centered-window-leak"

    def generate_signals(self, candles: pl.DataFrame) -> list[Signal]:
        rows = candles.rows(named=True)
        return [
            make_signal(
                self,
                rows[index],
                feature={
                    "centered_mean": (
                        rows[index - 1]["close"]
                        + rows[index]["close"]
                        + rows[index + 1]["close"]
                    )
                    / 3
                },
            )
            for index in range(1, len(rows) - 1)
        ]


@pytest.mark.parametrize(
    ("strategy", "case_id"),
    [
        (NextRowLeak(), "direct-next-row"),
        (GlobalStatisticLeak(), "full-frame-statistic-snapshot"),
        (CenteredWindowLeak(), "centered-window"),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_future_row_feature_leaks_fail_prefix_invariance(strategy: Any, case_id: str) -> None:
    del case_id
    assert_violation(strategy, "PREFIX_INVARIANCE")


class RetroactiveInsertion(EmptyStrategy):
    strategy_id = "retroactive-insertion"

    def generate_signals(self, candles: pl.DataFrame) -> list[Signal]:
        return [] if candles.height == 1 else [make_signal(self, candles.row(0, named=True))]


class RetroactiveDeletion(EmptyStrategy):
    strategy_id = "retroactive-deletion"

    def generate_signals(self, candles: pl.DataFrame) -> list[Signal]:
        return [make_signal(self, candles.row(0, named=True))] if candles.height == 1 else []


class SnapshotAlteration(GlobalStatisticLeak):
    strategy_id = "snapshot-alteration"


@pytest.mark.parametrize(
    "strategy",
    [RetroactiveInsertion(), RetroactiveDeletion(), SnapshotAlteration()],
    ids=["insertion", "deletion", "snapshot-only-alteration"],
)
def test_retroactive_output_changes_fail(strategy: Any) -> None:
    assert_violation(strategy, "PREFIX_INVARIANCE")


def test_completed_bar_cannot_be_timestamped_at_its_start() -> None:
    class StartTimestampStrategy(EmptyStrategy):
        strategy_id = "start-timestamp"

        def generate_signals(self, candles: pl.DataFrame) -> list[Signal]:
            row = candles.row(0, named=True)
            return [make_signal(self, row, timestamp=row["timestamp"])]

    assert_violation(StartTimestampStrategy(), "EARLY_SIGNAL_TIMESTAMP")


def test_future_dated_signal_beyond_prefix_cutoff_fails() -> None:
    class FutureTimestampStrategy(EmptyStrategy):
        strategy_id = "future-timestamp"

        def generate_signals(self, candles: pl.DataFrame) -> list[Signal]:
            row = candles.row(-1, named=True)
            return [
                make_signal(
                    self,
                    row,
                    timestamp=bar_available_at(row["timestamp"]) + timedelta(seconds=1),
                )
            ]

    assert_violation(FutureTimestampStrategy(), "FUTURE_SIGNAL_TIMESTAMP")


def test_knowledge_prefix_rejects_use_of_incomplete_0920_bar_at_0922() -> None:
    class MidBarLeak(EmptyStrategy):
        strategy_id = "mid-bar-leak"

        def generate_signals(self, candles: pl.DataFrame) -> list[Signal]:
            if candles.height < 2:
                return []
            row = candles.row(1, named=True)
            return [
                make_signal(
                    self,
                    row,
                    timestamp=row["timestamp"] + timedelta(minutes=2),
                    feature={"unavailable_close": row["close"]},
                )
            ]

    candles = make_candles().head(3).with_columns(
        pl.Series(
            "timestamp",
            [
                datetime(2025, 1, 2, 9, 15, tzinfo=MARKET_TIMEZONE),
                datetime(2025, 1, 2, 9, 20, tzinfo=MARKET_TIMEZONE),
                datetime(2025, 1, 2, 9, 25, tzinfo=MARKET_TIMEZONE),
            ],
        )
    )

    assert_violation(MidBarLeak(), "KNOWLEDGE_PREFIX", candles)


def test_hidden_state_changes_repeated_output() -> None:
    class StatefulStrategy(EmptyStrategy):
        strategy_id = "stateful"

        def __init__(self) -> None:
            self.calls = 0

        def generate_signals(self, candles: pl.DataFrame) -> list[Signal]:
            self.calls += 1
            return [
                make_signal(
                    self,
                    candles.row(0, named=True),
                    feature={"hidden_call_counter": self.calls},
                )
            ]

    assert_violation(StatefulStrategy(), "NONDETERMINISTIC_OUTPUT")


def test_cross_prefix_state_accumulation_changes_full_output() -> None:
    class CrossPrefixStateStrategy(EmptyStrategy):
        strategy_id = "cross-prefix-state"

        def __init__(self) -> None:
            self.shorter_prefix_seen = False

        def generate_signals(self, candles: pl.DataFrame) -> list[Signal]:
            if candles.height < 10:
                self.shorter_prefix_seen = True
                return []
            return [
                make_signal(
                    self,
                    candles.row(0, named=True),
                    feature={"shorter_prefix_seen": self.shorter_prefix_seen},
                )
            ]

    assert_violation(CrossPrefixStateStrategy(), "NONDETERMINISTIC_OUTPUT")


def test_strategy_parameter_mutation_is_rejected() -> None:
    class ParameterMutatingStrategy(EmptyStrategy):
        strategy_id = "parameter-mutator"

        def __init__(self) -> None:
            self.parameters = {"calls": 0}

        def generate_signals(self, candles: pl.DataFrame) -> list[Signal]:
            self.parameters["calls"] += 1
            return []

    assert_violation(ParameterMutatingStrategy(), "STRATEGY_CONFIGURATION_MUTATION")


def test_strategy_candle_prefix_mutation_is_rejected() -> None:
    class InputMutatingStrategy(EmptyStrategy):
        strategy_id = "input-mutator"

        def generate_signals(self, candles: pl.DataFrame) -> list[Signal]:
            candles.__init__(candles.head(0))
            return []

    assert_violation(InputMutatingStrategy(), "INPUT_MUTATION")


class OutputCaseStrategy(EmptyStrategy):
    strategy_id = "output-case"

    def __init__(self, case: str) -> None:
        self.case = case

    def generate_signals(self, candles: pl.DataFrame) -> Any:
        row = candles.row(0, named=True)
        valid = make_signal(self, row)
        if self.case == "non-list":
            return (valid,)
        if self.case == "non-signal":
            return [object()]
        if self.case == "wrong-strategy-id":
            return [make_signal(self, row, strategy_id="other")]
        if self.case == "wrong-version":
            return [make_signal(self, row, strategy_version="other")]
        if self.case == "wrong-symbol":
            return [make_signal(self, row, symbol="OTHER")]
        if self.case == "wrong-status":
            return [make_signal(self, row, status=SignalStatus.EXECUTED)]
        if self.case == "duplicate":
            return [valid, make_signal(self, row)]
        if self.case == "descending" and candles.height >= 2:
            return [make_signal(self, candles.row(1, named=True)), valid]
        return [valid]


@pytest.mark.parametrize(
    ("case", "category"),
    [
        ("non-list", "OUTPUT_CONTRACT"),
        ("non-signal", "OUTPUT_CONTRACT"),
        ("wrong-strategy-id", "SIGNAL_METADATA"),
        ("wrong-version", "SIGNAL_METADATA"),
        ("wrong-symbol", "SIGNAL_METADATA"),
        ("wrong-status", "SIGNAL_METADATA"),
        ("descending", "SIGNAL_ORDER"),
        ("duplicate", "DUPLICATE_SIGNAL"),
    ],
)
def test_output_contract_failures_are_categorized(case: str, category: str) -> None:
    assert_violation(OutputCaseStrategy(case), category)


class ConfigurableEmptyStrategy(EmptyStrategy):
    def __init__(
        self,
        *,
        strategy_id: Any = "configurable",
        strategy_version: Any = "1",
        warmup_bars: Any = 0,
        parameters: Any = None,
    ) -> None:
        self.strategy_id = strategy_id
        self.strategy_version = strategy_version
        self.warmup_bars = warmup_bars
        self.parameters = {} if parameters is None else parameters


@pytest.mark.parametrize(
    ("overrides", "error_type", "match"),
    [
        ({"strategy_id": "  "}, ValueError, "strategy_id.*nonempty"),
        ({"strategy_version": ""}, ValueError, "strategy_version.*nonempty"),
        ({"warmup_bars": True}, TypeError, "warmup_bars.*not Boolean"),
        ({"warmup_bars": -1}, ValueError, "warmup_bars.*non-negative"),
        ({"parameters": [1, 2]}, TypeError, "parameters.*Mapping"),
    ],
)
def test_strategy_input_contract_boundaries(
    overrides: dict[str, Any], error_type: type[Exception], match: str
) -> None:
    with pytest.raises(error_type, match=match):
        assert_strategy_prefix_invariant(ConfigurableEmptyStrategy(**overrides), make_candles())


def test_non_strategy_object_is_rejected() -> None:
    with pytest.raises(TypeError, match="Strategy protocol"):
        assert_strategy_prefix_invariant(object(), make_candles())  # type: ignore[arg-type]


def test_invalid_empty_and_short_candle_inputs_fail_clearly() -> None:
    with pytest.raises(ValueError, match="missing required candle column.*volume"):
        assert_strategy_prefix_invariant(EmptyStrategy(), make_candles().drop("volume"))

    with pytest.raises(ValueError, match="must not be empty"):
        assert_strategy_prefix_invariant(EmptyStrategy(), make_candles().clear())

    with pytest.raises(ValueError, match="at least two eligible prefixes"):
        assert_strategy_prefix_invariant(
            ConfigurableEmptyStrategy(warmup_bars=10), make_candles()
        )
