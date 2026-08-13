from collections.abc import Mapping
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from algo_trader import Side, Signal, SignalStatus
from algo_trader.strategies import Strategy, bar_available_at, validate_strategy_input

MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")


def make_candles() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 2, 9, 15, tzinfo=MARKET_TIMEZONE),
                datetime(2024, 1, 2, 9, 20, tzinfo=MARKET_TIMEZONE),
                datetime(2024, 1, 2, 9, 25, tzinfo=MARKET_TIMEZONE),
            ],
            "open": [100.0, 101.0, 102.0],
            "high": [102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
            "close": [101.0, 102.0, 103.0],
            "volume": [1_000.0, 1_100.0, 1_200.0],
            "symbol": ["TEST", "TEST", "TEST"],
        }
    )


class TestStrategy:
    strategy_id = "test-strategy"
    strategy_version = "1.0.0"
    parameters: Mapping[str, Any] = {"period": 2, "thresholds": [0.1, 0.2]}
    warmup_bars = 2

    def generate_signals(self, candles: pl.DataFrame) -> list[Signal]:
        validate_strategy_input(candles)
        source_row = candles.row(-1, named=True)
        return [
            Signal(
                strategy_id=self.strategy_id,
                strategy_version=self.strategy_version,
                symbol=source_row["symbol"],
                timestamp=bar_available_at(source_row["timestamp"]),
                side=Side.LONG,
                strategy_parameters=self.parameters,
                feature_snapshot={
                    "source_bar_start": source_row["timestamp"],
                    "close": source_row["close"],
                },
            )
        ]


def test_missing_required_column_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing required candle column.*volume"):
        validate_strategy_input(make_candles().drop("volume"))


def test_multiple_symbols_are_rejected() -> None:
    candles = make_candles().with_columns(
        pl.Series("symbol", ["TEST", "OTHER", "TEST"])
    )

    with pytest.raises(ValueError, match="exactly one symbol"):
        validate_strategy_input(candles)


def test_unsorted_timestamps_are_rejected() -> None:
    with pytest.raises(ValueError, match="ascending order"):
        validate_strategy_input(make_candles().reverse())


def test_duplicate_timestamps_are_rejected() -> None:
    candles = make_candles().with_columns(
        pl.Series(
            "timestamp",
            [
                datetime(2024, 1, 2, 9, 15, tzinfo=MARKET_TIMEZONE),
                datetime(2024, 1, 2, 9, 15, tzinfo=MARKET_TIMEZONE),
                datetime(2024, 1, 2, 9, 25, tzinfo=MARKET_TIMEZONE),
            ],
        )
    )

    with pytest.raises(ValueError, match="duplicates"):
        validate_strategy_input(candles)


def test_timezone_naive_timestamps_are_rejected() -> None:
    candles = make_candles().with_columns(pl.col("timestamp").dt.replace_time_zone(None))

    with pytest.raises(ValueError, match="timezone-aware"):
        validate_strategy_input(candles)


def test_valid_single_symbol_frame_is_accepted_without_mutation() -> None:
    candles = make_candles()
    original = candles.clone()

    assert validate_strategy_input(candles) is None
    assert candles.equals(original)


def test_bar_availability_adds_timeframe_and_preserves_timezone() -> None:
    bar_start = datetime(2024, 1, 2, 9, 15, tzinfo=MARKET_TIMEZONE)

    available_at = bar_available_at(bar_start)

    assert available_at == datetime(2024, 1, 2, 9, 20, tzinfo=MARKET_TIMEZONE)
    assert available_at.tzinfo is bar_start.tzinfo


def test_bar_availability_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        bar_available_at(datetime(2024, 1, 2, 9, 15))


@pytest.mark.parametrize("timeframe", [0, -5])
def test_bar_availability_rejects_non_positive_timeframe(timeframe: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        bar_available_at(
            datetime(2024, 1, 2, 9, 15, tzinfo=MARKET_TIMEZONE),
            timeframe,
        )


def test_strategy_protocol_and_generated_signal_contract() -> None:
    strategy = TestStrategy()
    candles = make_candles()

    signals = strategy.generate_signals(candles)

    assert isinstance(strategy, Strategy)
    assert strategy.strategy_id == "test-strategy"
    assert strategy.strategy_version == "1.0.0"
    assert strategy.parameters == {"period": 2, "thresholds": [0.1, 0.2]}
    assert strategy.warmup_bars == 2
    assert len(signals) == 1
    signal = signals[0]
    assert signal.status is SignalStatus.GENERATED
    assert signal.timestamp == bar_available_at(candles["timestamp"].item(-1))
    assert signal.timestamp != candles["timestamp"].item(-1)
    assert signal.strategy_parameters["period"] == 2
    assert signal.strategy_parameters["thresholds"] == (0.1, 0.2)
    assert signal.feature_snapshot == {
        "source_bar_start": candles["timestamp"].item(-1),
        "close": 103.0,
    }

