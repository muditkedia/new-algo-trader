"""Validation of canonical candle frames at the strategy boundary."""

from __future__ import annotations

import polars as pl

REQUIRED_CANDLE_COLUMNS = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "symbol",
)


def validate_strategy_input(candles: pl.DataFrame) -> None:
    """Validate the small structural contract strategies require.

    This function only validates and never sorts, deduplicates, fills, or
    otherwise mutates the supplied raw OHLCV values.
    """
    if not isinstance(candles, pl.DataFrame):
        raise TypeError("candles must be a Polars DataFrame")

    missing = [column for column in REQUIRED_CANDLE_COLUMNS if column not in candles.columns]
    if missing:
        raise ValueError(f"missing required candle column(s): {', '.join(missing)}")

    symbols = candles["symbol"].unique().to_list()
    if len(symbols) != 1 or symbols[0] is None:
        raise ValueError("strategy input must contain exactly one symbol")

    timestamp_dtype = candles.schema["timestamp"]
    if not isinstance(timestamp_dtype, pl.Datetime) or timestamp_dtype.time_zone is None:
        raise ValueError("strategy timestamps must be timezone-aware")

    timestamps = candles["timestamp"]
    if timestamps.null_count() > 0:
        raise ValueError("strategy timestamps must not contain null values")
    if timestamps.n_unique() != timestamps.len():
        raise ValueError("strategy timestamps must not contain duplicates")
    if not timestamps.is_sorted(descending=False):
        raise ValueError("strategy timestamps must be in ascending order")

