"""Aligned Polars Series wrappers around TA-Lib indicators."""

from __future__ import annotations

from collections.abc import Callable

import polars as pl
import talib


def sma(values: pl.Series, period: int) -> pl.Series:
    """Return TA-Lib SMA values aligned to the input Series."""
    return _single_input_indicator(values, period, "sma", talib.SMA)


def ema(values: pl.Series, period: int) -> pl.Series:
    """Return TA-Lib EMA values aligned to the input Series."""
    return _single_input_indicator(values, period, "ema", talib.EMA)


def rsi(values: pl.Series, period: int) -> pl.Series:
    """Return TA-Lib RSI values aligned to the input Series."""
    return _single_input_indicator(values, period, "rsi", talib.RSI)


def atr(candles: pl.DataFrame, period: int) -> pl.Series:
    """Return TA-Lib ATR values aligned to a candle DataFrame."""
    _validate_period(period)
    _require_columns(candles, ("high", "low", "close"))
    high = _numeric_values(candles["high"], "high")
    low = _numeric_values(candles["low"], "low")
    close = _numeric_values(candles["close"], "close")
    return pl.Series(f"atr_{period}", talib.ATR(high, low, close, timeperiod=period), pl.Float64)


def _single_input_indicator(
    values: pl.Series,
    period: int,
    name: str,
    calculation: Callable,
) -> pl.Series:
    _validate_period(period)
    numeric_values = _numeric_values(values, "values")
    result = calculation(numeric_values, timeperiod=period)
    return pl.Series(f"{name}_{period}", result, pl.Float64)


def _numeric_values(values: pl.Series, name: str):
    if not isinstance(values, pl.Series):
        raise TypeError(f"{name} must be a Polars Series")
    if not values.dtype.is_numeric():
        raise TypeError(f"{name} must have a numeric dtype")
    return values.cast(pl.Float64).to_numpy()


def _validate_period(period: int) -> None:
    if isinstance(period, bool) or not isinstance(period, int):
        raise TypeError("period must be an integer")
    if period <= 0:
        raise ValueError("period must be positive")


def _require_columns(candles: pl.DataFrame, required: tuple[str, ...]) -> None:
    if not isinstance(candles, pl.DataFrame):
        raise TypeError("candles must be a Polars DataFrame")
    missing = [column for column in required if column not in candles.columns]
    if missing:
        raise ValueError(f"missing required indicator column(s): {', '.join(missing)}")

