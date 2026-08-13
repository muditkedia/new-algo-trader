import numpy as np
import polars as pl
import pytest
import talib

from algo_trader.indicators import atr, ema, rsi, sma


def make_ohlc() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "high": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            "low": [9.0, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
            "close": [9.5, 10.5, 11.0, 12.5, 13.0, 14.5, 15.0, 16.5],
        }
    )


@pytest.mark.parametrize(
    ("calculation", "truth"),
    [(sma, talib.SMA), (ema, talib.EMA), (rsi, talib.RSI)],
)
def test_single_series_indicators_match_talib_and_preserve_alignment(
    calculation,
    truth,
) -> None:
    values = pl.Series("close", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    original = values.clone()

    result = calculation(values, period=3)
    expected = truth(values.to_numpy(), timeperiod=3)

    assert len(result) == len(values)
    assert result.dtype == pl.Float64
    np.testing.assert_allclose(result.to_numpy(), expected, equal_nan=True)
    assert result[:2].is_nan().all()
    assert values.equals(original)


def test_sma_known_numeric_case() -> None:
    result = sma(pl.Series([1.0, 2.0, 3.0, 4.0]), period=3)

    np.testing.assert_allclose(result.to_numpy(), [np.nan, np.nan, 2.0, 3.0], equal_nan=True)


def test_atr_matches_talib_and_preserves_frame() -> None:
    candles = make_ohlc()
    original = candles.clone()

    result = atr(candles, period=3)
    expected = talib.ATR(
        candles["high"].to_numpy(),
        candles["low"].to_numpy(),
        candles["close"].to_numpy(),
        timeperiod=3,
    )

    assert len(result) == candles.height
    assert result.dtype == pl.Float64
    np.testing.assert_allclose(result.to_numpy(), expected, equal_nan=True)
    assert result[:3].is_nan().all()
    assert candles.equals(original)


def test_atr_rejects_missing_required_column() -> None:
    with pytest.raises(ValueError, match="missing required indicator column.*high"):
        atr(make_ohlc().drop("high"), period=3)
