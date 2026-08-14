"""Runtime information-availability filtering over Broker candle retrieval."""

from datetime import datetime

import polars as pl

from algo_trader.broker import AngelOneCandleClient, BrokerInstrument
from algo_trader.data import bar_available_at


def get_completed_five_minute_candles(
    candle_client: AngelOneCandleClient,
    instrument: BrokerInstrument,
    start: datetime,
    end: datetime,
    now: datetime,
) -> pl.DataFrame:
    """Return only broker candles whose five-minute availability time is known."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    candles = candle_client.get_five_minute_candles(instrument, start, end)
    if candles.is_empty():
        return candles.clone()
    completed = candles.filter(
        pl.col("timestamp").map_elements(
            lambda value: bar_available_at(value, 5) <= now,
            return_dtype=pl.Boolean,
        )
    )
    return completed.clone()
