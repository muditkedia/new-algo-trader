"""Historical entry-order execution against start-stamped OHLCV candles."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import polars as pl

from algo_trader.data import CANONICAL_CANDLE_COLUMNS, bar_available_at
from algo_trader.domain import Fill, OrderIntent, OrderType, Side


class HistoricalExecutionSimulator:
    """Attempt broker-neutral entry fills without using pre-order candles."""

    def __init__(self, timeframe_minutes: int = 5) -> None:
        if isinstance(timeframe_minutes, bool) or not isinstance(timeframe_minutes, int):
            raise TypeError("timeframe_minutes must be an integer")
        if timeframe_minutes <= 0:
            raise ValueError("timeframe_minutes must be positive")
        self.timeframe_minutes = timeframe_minutes

    def fill_entry_order(self, order: OrderIntent, candles: pl.DataFrame) -> Fill | None:
        """Fill an entry intent from the first eligible historical opportunity.

        Candles are start-stamped. Only bars starting at or after the order
        timestamp are eligible, so the candle that generated a signal cannot
        fill that signal.
        """
        _validate_execution_candles(candles, order.signal.symbol)
        if candles.is_empty():
            return None

        eligible = candles.filter(
            pl.col("timestamp").dt.epoch("us") >= _epoch_microseconds(order.timestamp)
        )
        if eligible.is_empty():
            return None

        if order.order_type is OrderType.MARKET:
            row = eligible.select("timestamp", "open").row(0, named=True)
            return _simulated_fill(order, row["timestamp"], row["open"])

        return self._fill_limit_order(order, eligible)

    def _fill_limit_order(self, order: OrderIntent, eligible: pl.DataFrame) -> Fill | None:
        limit_price = order.limit_price
        if limit_price is None:  # Protected by OrderIntent; keeps static narrowing explicit.
            raise ValueError("a limit order requires limit_price")
        limit_value = float(limit_price)

        if order.signal.side is Side.LONG:
            open_fill = pl.col("open") <= limit_value
            touch_fill = pl.col("low") <= limit_value
        else:
            open_fill = pl.col("open") >= limit_value
            touch_fill = pl.col("high") >= limit_value

        opportunities = eligible.filter(open_fill | touch_fill).select(
            "timestamp",
            "open",
            open_fill.alias("fills_at_open"),
        )
        if opportunities.is_empty():
            return None

        row = opportunities.row(0, named=True)
        fills_at_open = row["fills_at_open"]
        fill_timestamp = (
            row["timestamp"]
            if fills_at_open
            else bar_available_at(row["timestamp"], self.timeframe_minutes)
        )
        fill_price = row["open"] if fills_at_open else limit_price
        return _simulated_fill(order, fill_timestamp, fill_price)


def _validate_execution_candles(candles: pl.DataFrame, expected_symbol: str) -> None:
    if not isinstance(candles, pl.DataFrame):
        raise TypeError("candles must be a Polars DataFrame")

    missing = [column for column in CANONICAL_CANDLE_COLUMNS if column not in candles.columns]
    if missing:
        raise ValueError(f"missing required execution column(s): {', '.join(missing)}")

    timestamp_dtype = candles.schema["timestamp"]
    if not isinstance(timestamp_dtype, pl.Datetime) or timestamp_dtype.time_zone is None:
        raise ValueError("execution timestamps must be timezone-aware")

    if candles.is_empty():
        return

    symbols = candles["symbol"].unique().to_list()
    if len(symbols) != 1 or symbols[0] is None:
        raise ValueError("execution candles must contain exactly one symbol")
    if symbols[0] != expected_symbol:
        raise ValueError(
            f"candle symbol {symbols[0]!r} does not match order symbol {expected_symbol!r}"
        )

    timestamps = candles["timestamp"]
    if timestamps.null_count() > 0:
        raise ValueError("execution timestamps must not contain null values")
    if timestamps.n_unique() != timestamps.len():
        raise ValueError("execution timestamps must not contain duplicates")
    if not timestamps.is_sorted(descending=False):
        raise ValueError("execution timestamps must be in ascending order")


def _epoch_microseconds(value: datetime) -> int:
    utc_value = value.astimezone(UTC)
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    elapsed = utc_value - epoch
    return (
        (elapsed.days * 86_400 + elapsed.seconds) * 1_000_000
        + elapsed.microseconds
    )


def _simulated_fill(order: OrderIntent, timestamp: datetime, price: object) -> Fill:
    return Fill(
        timestamp=timestamp,
        price=price if isinstance(price, Decimal) else Decimal(str(price)),
        quantity=order.quantity,
        slippage_per_unit=Decimal("0"),
        is_simulated=True,
    )
