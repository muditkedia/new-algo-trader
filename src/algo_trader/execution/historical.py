"""Historical entry and exit execution against start-stamped OHLCV candles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

import polars as pl

from algo_trader.data import CANONICAL_CANDLE_COLUMNS, bar_available_at
from algo_trader.domain import (
    ExitReason,
    Fill,
    OrderIntent,
    OrderType,
    ProtectiveExitSpec,
    Side,
)
from algo_trader.execution.slippage import ExecutionAction, NoSlippage, SlippageModel

_GENERIC_MARKET_EXIT_REASONS = frozenset(
    {ExitReason.TIME_EXIT, ExitReason.STRATEGY_EXIT, ExitReason.MANUAL}
)


@dataclass(frozen=True, slots=True)
class ExitResult:
    """Outcome of a simulated exit, including why the position was closed."""

    fill: Fill
    exit_reason: ExitReason


class HistoricalExecutionSimulator:
    """Simulate broker-neutral entries and exits from historical OHLCV candles.

    When both a protective stop and target are touched intrabar, with neither
    triggered at the open, STOP_LOSS wins because OHLC cannot reveal their order.
    """

    def __init__(
        self,
        timeframe_minutes: int = 5,
        slippage_model: SlippageModel | None = None,
    ) -> None:
        if isinstance(timeframe_minutes, bool) or not isinstance(timeframe_minutes, int):
            raise TypeError("timeframe_minutes must be an integer")
        if timeframe_minutes <= 0:
            raise ValueError("timeframe_minutes must be positive")
        self.timeframe_minutes = timeframe_minutes
        self.slippage_model = slippage_model if slippage_model is not None else NoSlippage()

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
            return self._create_fill(
                timestamp=row["timestamp"],
                raw_price=row["open"],
                quantity=order.quantity,
                action=_entry_action(order.signal.side),
            )

        return self._fill_limit_order(order, eligible)

    def fill_protective_exit(
        self,
        *,
        side: Side,
        symbol: str,
        quantity: int,
        entry_fill: Fill,
        protective_exit: ProtectiveExitSpec,
        candles: pl.DataFrame,
    ) -> ExitResult | None:
        """Fill the earliest eligible protective stop or target.

        Bars starting at ``entry_fill.timestamp`` are eligible. This includes
        the entry bar for open fills, while naturally excluding the source bar
        of an intrabar entry whose timestamp is that bar's availability time.
        """
        _validate_side(side)
        _validate_symbol(symbol)
        _validate_quantity(quantity)
        if not isinstance(entry_fill, Fill):
            raise TypeError("entry_fill must be a Fill")
        if quantity != entry_fill.quantity:
            raise ValueError("protective exit quantity must equal entry_fill.quantity")
        if not isinstance(protective_exit, ProtectiveExitSpec):
            raise TypeError("protective_exit must be a ProtectiveExitSpec")
        _validate_protective_geometry(side, entry_fill.price, protective_exit)
        _validate_execution_candles(candles, symbol)
        if candles.is_empty():
            return None

        eligible = candles.filter(
            pl.col("timestamp").dt.epoch("us") >= _epoch_microseconds(entry_fill.timestamp)
        )
        if eligible.is_empty():
            return None

        trigger_expressions = _protective_trigger_expressions(side, protective_exit)
        opportunities = eligible.filter(
            trigger_expressions["stop_at_open"]
            | trigger_expressions["target_at_open"]
            | trigger_expressions["stop_touched"]
            | trigger_expressions["target_touched"]
        ).select(
            "timestamp",
            "open",
            *(
                expression.alias(name)
                for name, expression in trigger_expressions.items()
            ),
        )
        if opportunities.is_empty():
            return None

        row = opportunities.row(0, named=True)
        raw_price, fill_timestamp, exit_reason = self._resolve_protective_trigger(
            row,
            protective_exit,
        )
        fill = self._create_fill(
            timestamp=fill_timestamp,
            raw_price=raw_price,
            quantity=quantity,
            action=_exit_action(side),
        )
        return ExitResult(fill=fill, exit_reason=exit_reason)

    def fill_market_exit(
        self,
        *,
        side: Side,
        symbol: str,
        quantity: int,
        requested_at: datetime,
        exit_reason: ExitReason,
        candles: pl.DataFrame,
    ) -> ExitResult | None:
        """Exit at the first candle open at or after the requested time."""
        _validate_side(side)
        _validate_symbol(symbol)
        _validate_quantity(quantity)
        _validate_aware_timestamp(requested_at, "requested_at")
        if not isinstance(exit_reason, ExitReason):
            raise TypeError("exit_reason must be an ExitReason")
        if exit_reason not in _GENERIC_MARKET_EXIT_REASONS:
            raise ValueError(
                "market exit reason must be TIME_EXIT, STRATEGY_EXIT, or MANUAL"
            )
        _validate_execution_candles(candles, symbol)
        if candles.is_empty():
            return None

        eligible = candles.filter(
            pl.col("timestamp").dt.epoch("us") >= _epoch_microseconds(requested_at)
        )
        if eligible.is_empty():
            return None

        row = eligible.select("timestamp", "open").row(0, named=True)
        fill = self._create_fill(
            timestamp=row["timestamp"],
            raw_price=row["open"],
            quantity=quantity,
            action=_exit_action(side),
        )
        return ExitResult(fill=fill, exit_reason=exit_reason)

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
        return self._create_fill(
            timestamp=fill_timestamp,
            raw_price=fill_price,
            quantity=order.quantity,
            action=_entry_action(order.signal.side),
        )

    def _resolve_protective_trigger(
        self,
        row: dict[str, object],
        protective_exit: ProtectiveExitSpec,
    ) -> tuple[object, datetime, ExitReason]:
        bar_start = row["timestamp"]
        if not isinstance(bar_start, datetime):
            raise TypeError("candle timestamp must materialize as a datetime")

        if row["stop_at_open"]:
            return row["open"], bar_start, ExitReason.STOP_LOSS
        if row["target_at_open"]:
            return row["open"], bar_start, ExitReason.TARGET_REACHED

        bar_end = bar_available_at(bar_start, self.timeframe_minutes)
        # If both are true, stop is deliberately checked first: STOP LOSS WINS.
        if row["stop_touched"]:
            return protective_exit.stop_price, bar_end, ExitReason.STOP_LOSS
        return protective_exit.target_price, bar_end, ExitReason.TARGET_REACHED

    def _create_fill(
        self,
        *,
        timestamp: datetime,
        raw_price: object,
        quantity: int,
        action: ExecutionAction,
    ) -> Fill:
        raw_decimal = _as_price_decimal(raw_price)
        adjusted_price = self.slippage_model.apply(raw_decimal, action)
        if not isinstance(adjusted_price, Decimal):
            raise TypeError("slippage models must return a Decimal price")
        if not adjusted_price.is_finite() or adjusted_price <= 0:
            raise ValueError("slippage-adjusted execution price must be finite and positive")
        if action is ExecutionAction.BUY and adjusted_price < raw_decimal:
            raise ValueError("BUY slippage must not improve the raw execution price")
        if action is ExecutionAction.SELL and adjusted_price > raw_decimal:
            raise ValueError("SELL slippage must not improve the raw execution price")

        return Fill(
            timestamp=timestamp,
            price=adjusted_price,
            quantity=quantity,
            slippage_per_unit=abs(adjusted_price - raw_decimal),
            is_simulated=True,
        )


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


def _validate_protective_geometry(
    side: Side,
    entry_price: Decimal,
    protective_exit: ProtectiveExitSpec,
) -> None:
    stop_price = protective_exit.stop_price
    target_price = protective_exit.target_price
    if side is Side.LONG:
        if stop_price is not None and stop_price >= entry_price:
            raise ValueError("LONG stop_price must be below entry price")
        if target_price is not None and target_price <= entry_price:
            raise ValueError("LONG target_price must be above entry price")
    else:
        if stop_price is not None and stop_price <= entry_price:
            raise ValueError("SHORT stop_price must be above entry price")
        if target_price is not None and target_price >= entry_price:
            raise ValueError("SHORT target_price must be below entry price")


def _protective_trigger_expressions(
    side: Side,
    protective_exit: ProtectiveExitSpec,
) -> dict[str, pl.Expr]:
    stop_price = (
        float(protective_exit.stop_price)
        if protective_exit.stop_price is not None
        else None
    )
    target_price = (
        float(protective_exit.target_price)
        if protective_exit.target_price is not None
        else None
    )
    never = pl.lit(False)

    if side is Side.LONG:
        return {
            "stop_at_open": pl.col("open") <= stop_price if stop_price is not None else never,
            "target_at_open": (
                pl.col("open") >= target_price if target_price is not None else never
            ),
            "stop_touched": pl.col("low") <= stop_price if stop_price is not None else never,
            "target_touched": (
                pl.col("high") >= target_price if target_price is not None else never
            ),
        }
    return {
        "stop_at_open": pl.col("open") >= stop_price if stop_price is not None else never,
        "target_at_open": (
            pl.col("open") <= target_price if target_price is not None else never
        ),
        "stop_touched": pl.col("high") >= stop_price if stop_price is not None else never,
        "target_touched": (
            pl.col("low") <= target_price if target_price is not None else never
        ),
    }


def _entry_action(side: Side) -> ExecutionAction:
    return ExecutionAction.BUY if side is Side.LONG else ExecutionAction.SELL


def _exit_action(side: Side) -> ExecutionAction:
    return ExecutionAction.SELL if side is Side.LONG else ExecutionAction.BUY


def _validate_side(side: Side) -> None:
    if not isinstance(side, Side):
        raise TypeError("side must be a Side")


def _validate_symbol(symbol: str) -> None:
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")


def _validate_quantity(quantity: int) -> None:
    if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
        raise ValueError("quantity must be a positive integer")


def _validate_aware_timestamp(value: datetime, name: str) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _epoch_microseconds(value: datetime) -> int:
    utc_value = value.astimezone(UTC)
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    elapsed = utc_value - epoch
    return (elapsed.days * 86_400 + elapsed.seconds) * 1_000_000 + elapsed.microseconds


def _as_price_decimal(value: object) -> Decimal:
    try:
        price = value if isinstance(value, Decimal) else Decimal(str(value))
    except Exception as error:
        raise ValueError("raw execution price must be numeric") from error
    if not price.is_finite() or price <= 0:
        raise ValueError("raw execution price must be finite and positive")
    return price
