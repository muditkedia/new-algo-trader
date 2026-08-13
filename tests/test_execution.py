from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from algo_trader import OrderIntent, OrderType, Side, Signal
from algo_trader.execution import FixedBasisPointsSlippage, HistoricalExecutionSimulator

MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")


def at(hour: int, minute: int) -> datetime:
    return datetime(2024, 1, 2, hour, minute, tzinfo=MARKET_TIMEZONE)


def make_candles(
    *,
    timestamps: list[datetime] | None = None,
    opens: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    symbols: list[str] | None = None,
) -> pl.DataFrame:
    timestamps = timestamps or [at(9, 15), at(9, 20), at(9, 25)]
    opens = opens or [100.0, 101.0, 102.0]
    highs = highs or [102.0, 103.0, 104.0]
    lows = lows or [99.0, 100.0, 101.0]
    symbols = symbols or ["TEST"] * len(timestamps)
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": [101.0] * len(timestamps),
            "volume": [1_000.0] * len(timestamps),
            "symbol": symbols,
        }
    )


def make_order(
    *,
    side: Side = Side.LONG,
    order_type: OrderType = OrderType.MARKET,
    timestamp: datetime | None = None,
    limit_price: Decimal | None = None,
) -> OrderIntent:
    timestamp = timestamp or at(9, 20)
    signal = Signal(
        strategy_id="test-strategy",
        strategy_version="1.0.0",
        symbol="TEST",
        timestamp=timestamp,
        side=side,
        strategy_parameters={},
        feature_snapshot={},
    )
    return OrderIntent(
        signal=signal,
        timestamp=timestamp,
        quantity=10,
        requested_notional=50_000,
        order_type=order_type,
        limit_price=limit_price,
    )


def assert_simulated_fill(fill, *, timestamp: datetime, price: str) -> None:
    assert fill is not None
    assert fill.timestamp == timestamp
    assert fill.price == Decimal(price)
    assert fill.quantity == 10
    assert fill.is_simulated is True
    assert fill.slippage_per_unit == Decimal("0")


def test_market_skips_pre_signal_bar_and_fills_eligible_open_without_mutation() -> None:
    candles = make_candles(opens=[90.0, 101.0, 102.0])
    original = candles.clone()

    fill = HistoricalExecutionSimulator().fill_entry_order(make_order(), candles)

    assert_simulated_fill(fill, timestamp=at(9, 20), price="101.0")
    assert candles.equals(original)


def test_market_fills_first_later_available_candle() -> None:
    candles = make_candles(
        timestamps=[at(9, 15), at(9, 25)],
        opens=[100.0, 104.0],
        highs=[101.0, 105.0],
        lows=[99.0, 103.0],
    )

    fill = HistoricalExecutionSimulator().fill_entry_order(make_order(), candles)

    assert_simulated_fill(fill, timestamp=at(9, 25), price="104.0")


def test_market_with_no_eligible_future_candle_returns_none() -> None:
    candles = make_candles(
        timestamps=[at(9, 15)],
        opens=[100.0],
        highs=[101.0],
        lows=[99.0],
    )

    assert HistoricalExecutionSimulator().fill_entry_order(make_order(), candles) is None


def test_long_limit_open_below_limit_fills_at_improved_open() -> None:
    candles = make_candles(opens=[100.0, 98.0, 102.0])
    order = make_order(order_type=OrderType.LIMIT, limit_price=Decimal("100"))

    fill = HistoricalExecutionSimulator().fill_entry_order(order, candles)

    assert_simulated_fill(fill, timestamp=at(9, 20), price="98.0")


def test_long_limit_touch_fills_at_limit_at_bar_availability() -> None:
    candles = make_candles(opens=[100.0, 102.0, 103.0], lows=[99.0, 100.0, 101.0])
    order = make_order(order_type=OrderType.LIMIT, limit_price=Decimal("100"))

    fill = HistoricalExecutionSimulator().fill_entry_order(order, candles)

    assert_simulated_fill(fill, timestamp=at(9, 25), price="100")


def test_long_limit_never_touched_returns_none() -> None:
    candles = make_candles(opens=[100.0, 102.0, 103.0], lows=[99.0, 101.0, 102.0])
    order = make_order(order_type=OrderType.LIMIT, limit_price=Decimal("100"))

    assert HistoricalExecutionSimulator().fill_entry_order(order, candles) is None


def test_short_limit_open_above_limit_fills_at_improved_open() -> None:
    candles = make_candles(opens=[100.0, 102.0, 98.0])
    order = make_order(
        side=Side.SHORT,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("100"),
    )

    fill = HistoricalExecutionSimulator().fill_entry_order(order, candles)

    assert_simulated_fill(fill, timestamp=at(9, 20), price="102.0")


def test_short_limit_touch_fills_at_limit_at_bar_availability() -> None:
    candles = make_candles(opens=[100.0, 98.0, 97.0], highs=[101.0, 100.0, 99.0])
    order = make_order(
        side=Side.SHORT,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("100"),
    )

    fill = HistoricalExecutionSimulator().fill_entry_order(order, candles)

    assert_simulated_fill(fill, timestamp=at(9, 25), price="100")


def test_short_limit_never_touched_returns_none() -> None:
    candles = make_candles(opens=[100.0, 98.0, 97.0], highs=[101.0, 99.0, 98.0])
    order = make_order(
        side=Side.SHORT,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("100"),
    )

    assert HistoricalExecutionSimulator().fill_entry_order(order, candles) is None


def test_wrong_symbol_is_rejected() -> None:
    candles = make_candles(symbols=["OTHER"] * 3)

    with pytest.raises(ValueError, match="does not match order symbol"):
        HistoricalExecutionSimulator().fill_entry_order(make_order(), candles)


def test_multiple_symbols_are_rejected() -> None:
    candles = make_candles(symbols=["TEST", "OTHER", "TEST"])

    with pytest.raises(ValueError, match="exactly one symbol"):
        HistoricalExecutionSimulator().fill_entry_order(make_order(), candles)


def test_missing_execution_column_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing required execution column.*open"):
        HistoricalExecutionSimulator().fill_entry_order(
            make_order(),
            make_candles().drop("open"),
        )


def test_timezone_naive_timestamps_are_rejected() -> None:
    candles = make_candles().with_columns(pl.col("timestamp").dt.replace_time_zone(None))

    with pytest.raises(ValueError, match="timezone-aware"):
        HistoricalExecutionSimulator().fill_entry_order(make_order(), candles)


def test_unsorted_timestamps_are_rejected() -> None:
    with pytest.raises(ValueError, match="ascending order"):
        HistoricalExecutionSimulator().fill_entry_order(make_order(), make_candles().reverse())


def test_duplicate_timestamps_are_rejected() -> None:
    candles = make_candles(timestamps=[at(9, 15), at(9, 20), at(9, 20)])

    with pytest.raises(ValueError, match="duplicates"):
        HistoricalExecutionSimulator().fill_entry_order(make_order(), candles)


def test_empty_valid_candle_frame_returns_none() -> None:
    candles = make_candles().clear()

    assert HistoricalExecutionSimulator().fill_entry_order(make_order(), candles) is None


@pytest.mark.parametrize(
    ("side", "expected_price"),
    [(Side.LONG, Decimal("100.10")), (Side.SHORT, Decimal("99.90"))],
)
def test_market_entry_applies_exact_adverse_slippage(
    side: Side,
    expected_price: Decimal,
) -> None:
    candles = make_candles(opens=[90.0, 100.0, 102.0])
    simulator = HistoricalExecutionSimulator(
        slippage_model=FixedBasisPointsSlippage(Decimal("10"))
    )

    fill = simulator.fill_entry_order(make_order(side=side), candles)

    assert fill is not None
    assert fill.price == expected_price
    assert fill.slippage_per_unit == Decimal("0.10")


@pytest.mark.parametrize(
    ("side", "open_price", "expected_price"),
    [
        (Side.LONG, 98.0, Decimal("98.098")),
        (Side.SHORT, 102.0, Decimal("101.898")),
    ],
)
def test_limit_entry_selects_improved_raw_price_before_slippage(
    side: Side,
    open_price: float,
    expected_price: Decimal,
) -> None:
    candles = make_candles(opens=[100.0, open_price, 103.0])
    simulator = HistoricalExecutionSimulator(
        slippage_model=FixedBasisPointsSlippage(Decimal("10"))
    )
    order = make_order(
        side=side,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("100"),
    )

    fill = simulator.fill_entry_order(order, candles)

    assert fill is not None
    assert fill.price == expected_price
    assert fill.slippage_per_unit == abs(expected_price - Decimal(str(open_price)))
