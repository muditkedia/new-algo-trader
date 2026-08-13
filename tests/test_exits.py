from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from algo_trader import ExitReason, Fill, ProtectiveExitSpec, Side
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
    opens = opens or [50.0, 100.0, 100.0]
    highs = highs or [200.0, 104.0, 104.0]
    lows = lows or [1.0, 96.0, 96.0]
    symbols = symbols or ["TEST"] * len(timestamps)
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": [100.0] * len(timestamps),
            "volume": [1_000.0] * len(timestamps),
            "symbol": symbols,
        }
    )


def make_entry_fill(timestamp: datetime | None = None) -> Fill:
    return Fill(
        timestamp=timestamp or at(9, 20),
        price=Decimal("100"),
        quantity=10,
        is_simulated=True,
    )


def protective_exit(
    side: Side,
    candles: pl.DataFrame,
    protective_exit: ProtectiveExitSpec,
    *,
    entry_fill: Fill | None = None,
    simulator: HistoricalExecutionSimulator | None = None,
):
    return (simulator or HistoricalExecutionSimulator()).fill_protective_exit(
        side=side,
        symbol="TEST",
        quantity=10,
        entry_fill=entry_fill or make_entry_fill(),
        protective_exit=protective_exit,
        candles=candles,
    )


@pytest.mark.parametrize(
    ("opens", "highs", "lows", "specification", "price", "timestamp", "reason"),
    [
        (
            [50.0, 94.0, 100.0],
            [200.0, 96.0, 104.0],
            [1.0, 93.0, 96.0],
            ProtectiveExitSpec(stop_price=Decimal("95")),
            Decimal("94.0"),
            at(9, 20),
            ExitReason.STOP_LOSS,
        ),
        (
            [50.0, 100.0, 100.0],
            [200.0, 102.0, 104.0],
            [1.0, 94.0, 96.0],
            ProtectiveExitSpec(stop_price=Decimal("95")),
            Decimal("95"),
            at(9, 25),
            ExitReason.STOP_LOSS,
        ),
        (
            [50.0, 106.0, 100.0],
            [200.0, 107.0, 104.0],
            [1.0, 104.0, 96.0],
            ProtectiveExitSpec(target_price=Decimal("105")),
            Decimal("106.0"),
            at(9, 20),
            ExitReason.TARGET_REACHED,
        ),
        (
            [50.0, 100.0, 100.0],
            [200.0, 106.0, 104.0],
            [1.0, 98.0, 96.0],
            ProtectiveExitSpec(target_price=Decimal("105")),
            Decimal("105"),
            at(9, 25),
            ExitReason.TARGET_REACHED,
        ),
    ],
    ids=["stop-gap", "stop-touch", "target-gap", "target-touch"],
)
def test_long_protective_exit_rules(
    opens: list[float],
    highs: list[float],
    lows: list[float],
    specification: ProtectiveExitSpec,
    price: Decimal,
    timestamp: datetime,
    reason: ExitReason,
) -> None:
    result = protective_exit(
        Side.LONG,
        make_candles(opens=opens, highs=highs, lows=lows),
        specification,
    )

    assert result is not None
    assert result.fill.price == price
    assert result.fill.timestamp == timestamp
    assert result.exit_reason is reason
    assert result.fill.is_simulated is True


def test_long_protective_exit_returns_none_when_neither_level_is_touched() -> None:
    specification = ProtectiveExitSpec(
        stop_price=Decimal("95"),
        target_price=Decimal("105"),
    )

    assert protective_exit(Side.LONG, make_candles(), specification) is None


def test_long_ambiguous_intrabar_stop_and_target_uses_stop_loss() -> None:
    candles = make_candles(highs=[200.0, 106.0, 104.0], lows=[1.0, 94.0, 96.0])
    original = candles.clone()
    specification = ProtectiveExitSpec(
        stop_price=Decimal("95"),
        target_price=Decimal("105"),
    )

    result = protective_exit(Side.LONG, candles, specification)

    assert result is not None
    assert result.exit_reason is ExitReason.STOP_LOSS
    assert result.fill.price == Decimal("95")
    assert result.fill.timestamp == at(9, 25)
    assert candles.equals(original)


@pytest.mark.parametrize(
    ("opens", "highs", "lows", "specification", "price", "timestamp", "reason"),
    [
        (
            [50.0, 106.0, 100.0],
            [200.0, 107.0, 104.0],
            [1.0, 104.0, 96.0],
            ProtectiveExitSpec(stop_price=Decimal("105")),
            Decimal("106.0"),
            at(9, 20),
            ExitReason.STOP_LOSS,
        ),
        (
            [50.0, 100.0, 100.0],
            [200.0, 106.0, 104.0],
            [1.0, 98.0, 96.0],
            ProtectiveExitSpec(stop_price=Decimal("105")),
            Decimal("105"),
            at(9, 25),
            ExitReason.STOP_LOSS,
        ),
        (
            [50.0, 94.0, 100.0],
            [200.0, 96.0, 104.0],
            [1.0, 93.0, 96.0],
            ProtectiveExitSpec(target_price=Decimal("95")),
            Decimal("94.0"),
            at(9, 20),
            ExitReason.TARGET_REACHED,
        ),
        (
            [50.0, 100.0, 100.0],
            [200.0, 102.0, 104.0],
            [1.0, 94.0, 96.0],
            ProtectiveExitSpec(target_price=Decimal("95")),
            Decimal("95"),
            at(9, 25),
            ExitReason.TARGET_REACHED,
        ),
    ],
    ids=["stop-gap", "stop-touch", "target-gap", "target-touch"],
)
def test_short_protective_exit_rules(
    opens: list[float],
    highs: list[float],
    lows: list[float],
    specification: ProtectiveExitSpec,
    price: Decimal,
    timestamp: datetime,
    reason: ExitReason,
) -> None:
    result = protective_exit(
        Side.SHORT,
        make_candles(opens=opens, highs=highs, lows=lows),
        specification,
    )

    assert result is not None
    assert result.fill.price == price
    assert result.fill.timestamp == timestamp
    assert result.exit_reason is reason
    assert result.fill.is_simulated is True


def test_short_protective_exit_returns_none_when_neither_level_is_touched() -> None:
    specification = ProtectiveExitSpec(
        stop_price=Decimal("105"),
        target_price=Decimal("95"),
    )

    assert protective_exit(Side.SHORT, make_candles(), specification) is None


def test_short_ambiguous_intrabar_stop_and_target_uses_stop_loss() -> None:
    candles = make_candles(highs=[200.0, 106.0, 104.0], lows=[1.0, 94.0, 96.0])
    specification = ProtectiveExitSpec(
        stop_price=Decimal("105"),
        target_price=Decimal("95"),
    )

    result = protective_exit(Side.SHORT, candles, specification)

    assert result is not None
    assert result.exit_reason is ExitReason.STOP_LOSS
    assert result.fill.price == Decimal("105")
    assert result.fill.timestamp == at(9, 25)


@pytest.mark.parametrize(
    ("side", "opens", "highs", "lows", "expected_price"),
    [
        (
            Side.LONG,
            [50.0, 106.0, 100.0],
            [200.0, 107.0, 104.0],
            [1.0, 94.0, 96.0],
            Decimal("106.0"),
        ),
        (
            Side.SHORT,
            [50.0, 94.0, 100.0],
            [200.0, 106.0, 104.0],
            [1.0, 93.0, 96.0],
            Decimal("94.0"),
        ),
    ],
)
def test_target_triggered_at_open_precedes_opposing_intrabar_stop_touch(
    side: Side,
    opens: list[float],
    highs: list[float],
    lows: list[float],
    expected_price: Decimal,
) -> None:
    specification = (
        ProtectiveExitSpec(stop_price=Decimal("95"), target_price=Decimal("105"))
        if side is Side.LONG
        else ProtectiveExitSpec(stop_price=Decimal("105"), target_price=Decimal("95"))
    )

    result = protective_exit(
        side,
        make_candles(opens=opens, highs=highs, lows=lows),
        specification,
    )

    assert result is not None
    assert result.exit_reason is ExitReason.TARGET_REACHED
    assert result.fill.price == expected_price
    assert result.fill.timestamp == at(9, 20)


def test_open_entry_can_exit_during_same_candle() -> None:
    candles = make_candles(lows=[1.0, 94.0, 96.0])

    result = protective_exit(
        Side.LONG,
        candles,
        ProtectiveExitSpec(stop_price=Decimal("95")),
        entry_fill=make_entry_fill(at(9, 20)),
    )

    assert result is not None
    assert result.fill.timestamp == at(9, 25)


def test_intrabar_touch_entry_cannot_exit_from_its_source_candle() -> None:
    candles = make_candles(
        timestamps=[at(9, 20), at(9, 25)],
        opens=[100.0, 100.0],
        highs=[106.0, 104.0],
        lows=[94.0, 96.0],
    )

    result = protective_exit(
        Side.LONG,
        candles,
        ProtectiveExitSpec(
            stop_price=Decimal("95"),
            target_price=Decimal("105"),
        ),
        entry_fill=make_entry_fill(at(9, 25)),
    )

    assert result is None


@pytest.mark.parametrize(
    ("side", "specification", "expected_price", "expected_slippage"),
    [
        (
            Side.LONG,
            ProtectiveExitSpec(target_price=Decimal("105")),
            Decimal("104.895"),
            Decimal("0.105"),
        ),
        (
            Side.SHORT,
            ProtectiveExitSpec(target_price=Decimal("95")),
            Decimal("95.095"),
            Decimal("0.095"),
        ),
    ],
)
def test_protective_exit_applies_adverse_slippage_after_raw_trigger(
    side: Side,
    specification: ProtectiveExitSpec,
    expected_price: Decimal,
    expected_slippage: Decimal,
) -> None:
    candles = make_candles(highs=[200.0, 106.0, 104.0], lows=[1.0, 94.0, 96.0])
    simulator = HistoricalExecutionSimulator(
        slippage_model=FixedBasisPointsSlippage(Decimal("10"))
    )

    result = protective_exit(
        side,
        candles,
        specification,
        simulator=simulator,
    )

    assert result is not None
    assert result.fill.price == expected_price
    assert result.fill.slippage_per_unit == expected_slippage


@pytest.mark.parametrize(
    ("side", "reason", "expected_price"),
    [
        (Side.LONG, ExitReason.TIME_EXIT, Decimal("99.90")),
        (Side.SHORT, ExitReason.STRATEGY_EXIT, Decimal("100.10")),
        (Side.LONG, ExitReason.MANUAL, Decimal("99.90")),
    ],
)
def test_market_exit_uses_first_eligible_open_and_adverse_slippage(
    side: Side,
    reason: ExitReason,
    expected_price: Decimal,
) -> None:
    candles = make_candles(
        timestamps=[at(9, 15), at(9, 25)],
        opens=[50.0, 100.0],
        highs=[200.0, 101.0],
        lows=[1.0, 99.0],
    )
    simulator = HistoricalExecutionSimulator(
        slippage_model=FixedBasisPointsSlippage(Decimal("10"))
    )

    result = simulator.fill_market_exit(
        side=side,
        symbol="TEST",
        quantity=10,
        requested_at=at(9, 20),
        exit_reason=reason,
        candles=candles,
    )

    assert result is not None
    assert result.fill.timestamp == at(9, 25)
    assert result.fill.price == expected_price
    assert result.fill.slippage_per_unit == Decimal("0.10")
    assert result.exit_reason is reason


@pytest.mark.parametrize(
    ("side", "specification", "message"),
    [
        (Side.LONG, ProtectiveExitSpec(stop_price=Decimal("100")), "LONG stop_price"),
        (Side.LONG, ProtectiveExitSpec(target_price=Decimal("100")), "LONG target_price"),
        (Side.SHORT, ProtectiveExitSpec(stop_price=Decimal("100")), "SHORT stop_price"),
        (Side.SHORT, ProtectiveExitSpec(target_price=Decimal("100")), "SHORT target_price"),
    ],
)
def test_invalid_protective_geometry_is_rejected(
    side: Side,
    specification: ProtectiveExitSpec,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        protective_exit(side, make_candles(), specification)


def test_protective_exit_rejects_wrong_symbol() -> None:
    with pytest.raises(ValueError, match="does not match order symbol"):
        protective_exit(
            Side.LONG,
            make_candles(symbols=["OTHER"] * 3),
            ProtectiveExitSpec(stop_price=Decimal("95")),
        )


def test_protective_exit_rejects_multiple_symbols() -> None:
    with pytest.raises(ValueError, match="exactly one symbol"):
        protective_exit(
            Side.LONG,
            make_candles(symbols=["TEST", "OTHER", "TEST"]),
            ProtectiveExitSpec(stop_price=Decimal("95")),
        )


def test_protective_exit_rejects_incomplete_schema() -> None:
    with pytest.raises(ValueError, match="missing required execution column.*close"):
        protective_exit(
            Side.LONG,
            make_candles().drop("close"),
            ProtectiveExitSpec(stop_price=Decimal("95")),
        )


def test_protective_exit_rejects_naive_timestamps() -> None:
    candles = make_candles().with_columns(pl.col("timestamp").dt.replace_time_zone(None))

    with pytest.raises(ValueError, match="timezone-aware"):
        protective_exit(
            Side.LONG,
            candles,
            ProtectiveExitSpec(stop_price=Decimal("95")),
        )


def test_protective_exit_rejects_unsorted_timestamps() -> None:
    with pytest.raises(ValueError, match="ascending order"):
        protective_exit(
            Side.LONG,
            make_candles().reverse(),
            ProtectiveExitSpec(stop_price=Decimal("95")),
        )


def test_protective_exit_rejects_duplicate_timestamps() -> None:
    candles = make_candles(timestamps=[at(9, 15), at(9, 20), at(9, 20)])

    with pytest.raises(ValueError, match="duplicates"):
        protective_exit(
            Side.LONG,
            candles,
            ProtectiveExitSpec(stop_price=Decimal("95")),
        )


def test_protective_exit_empty_valid_frame_returns_none() -> None:
    assert (
        protective_exit(
            Side.LONG,
            make_candles().clear(),
            ProtectiveExitSpec(stop_price=Decimal("95")),
        )
        is None
    )


def test_protective_exit_rejects_non_positive_quantity() -> None:
    with pytest.raises(ValueError, match="quantity must be a positive integer"):
        HistoricalExecutionSimulator().fill_protective_exit(
            side=Side.LONG,
            symbol="TEST",
            quantity=0,
            entry_fill=make_entry_fill(),
            protective_exit=ProtectiveExitSpec(stop_price=Decimal("95")),
            candles=make_candles().clear(),
        )


def test_protective_exit_accepts_quantity_matching_entry_fill() -> None:
    entry_fill = make_entry_fill()
    result = HistoricalExecutionSimulator().fill_protective_exit(
        side=Side.LONG,
        symbol="TEST",
        quantity=10,
        entry_fill=entry_fill,
        protective_exit=ProtectiveExitSpec(target_price=Decimal("105")),
        candles=make_candles(highs=[200.0, 106.0, 104.0]),
    )

    assert result is not None
    assert result.fill.quantity == entry_fill.quantity == 10


def test_protective_exit_rejects_quantity_different_from_entry_fill() -> None:
    with pytest.raises(ValueError, match="must equal entry_fill.quantity"):
        HistoricalExecutionSimulator().fill_protective_exit(
            side=Side.LONG,
            symbol="TEST",
            quantity=5,
            entry_fill=make_entry_fill(),
            protective_exit=ProtectiveExitSpec(stop_price=Decimal("95")),
            candles=make_candles(),
        )


def test_protective_exit_rejects_non_fill_entry() -> None:
    with pytest.raises(TypeError, match="entry_fill must be a Fill"):
        HistoricalExecutionSimulator().fill_protective_exit(
            side=Side.LONG,
            symbol="TEST",
            quantity=10,
            entry_fill="not-a-fill",  # type: ignore[arg-type]
            protective_exit=ProtectiveExitSpec(stop_price=Decimal("95")),
            candles=make_candles(),
        )


def test_market_exit_rejects_naive_requested_timestamp() -> None:
    with pytest.raises(ValueError, match="requested_at must be timezone-aware"):
        HistoricalExecutionSimulator().fill_market_exit(
            side=Side.LONG,
            symbol="TEST",
            quantity=10,
            requested_at=datetime(2024, 1, 2, 9, 20),
            exit_reason=ExitReason.MANUAL,
            candles=make_candles(),
        )


def test_market_exit_empty_valid_frame_returns_none() -> None:
    result = HistoricalExecutionSimulator().fill_market_exit(
        side=Side.LONG,
        symbol="TEST",
        quantity=10,
        requested_at=at(9, 20),
        exit_reason=ExitReason.MANUAL,
        candles=make_candles().clear(),
    )

    assert result is None


@pytest.mark.parametrize(
    "exit_reason",
    [ExitReason.STOP_LOSS, ExitReason.TARGET_REACHED],
)
def test_market_exit_rejects_protective_exit_reasons(exit_reason: ExitReason) -> None:
    with pytest.raises(ValueError, match="TIME_EXIT, STRATEGY_EXIT, or MANUAL"):
        HistoricalExecutionSimulator().fill_market_exit(
            side=Side.LONG,
            symbol="TEST",
            quantity=10,
            requested_at=at(9, 20),
            exit_reason=exit_reason,
            candles=make_candles(),
        )
