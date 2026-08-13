from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from algo_trader import (
    ExitReason,
    Fill,
    MLScore,
    Side,
    Signal,
    SignalStatus,
    Trade,
)
from algo_trader.costs import (
    BrokeragePlan,
    GSTTaxableComponent,
    IntradayCostSchedule,
    IntradayCostScheduleBook,
    TransactionAction,
    calculate_leg_costs,
    calculate_round_trip_costs,
    calculate_round_trip_costs_from_book,
)

MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")
GST_COMPONENTS = frozenset(GSTTaxableComponent)


def at(hour: int, minute: int, *, day: int = 2) -> datetime:
    return datetime(2025, 1, day, hour, minute, tzinfo=MARKET_TIMEZONE)


def make_fill(
    price: str,
    *,
    quantity: int = 1,
    timestamp: datetime | None = None,
    slippage_per_unit: str = "0",
) -> Fill:
    return Fill(
        timestamp=timestamp or at(9, 20),
        price=Decimal(price),
        quantity=quantity,
        slippage_per_unit=Decimal(slippage_per_unit),
        is_simulated=True,
    )


def make_schedule(
    plan: BrokeragePlan = BrokeragePlan.PLUS,
    **updates,
) -> IntradayCostSchedule:
    values = {
        "schedule_id": f"fixture-{plan.value.lower()}",
        "effective_from": datetime(2025, 1, 1).date(),
        "effective_to": None,
        "brokerage_plan": plan,
        "brokerage_rate": Decimal("0.001"),
        "brokerage_minimum": Decimal("5"),
        "brokerage_maximum": (
            Decimal("20") if plan is BrokeragePlan.PLUS else Decimal("30")
        ),
        # Decimal turnover fractions: 0.0030699%, Rs 10/crore, explicit fixture IPFT,
        # 0.025% sell STT, 0.003% buy stamp duty, and 18% GST.
        "exchange_transaction_rate": Decimal("0.000030699"),
        "sebi_turnover_rate": Decimal("0.000001"),
        "ipft_rate": Decimal("0.0000015"),
        "stt_rate": Decimal("0.00025"),
        "stamp_duty_rate": Decimal("0.00003"),
        "gst_rate": Decimal("0.18"),
        "gst_taxable_components": GST_COMPONENTS,
    }
    values.update(updates)
    return IntradayCostSchedule(**values)


@pytest.mark.parametrize(
    ("plan", "turnover", "expected_brokerage"),
    [
        (BrokeragePlan.PLUS, "1000", Decimal("5")),
        (BrokeragePlan.PLUS, "10000", Decimal("10")),
        (BrokeragePlan.PLUS, "50000", Decimal("20")),
        (BrokeragePlan.PRO_PLUS, "1000", Decimal("5")),
        (BrokeragePlan.PRO_PLUS, "10000", Decimal("10")),
        (BrokeragePlan.PRO_PLUS, "50000", Decimal("30")),
    ],
)
def test_brokerage_minimum_percentage_and_plan_cap(
    plan: BrokeragePlan,
    turnover: str,
    expected_brokerage: Decimal,
) -> None:
    breakdown = calculate_leg_costs(
        fill=make_fill(turnover),
        action=TransactionAction.BUY,
        schedule=make_schedule(plan),
    )

    assert breakdown.brokerage == expected_brokerage


def test_generic_schedule_uses_explicit_historical_brokerage_terms() -> None:
    schedule = make_schedule(
        brokerage_rate=Decimal("0.0005"),
        brokerage_minimum=Decimal("2"),
        brokerage_maximum=Decimal("15"),
    )

    breakdown = calculate_leg_costs(
        fill=make_fill("10000"),
        action=TransactionAction.BUY,
        schedule=schedule,
    )

    assert breakdown.brokerage == Decimal("5.0000")


def test_schedule_rejects_brokerage_maximum_below_minimum() -> None:
    with pytest.raises(ValidationError, match="maximum must be at least"):
        make_schedule(
            brokerage_minimum=Decimal("10"),
            brokerage_maximum=Decimal("5"),
        )


def test_long_round_trip_charge_direction_and_exact_gst_basis() -> None:
    schedule = make_schedule()
    result = calculate_round_trip_costs(
        side=Side.LONG,
        entry_fill=make_fill("100", quantity=10),
        exit_fill=make_fill("110", quantity=10, timestamp=at(15, 20)),
        schedule=schedule,
    )

    assert result.entry.turnover == Decimal("1000")
    assert result.exit.turnover == Decimal("1100")
    assert result.entry.stamp_duty == Decimal("1000") * schedule.stamp_duty_rate
    assert result.entry.stt == 0
    assert result.exit.stamp_duty == 0
    assert result.exit.stt == Decimal("1100") * schedule.stt_rate
    _assert_two_sided_components(result.entry, schedule)
    _assert_two_sided_components(result.exit, schedule)
    _assert_exact_gst_basis(result.entry, schedule)
    _assert_exact_gst_basis(result.exit, schedule)


def test_short_round_trip_charge_direction_and_exact_gst_basis() -> None:
    schedule = make_schedule()
    result = calculate_round_trip_costs(
        side=Side.SHORT,
        entry_fill=make_fill("100", quantity=10),
        exit_fill=make_fill("90", quantity=10, timestamp=at(15, 20)),
        schedule=schedule,
    )

    assert result.entry.stt == Decimal("1000") * schedule.stt_rate
    assert result.entry.stamp_duty == 0
    assert result.exit.stt == 0
    assert result.exit.stamp_duty == Decimal("900") * schedule.stamp_duty_rate
    _assert_two_sided_components(result.entry, schedule)
    _assert_two_sided_components(result.exit, schedule)
    _assert_exact_gst_basis(result.entry, schedule)
    _assert_exact_gst_basis(result.exit, schedule)


def test_leg_and_round_trip_totals_reconcile_exactly() -> None:
    result = calculate_round_trip_costs(
        side=Side.LONG,
        entry_fill=make_fill("100", quantity=10),
        exit_fill=make_fill("101", quantity=10, timestamp=at(15, 20)),
        schedule=make_schedule(),
    )
    entry_components = (
        result.entry.brokerage
        + result.entry.exchange_transaction_charge
        + result.entry.sebi_turnover_fee
        + result.entry.ipft
        + result.entry.stt
        + result.entry.stamp_duty
        + result.entry.gst
    )

    assert result.entry.total == entry_components
    assert result.total == result.entry.total + result.exit.total


def test_turnover_uses_final_fill_price_and_does_not_add_slippage_again() -> None:
    schedule = make_schedule()
    slipped_fill = make_fill(
        "100.10",
        quantity=10,
        slippage_per_unit="0.10",
    )
    same_price_without_slippage_metadata = make_fill("100.10", quantity=10)

    slipped_costs = calculate_leg_costs(
        fill=slipped_fill,
        action=TransactionAction.BUY,
        schedule=schedule,
    )
    comparison_costs = calculate_leg_costs(
        fill=same_price_without_slippage_metadata,
        action=TransactionAction.BUY,
        schedule=schedule,
    )

    assert slipped_costs.turnover == Decimal("1001.00")
    assert slipped_costs == comparison_costs
    assert "slippage" not in type(slipped_costs).model_fields


def test_round_trip_total_is_compatible_with_trade_total_costs() -> None:
    entry_fill = make_fill("100", quantity=10)
    exit_fill = make_fill("110", quantity=10, timestamp=at(15, 20))
    costs = calculate_round_trip_costs(
        side=Side.LONG,
        entry_fill=entry_fill,
        exit_fill=exit_fill,
        schedule=make_schedule(),
    )
    signal = Signal(
        strategy_id="compatibility-test",
        strategy_version="1",
        symbol="TEST",
        timestamp=at(9, 15),
        side=Side.LONG,
        status=SignalStatus.EXECUTED,
    )
    ml_score = MLScore(
        model_version="test",
        quality_score=Decimal("0.5"),
        calibrated_probability=Decimal("0.5"),
        predicted_net_return=Decimal("0.01"),
        recommended_notional=50_000,
    )
    gross_pnl = Decimal("1000")

    trade = Trade(
        signal=signal,
        ml_score=ml_score,
        target_notional=50_000,
        entry_fill=entry_fill,
        exit_fill=exit_fill,
        gross_pnl=gross_pnl,
        total_costs=costs.total,
        net_pnl=gross_pnl - costs.total,
        mfe_return=Decimal("0.02"),
        mae_return=Decimal("-0.01"),
        exit_reason=ExitReason.STRATEGY_EXIT,
    )

    assert trade.total_costs == costs.total


def test_round_trip_rejects_mismatched_quantities() -> None:
    with pytest.raises(ValueError, match="quantities must match"):
        calculate_round_trip_costs(
            side=Side.LONG,
            entry_fill=make_fill("100", quantity=10),
            exit_fill=make_fill("101", quantity=9, timestamp=at(15, 20)),
            schedule=make_schedule(),
        )


def test_round_trip_rejects_exit_before_entry() -> None:
    with pytest.raises(ValueError, match="cannot precede"):
        calculate_round_trip_costs(
            side=Side.LONG,
            entry_fill=make_fill("100", timestamp=at(10, 0)),
            exit_fill=make_fill("101", timestamp=at(9, 55)),
            schedule=make_schedule(),
        )


def test_round_trip_rejects_overnight_position_in_market_timezone() -> None:
    with pytest.raises(ValueError, match="same Asia/Kolkata date"):
        calculate_round_trip_costs(
            side=Side.LONG,
            entry_fill=make_fill("100", timestamp=at(15, 25)),
            exit_fill=make_fill("101", timestamp=at(9, 15, day=3)),
            schedule=make_schedule(),
        )


def test_cost_calculation_does_not_mutate_source_fills() -> None:
    entry_fill = make_fill("100", quantity=10)
    exit_fill = make_fill("101", quantity=10, timestamp=at(15, 20))
    entry_before = entry_fill.model_dump()
    exit_before = exit_fill.model_dump()

    calculate_round_trip_costs(
        side=Side.LONG,
        entry_fill=entry_fill,
        exit_fill=exit_fill,
        schedule=make_schedule(),
    )

    assert entry_fill.model_dump() == entry_before
    assert exit_fill.model_dump() == exit_before


@pytest.mark.parametrize(
    "field",
    [
        "brokerage_rate",
        "brokerage_minimum",
        "brokerage_maximum",
        "exchange_transaction_rate",
        "ipft_rate",
        "gst_rate",
    ],
)
@pytest.mark.parametrize("value", [Decimal("-0.1"), Decimal("NaN"), Decimal("Infinity")])
def test_schedule_rejects_negative_and_non_finite_rates(
    field: str,
    value: Decimal,
) -> None:
    with pytest.raises(ValidationError):
        make_schedule(**{field: value})


def test_schedule_requires_explicit_ipft_rate() -> None:
    values = make_schedule().model_dump(exclude={"ipft_rate"})

    with pytest.raises(ValidationError, match="ipft_rate"):
        IntradayCostSchedule(**values)


def test_historical_gst_policy_taxes_only_explicitly_selected_components() -> None:
    brokerage_only = frozenset({GSTTaxableComponent.BROKERAGE})
    schedule = make_schedule(gst_taxable_components=brokerage_only)

    breakdown = calculate_leg_costs(
        fill=make_fill("10000"),
        action=TransactionAction.BUY,
        schedule=schedule,
    )

    assert breakdown.gst == breakdown.brokerage * schedule.gst_rate
    assert breakdown.exchange_transaction_charge > 0
    assert breakdown.sebi_turnover_fee > 0
    assert breakdown.ipft > 0


def test_empty_gst_taxable_component_set_is_accepted() -> None:
    schedule = make_schedule(gst_taxable_components=frozenset())

    assert schedule.gst_taxable_components == frozenset()


def test_zero_gst_rate_with_empty_taxable_set_produces_zero_gst() -> None:
    schedule = make_schedule(
        gst_rate=Decimal("0"),
        gst_taxable_components=frozenset(),
    )

    breakdown = calculate_leg_costs(
        fill=make_fill("10000"),
        action=TransactionAction.SELL,
        schedule=schedule,
    )

    assert breakdown.gst == Decimal("0")


def test_schedule_book_calculation_retains_exact_selected_schedule_id() -> None:
    schedule = make_schedule(schedule_id="selected-plus")
    result = calculate_round_trip_costs_from_book(
        side=Side.LONG,
        entry_fill=make_fill("100", quantity=10),
        exit_fill=make_fill("101", quantity=10, timestamp=at(15, 20)),
        plan=BrokeragePlan.PLUS,
        schedule_book=IntradayCostScheduleBook([schedule]),
    )

    assert result.schedule_id == "selected-plus"


def test_direct_calculation_rejects_schedule_outside_trade_date() -> None:
    schedule = make_schedule(
        effective_from=datetime(2025, 2, 1).date(),
        effective_to=None,
    )

    with pytest.raises(LookupError, match="does not support"):
        calculate_round_trip_costs(
            side=Side.LONG,
            entry_fill=make_fill("100"),
            exit_fill=make_fill("101", timestamp=at(15, 20)),
            schedule=schedule,
        )


def _assert_two_sided_components(leg, schedule: IntradayCostSchedule) -> None:
    assert leg.exchange_transaction_charge == leg.turnover * schedule.exchange_transaction_rate
    assert leg.sebi_turnover_fee == leg.turnover * schedule.sebi_turnover_rate
    assert leg.ipft == leg.turnover * schedule.ipft_rate


def _assert_exact_gst_basis(leg, schedule: IntradayCostSchedule) -> None:
    taxable_basis = (
        leg.brokerage
        + leg.exchange_transaction_charge
        + leg.sebi_turnover_fee
        + leg.ipft
    )
    assert leg.gst == taxable_basis * schedule.gst_rate
    assert leg.gst != (taxable_basis + leg.stt + leg.stamp_duty) * schedule.gst_rate
