from datetime import date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from algo_trader import Fill, Side
from algo_trader.costs import (
    BACKTEST_COST_POLICY_SOURCE_DATE,
    BacktestCostPolicy,
    BrokeragePlan,
    GSTTaxableComponent,
    IntradayCostSchedule,
    backtest_policy,
    calculate_round_trip_costs,
    get_fixed_current_backtest_cost_policy,
)

MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")
EXPECTED_GST_COMPONENTS = frozenset(GSTTaxableComponent)


def make_fill(year: int, hour: int, price: str) -> Fill:
    return Fill(
        timestamp=datetime(year, 1, 2, hour, 20, tzinfo=MARKET_TIMEZONE),
        price=Decimal(price),
        quantity=10,
        is_simulated=True,
    )


@pytest.mark.parametrize(
    ("plan", "policy_id", "brokerage_maximum"),
    [
        (
            BrokeragePlan.PLUS,
            "angel-one-nse-intraday-backtest-2026-08-14-plus",
            Decimal("20"),
        ),
        (
            BrokeragePlan.PRO_PLUS,
            "angel-one-nse-intraday-backtest-2026-08-14-pro-plus",
            Decimal("30"),
        ),
    ],
)
def test_fixed_policy_contains_exact_supplied_snapshot(
    plan: BrokeragePlan,
    policy_id: str,
    brokerage_maximum: Decimal,
) -> None:
    policy = get_fixed_current_backtest_cost_policy(plan)
    schedule = policy.schedule

    assert policy.policy_id == policy_id
    assert policy.source_as_of_date == BACKTEST_COST_POLICY_SOURCE_DATE == date(2026, 8, 14)
    assert policy.brokerage_plan is schedule.brokerage_plan is plan
    assert schedule.effective_from == date.min
    assert schedule.effective_to is None
    assert schedule.brokerage_rate == Decimal("0.001")
    assert schedule.brokerage_minimum == Decimal("5")
    assert schedule.brokerage_maximum == brokerage_maximum
    assert schedule.exchange_transaction_rate == Decimal("0.000030699")
    assert schedule.sebi_turnover_rate == Decimal("0.000001")
    assert schedule.ipft_rate == Decimal("0.000000001")
    assert schedule.stt_rate == Decimal("0.00025")
    assert schedule.stamp_duty_rate == Decimal("0.00003")
    assert schedule.gst_rate == Decimal("0.18")
    assert schedule.gst_taxable_components == EXPECTED_GST_COMPONENTS


def test_fixed_plan_policy_ids_are_distinct() -> None:
    plus = get_fixed_current_backtest_cost_policy(BrokeragePlan.PLUS)
    pro_plus = get_fixed_current_backtest_cost_policy(BrokeragePlan.PRO_PLUS)

    assert plus.policy_id != pro_plus.policy_id


def test_policy_validates_source_date_and_matching_plan() -> None:
    plus = get_fixed_current_backtest_cost_policy(BrokeragePlan.PLUS)

    with pytest.raises(ValidationError, match="valid date"):
        BacktestCostPolicy(
            policy_id="invalid-date",
            source_as_of_date=datetime(2026, 8, 14),
            brokerage_plan=BrokeragePlan.PLUS,
            schedule=plus.schedule,
        )
    with pytest.raises(ValidationError, match="must match"):
        BacktestCostPolicy(
            policy_id="mismatch",
            source_as_of_date=date(2026, 8, 14),
            brokerage_plan=BrokeragePlan.PRO_PLUS,
            schedule=plus.schedule,
        )


def test_policy_module_introduces_no_duplicate_cost_calculator() -> None:
    assert not any(name.startswith("calculate_") for name in vars(backtest_policy))


@pytest.mark.parametrize("year", [2015, 2026])
def test_fixed_policy_calculates_round_trip_on_far_separated_dates(year: int) -> None:
    result = calculate_round_trip_costs(
        side=Side.LONG,
        entry_fill=make_fill(year, 9, "100"),
        exit_fill=make_fill(year, 15, "101"),
        schedule=get_fixed_current_backtest_cost_policy(BrokeragePlan.PLUS).schedule,
    )

    assert result.total > 0


def test_otherwise_identical_historical_fills_use_identical_policy_rates() -> None:
    policy = get_fixed_current_backtest_cost_policy(BrokeragePlan.PRO_PLUS)
    old = calculate_round_trip_costs(
        side=Side.SHORT,
        entry_fill=make_fill(2015, 9, "100"),
        exit_fill=make_fill(2015, 15, "99"),
        schedule=policy.schedule,
    )
    recent = calculate_round_trip_costs(
        side=Side.SHORT,
        entry_fill=make_fill(2026, 9, "100"),
        exit_fill=make_fill(2026, 15, "99"),
        schedule=policy.schedule,
    )

    assert old == recent


def test_ordinary_finite_schedule_still_rejects_unsupported_date() -> None:
    fixed = get_fixed_current_backtest_cost_policy(BrokeragePlan.PLUS).schedule
    finite = IntradayCostSchedule(
        **fixed.model_dump(
            exclude={"schedule_id", "effective_from", "effective_to"}
        ),
        schedule_id="ordinary-finite-range",
        effective_from=date(2026, 1, 1),
        effective_to=date(2027, 1, 1),
    )

    with pytest.raises(LookupError, match="does not support"):
        calculate_round_trip_costs(
            side=Side.LONG,
            entry_fill=make_fill(2015, 9, "100"),
            exit_fill=make_fill(2015, 15, "101"),
            schedule=finite,
        )
