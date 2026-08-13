from datetime import date
from decimal import Decimal

import pytest

from algo_trader.costs import (
    BrokeragePlan,
    GSTTaxableComponent,
    IntradayCostSchedule,
    IntradayCostScheduleBook,
)


def make_schedule(
    schedule_id: str,
    effective_from: date,
    effective_to: date | None,
    *,
    plan: BrokeragePlan = BrokeragePlan.PLUS,
    brokerage_rate: Decimal = Decimal("0.001"),
    brokerage_minimum: Decimal = Decimal("5"),
    brokerage_maximum: Decimal | None = None,
) -> IntradayCostSchedule:
    return IntradayCostSchedule(
        schedule_id=schedule_id,
        effective_from=effective_from,
        effective_to=effective_to,
        brokerage_plan=plan,
        brokerage_rate=brokerage_rate,
        brokerage_minimum=brokerage_minimum,
        brokerage_maximum=(
            brokerage_maximum
            if brokerage_maximum is not None
            else (Decimal("20") if plan is BrokeragePlan.PLUS else Decimal("30"))
        ),
        exchange_transaction_rate=Decimal("0.000030699"),
        sebi_turnover_rate=Decimal("0.000001"),
        ipft_rate=Decimal("0.0000015"),
        stt_rate=Decimal("0.00025"),
        stamp_duty_rate=Decimal("0.00003"),
        gst_rate=Decimal("0.18"),
        gst_taxable_components=frozenset(GSTTaxableComponent),
    )


def test_effective_from_is_inclusive() -> None:
    schedule = make_schedule("current", date(2025, 1, 1), None)

    assert IntradayCostScheduleBook([schedule]).select(
        date(2025, 1, 1), BrokeragePlan.PLUS
    ) is schedule


def test_effective_to_is_exclusive() -> None:
    schedule = make_schedule("closed", date(2025, 1, 1), date(2025, 11, 17))
    book = IntradayCostScheduleBook([schedule])

    assert book.select(date(2025, 11, 16), BrokeragePlan.PLUS) is schedule
    with pytest.raises(LookupError, match="no cost schedule"):
        book.select(date(2025, 11, 17), BrokeragePlan.PLUS)


def test_adjacent_schedules_share_boundary_without_overlap() -> None:
    first = make_schedule("first", date(2025, 1, 1), date(2025, 11, 17))
    second = make_schedule("second", date(2025, 11, 17), None)
    book = IntradayCostScheduleBook([second, first])

    assert book.select(date(2025, 11, 16), BrokeragePlan.PLUS) is first
    assert book.select(date(2025, 11, 17), BrokeragePlan.PLUS) is second


def test_same_plan_can_have_different_brokerage_terms_across_adjacent_regimes() -> None:
    earlier = make_schedule(
        "earlier",
        date(2025, 1, 1),
        date(2025, 11, 17),
        brokerage_rate=Decimal("0.0005"),
        brokerage_minimum=Decimal("2"),
        brokerage_maximum=Decimal("15"),
    )
    later = make_schedule(
        "later",
        date(2025, 11, 17),
        None,
        brokerage_rate=Decimal("0.001"),
        brokerage_minimum=Decimal("5"),
        brokerage_maximum=Decimal("20"),
    )
    book = IntradayCostScheduleBook([earlier, later])

    assert book.select(date(2025, 11, 16), BrokeragePlan.PLUS).brokerage_rate == Decimal(
        "0.0005"
    )
    assert book.select(date(2025, 11, 17), BrokeragePlan.PLUS).brokerage_rate == Decimal(
        "0.001"
    )


def test_overlapping_schedules_for_same_plan_are_rejected() -> None:
    first = make_schedule("first", date(2025, 1, 1), date(2025, 12, 1))
    overlapping = make_schedule("overlap", date(2025, 11, 17), None)

    with pytest.raises(ValueError, match="must not overlap"):
        IntradayCostScheduleBook([first, overlapping])


def test_multiple_overlapping_open_ended_schedules_are_rejected() -> None:
    first = make_schedule("first", date(2025, 1, 1), None)
    second = make_schedule("second", date(2025, 11, 17), None)

    with pytest.raises(ValueError, match="must not overlap"):
        IntradayCostScheduleBook([first, second])


def test_same_dates_for_different_plans_are_selected_exactly() -> None:
    plus = make_schedule("plus", date(2025, 1, 1), None)
    pro_plus = make_schedule(
        "pro-plus",
        date(2025, 1, 1),
        None,
        plan=BrokeragePlan.PRO_PLUS,
    )
    book = IntradayCostScheduleBook([pro_plus, plus])

    assert book.select(date(2025, 5, 1), BrokeragePlan.PLUS) is plus
    assert book.select(date(2025, 5, 1), BrokeragePlan.PRO_PLUS) is pro_plus


@pytest.mark.parametrize(
    "unsupported_date",
    [date(2024, 12, 31), date(2025, 6, 1), date(2026, 1, 1)],
)
def test_dates_outside_covered_range_fail_without_latest_fallback(
    unsupported_date: date,
) -> None:
    schedule = make_schedule("limited", date(2025, 1, 1), date(2025, 6, 1))

    with pytest.raises(LookupError, match="no cost schedule"):
        IntradayCostScheduleBook([schedule]).select(
            unsupported_date,
            BrokeragePlan.PLUS,
        )


def test_missing_plan_fails_without_fallback() -> None:
    book = IntradayCostScheduleBook(
        [make_schedule("plus-only", date(2025, 1, 1), None)]
    )

    with pytest.raises(LookupError, match="PRO_PLUS"):
        book.select(date(2025, 5, 1), BrokeragePlan.PRO_PLUS)


def test_invalid_plan_type_is_rejected() -> None:
    book = IntradayCostScheduleBook(
        [make_schedule("plus-only", date(2025, 1, 1), None)]
    )

    with pytest.raises(TypeError, match="BrokeragePlan"):
        book.select(date(2025, 5, 1), "PLUS")  # type: ignore[arg-type]
