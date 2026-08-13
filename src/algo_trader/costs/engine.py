"""Exact NSE equity-intraday transaction-cost calculations."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from zoneinfo import ZoneInfo

from algo_trader.costs.models import (
    BrokeragePlan,
    GSTTaxableComponent,
    IntradayCostSchedule,
    IntradayCostScheduleBook,
    LegCostBreakdown,
    RoundTripCostBreakdown,
    TransactionAction,
)
from algo_trader.domain import Fill, Side

MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")


def calculate_leg_costs(
    *,
    fill: Fill,
    action: TransactionAction,
    schedule: IntradayCostSchedule,
) -> LegCostBreakdown:
    """Calculate exact costs for one executed order under an effective schedule."""
    _validate_fill(fill, "fill")
    if not isinstance(action, TransactionAction):
        raise TypeError("action must be a TransactionAction")
    if not isinstance(schedule, IntradayCostSchedule):
        raise TypeError("schedule must be an IntradayCostSchedule")

    trading_date = _market_date(fill)
    _validate_schedule_date(schedule, trading_date)
    turnover = fill.price * fill.quantity
    if turnover <= 0:
        raise ValueError("turnover must be positive")

    brokerage = min(
        max(turnover * schedule.brokerage_rate, schedule.brokerage_minimum),
        schedule.brokerage_maximum,
    )
    exchange_transaction_charge = turnover * schedule.exchange_transaction_rate
    sebi_turnover_fee = turnover * schedule.sebi_turnover_rate
    ipft = turnover * schedule.ipft_rate
    stt = turnover * schedule.stt_rate if action is TransactionAction.SELL else Decimal("0")
    stamp_duty = (
        turnover * schedule.stamp_duty_rate
        if action is TransactionAction.BUY
        else Decimal("0")
    )

    taxable_values = {
        GSTTaxableComponent.BROKERAGE: brokerage,
        GSTTaxableComponent.EXCHANGE_TRANSACTION_CHARGE: exchange_transaction_charge,
        GSTTaxableComponent.SEBI_TURNOVER_FEE: sebi_turnover_fee,
        GSTTaxableComponent.IPFT: ipft,
    }
    gst_taxable_basis = sum(
        (
            taxable_values[component]
            for component in schedule.gst_taxable_components
        ),
        start=Decimal("0"),
    )
    gst = gst_taxable_basis * schedule.gst_rate

    return LegCostBreakdown(
        turnover=turnover,
        brokerage=brokerage,
        exchange_transaction_charge=exchange_transaction_charge,
        sebi_turnover_fee=sebi_turnover_fee,
        ipft=ipft,
        stt=stt,
        stamp_duty=stamp_duty,
        gst=gst,
    )


def calculate_round_trip_costs(
    *,
    side: Side,
    entry_fill: Fill,
    exit_fill: Fill,
    schedule: IntradayCostSchedule,
) -> RoundTripCostBreakdown:
    """Calculate exact entry and exit costs using one explicit schedule."""
    trading_date = _validate_round_trip(side, entry_fill, exit_fill)
    if not isinstance(schedule, IntradayCostSchedule):
        raise TypeError("schedule must be an IntradayCostSchedule")
    _validate_schedule_date(schedule, trading_date)

    entry_action, exit_action = _actions_for_side(side)
    entry_costs = calculate_leg_costs(
        fill=entry_fill,
        action=entry_action,
        schedule=schedule,
    )
    exit_costs = calculate_leg_costs(
        fill=exit_fill,
        action=exit_action,
        schedule=schedule,
    )
    return RoundTripCostBreakdown(
        schedule_id=schedule.schedule_id,
        entry=entry_costs,
        exit=exit_costs,
    )


def calculate_round_trip_costs_from_book(
    *,
    side: Side,
    entry_fill: Fill,
    exit_fill: Fill,
    plan: BrokeragePlan,
    schedule_book: IntradayCostScheduleBook,
) -> RoundTripCostBreakdown:
    """Select the exact plan/date schedule, then calculate round-trip costs."""
    trading_date = _validate_round_trip(side, entry_fill, exit_fill)
    if not isinstance(schedule_book, IntradayCostScheduleBook):
        raise TypeError("schedule_book must be an IntradayCostScheduleBook")
    schedule = schedule_book.select(trading_date, plan)
    return calculate_round_trip_costs(
        side=side,
        entry_fill=entry_fill,
        exit_fill=exit_fill,
        schedule=schedule,
    )


def _validate_round_trip(side: Side, entry_fill: Fill, exit_fill: Fill) -> date:
    if not isinstance(side, Side):
        raise TypeError("side must be a Side")
    _validate_fill(entry_fill, "entry_fill")
    _validate_fill(exit_fill, "exit_fill")
    if entry_fill.quantity != exit_fill.quantity:
        raise ValueError("entry and exit fill quantities must match")
    if exit_fill.timestamp < entry_fill.timestamp:
        raise ValueError("exit fill timestamp cannot precede entry fill timestamp")

    entry_date = _market_date(entry_fill)
    exit_date = _market_date(exit_fill)
    if entry_date != exit_date:
        raise ValueError("entry and exit fills must occur on the same Asia/Kolkata date")
    return entry_date


def _validate_fill(fill: Fill, name: str) -> None:
    if not isinstance(fill, Fill):
        raise TypeError(f"{name} must be a Fill")


def _market_date(fill: Fill) -> date:
    return fill.timestamp.astimezone(MARKET_TIMEZONE).date()


def _validate_schedule_date(schedule: IntradayCostSchedule, trading_date: date) -> None:
    if not schedule.is_effective_on(trading_date):
        raise LookupError(
            f"schedule {schedule.schedule_id!r} does not support {trading_date.isoformat()}"
        )


def _actions_for_side(side: Side) -> tuple[TransactionAction, TransactionAction]:
    if side is Side.LONG:
        return TransactionAction.BUY, TransactionAction.SELL
    return TransactionAction.SELL, TransactionAction.BUY

