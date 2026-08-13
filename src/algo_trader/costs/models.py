"""Immutable models and effective-date schedules for intraday equity costs."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    computed_field,
    model_validator,
)

NonNegativeDecimal = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
StrictDate = Annotated[date, Field(strict=True)]


class BrokeragePlan(StrEnum):
    """Explicitly supported Angel One intraday brokerage plans."""

    PLUS = "PLUS"
    PRO_PLUS = "PRO_PLUS"


class TransactionAction(StrEnum):
    """Cost-owned transaction direction, distinct from LONG/SHORT position side."""

    BUY = "BUY"
    SELL = "SELL"


class GSTTaxableComponent(StrEnum):
    """Cost components that a schedule may include in its GST basis."""

    BROKERAGE = "brokerage"
    EXCHANGE_TRANSACTION_CHARGE = "exchange_transaction_charge"
    SEBI_TURNOVER_FEE = "sebi_turnover_fee"
    IPFT = "ipft"


class FrozenCostModel(BaseModel):
    """Validation policy for immutable cost-owned records."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class LegCostBreakdown(FrozenCostModel):
    """Exact costs charged on one executed BUY or SELL leg."""

    turnover: NonNegativeDecimal
    brokerage: NonNegativeDecimal
    exchange_transaction_charge: NonNegativeDecimal
    sebi_turnover_fee: NonNegativeDecimal
    ipft: NonNegativeDecimal
    stt: NonNegativeDecimal
    stamp_duty: NonNegativeDecimal
    gst: NonNegativeDecimal

    @computed_field
    @property
    def total(self) -> Decimal:
        """Exact sum of cost components; turnover is not itself a cost."""
        return sum(
            (
                self.brokerage,
                self.exchange_transaction_charge,
                self.sebi_turnover_fee,
                self.ipft,
                self.stt,
                self.stamp_duty,
                self.gst,
            ),
            start=Decimal("0"),
        )


class RoundTripCostBreakdown(FrozenCostModel):
    """Exact entry and exit costs calculated under one identified schedule."""

    schedule_id: NonEmptyStr
    entry: LegCostBreakdown
    exit: LegCostBreakdown

    @computed_field
    @property
    def total(self) -> Decimal:
        """Exact sum of entry and exit costs."""
        return self.entry.total + self.exit.total


class IntradayCostSchedule(FrozenCostModel):
    """Generic effective-date regime record over ``[from, to)`` dates.

    All percentage-like values are Decimal fractions of turnover: for example,
    ``0.00025`` means 0.025%. IPFT is deliberately required with no default.
    Correct historical rates and GST policy come from a separately verified
    schedule catalogue, never from hidden assumptions about the plan name.
    """

    schedule_id: NonEmptyStr
    effective_from: StrictDate
    effective_to: StrictDate | None = None
    brokerage_plan: BrokeragePlan
    brokerage_rate: NonNegativeDecimal
    brokerage_minimum: NonNegativeDecimal
    brokerage_maximum: NonNegativeDecimal
    exchange_transaction_rate: NonNegativeDecimal
    sebi_turnover_rate: NonNegativeDecimal
    ipft_rate: NonNegativeDecimal
    stt_rate: NonNegativeDecimal
    stamp_duty_rate: NonNegativeDecimal
    gst_rate: NonNegativeDecimal
    gst_taxable_components: frozenset[GSTTaxableComponent]

    @model_validator(mode="after")
    def validate_schedule(self) -> IntradayCostSchedule:
        if self.effective_to is not None and self.effective_to <= self.effective_from:
            raise ValueError("effective_to must be later than effective_from")
        if self.brokerage_maximum < self.brokerage_minimum:
            raise ValueError("brokerage_maximum must be at least brokerage_minimum")
        return self

    def is_effective_on(self, trading_date: date) -> bool:
        """Return whether a date is inside this schedule's half-open range."""
        return self.effective_from <= trading_date and (
            self.effective_to is None or trading_date < self.effective_to
        )


class IntradayCostScheduleBook:
    """Deterministic plan/date selector for non-overlapping schedules."""

    def __init__(self, schedules: Iterable[IntradayCostSchedule]) -> None:
        selected = tuple(schedules)
        if not selected:
            raise ValueError("at least one cost schedule is required")
        if any(not isinstance(schedule, IntradayCostSchedule) for schedule in selected):
            raise TypeError("all schedules must be IntradayCostSchedule instances")

        schedule_ids = [schedule.schedule_id for schedule in selected]
        if len(schedule_ids) != len(set(schedule_ids)):
            raise ValueError("schedule_id values must be unique")

        ordered = tuple(
            sorted(
                selected,
                key=lambda schedule: (
                    schedule.brokerage_plan.value,
                    schedule.effective_from,
                ),
            )
        )
        self._validate_non_overlapping(ordered)
        self._schedules = ordered

    @property
    def schedules(self) -> tuple[IntradayCostSchedule, ...]:
        """Return the immutable, deterministic schedule ordering."""
        return self._schedules

    def select(self, trading_date: date, plan: BrokeragePlan) -> IntradayCostSchedule:
        """Select an exact plan/date match without any latest-rate fallback."""
        if isinstance(trading_date, datetime) or not isinstance(trading_date, date):
            raise TypeError("trading_date must be a date")
        if not isinstance(plan, BrokeragePlan):
            raise TypeError("plan must be a BrokeragePlan")

        for schedule in self._schedules:
            if schedule.brokerage_plan is plan and schedule.is_effective_on(trading_date):
                return schedule
        raise LookupError(f"no cost schedule for plan {plan.value} on {trading_date.isoformat()}")

    @staticmethod
    def _validate_non_overlapping(
        schedules: tuple[IntradayCostSchedule, ...],
    ) -> None:
        previous_by_plan: dict[BrokeragePlan, IntradayCostSchedule] = {}
        for schedule in schedules:
            previous = previous_by_plan.get(schedule.brokerage_plan)
            if previous is not None and (
                previous.effective_to is None
                or schedule.effective_from < previous.effective_to
            ):
                raise ValueError(
                    "cost schedule ranges must not overlap within a brokerage plan"
                )
            previous_by_plan[schedule.brokerage_plan] = schedule
