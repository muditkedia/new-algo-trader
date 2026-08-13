"""Explicit fixed-current transaction-cost policies for historical backtests."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from algo_trader.costs.models import (
    BrokeragePlan,
    GSTTaxableComponent,
    IntradayCostSchedule,
)

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
StrictDate = Annotated[date, Field(strict=True)]

BACKTEST_COST_POLICY_SOURCE_DATE = date(2026, 8, 14)
_POLICY_ID_PREFIX = "angel-one-nse-intraday-backtest-2026-08-14"
_GST_COMPONENTS = frozenset(
    {
        GSTTaxableComponent.BROKERAGE,
        GSTTaxableComponent.EXCHANGE_TRANSACTION_CHARGE,
        GSTTaxableComponent.SEBI_TURNOVER_FEE,
        GSTTaxableComponent.IPFT,
    }
)


class BacktestCostPolicy(BaseModel):
    """Immutable fixed snapshot used to model transaction costs in backtests.

    The wrapped schedule's ``date.min`` to open-ended range means universal
    backtest model applicability. It does not claim that these charges were
    historically effective throughout that range.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_id: NonEmptyStr
    source_as_of_date: StrictDate
    brokerage_plan: BrokeragePlan
    schedule: IntradayCostSchedule

    @model_validator(mode="after")
    def validate_policy(self) -> BacktestCostPolicy:
        if self.schedule.brokerage_plan is not self.brokerage_plan:
            raise ValueError("schedule brokerage_plan must match policy brokerage_plan")
        return self


def _make_policy(plan: BrokeragePlan, brokerage_maximum: Decimal) -> BacktestCostPolicy:
    plan_slug = plan.value.lower().replace("_", "-")
    policy_id = f"{_POLICY_ID_PREFIX}-{plan_slug}"
    schedule = IntradayCostSchedule(
        schedule_id=policy_id,
        effective_from=date.min,
        effective_to=None,
        brokerage_plan=plan,
        brokerage_rate=Decimal("0.001"),
        brokerage_minimum=Decimal("5"),
        brokerage_maximum=brokerage_maximum,
        exchange_transaction_rate=Decimal("0.000030699"),
        sebi_turnover_rate=Decimal("0.000001"),
        ipft_rate=Decimal("0.000000001"),
        stt_rate=Decimal("0.00025"),
        stamp_duty_rate=Decimal("0.00003"),
        gst_rate=Decimal("0.18"),
        gst_taxable_components=_GST_COMPONENTS,
    )
    return BacktestCostPolicy(
        policy_id=policy_id,
        source_as_of_date=BACKTEST_COST_POLICY_SOURCE_DATE,
        brokerage_plan=plan,
        schedule=schedule,
    )


_FIXED_POLICIES = {
    BrokeragePlan.PLUS: _make_policy(BrokeragePlan.PLUS, Decimal("20")),
    BrokeragePlan.PRO_PLUS: _make_policy(BrokeragePlan.PRO_PLUS, Decimal("30")),
}


def get_fixed_current_backtest_cost_policy(
    plan: BrokeragePlan,
) -> BacktestCostPolicy:
    """Return the immutable 2026-08-14 cost snapshot for one brokerage plan."""
    if not isinstance(plan, BrokeragePlan):
        raise TypeError("plan must be a BrokeragePlan")
    return _FIXED_POLICIES[plan]
