"""Immutable requests, records, and results for historical backtests."""

from __future__ import annotations

from datetime import date, datetime, time
from decimal import Decimal
from enum import StrEnum
from typing import Annotated
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from algo_trader.costs import BrokeragePlan, RoundTripCostBreakdown
from algo_trader.domain import ProtectiveExitSpec, Trade
from algo_trader.portfolio import (
    AllocationCandidate,
    AllocationDecision,
    CandidateIdentity,
    PortfolioState,
)

MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
StrictPositiveDecimal = Annotated[
    Decimal,
    Field(strict=True, gt=0, allow_inf_nan=False),
]


def _validate_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


class FrozenBacktestModel(BaseModel):
    """Validation policy for immutable backtest-owned records."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class BacktestTradeRequest(FrozenBacktestModel):
    """Fully specified trade intent supplied to the orchestration layer."""

    candidate: AllocationCandidate
    protective_exit: ProtectiveExitSpec | None = None
    strategy_exit_at: datetime | None = None

    @model_validator(mode="after")
    def validate_strategy_exit(self) -> BacktestTradeRequest:
        if self.strategy_exit_at is None:
            return self
        _validate_aware(self.strategy_exit_at, "strategy_exit_at")
        order_timestamp = self.candidate.order_intent.timestamp
        if self.strategy_exit_at < order_timestamp:
            raise ValueError("strategy_exit_at cannot precede order_intent.timestamp")
        if (
            self.strategy_exit_at.astimezone(MARKET_TIMEZONE).date()
            != order_timestamp.astimezone(MARKET_TIMEZONE).date()
        ):
            raise ValueError("strategy_exit_at must be on the candidate's trading date")
        return self


class BacktestConfig(FrozenBacktestModel):
    """Explicit reproducibility and capital settings for one backtest run."""

    run_id: NonEmptyStr
    git_commit: NonEmptyStr
    window_start: datetime
    window_end: datetime
    brokerage_plan: BrokeragePlan
    initial_capital: StrictPositiveDecimal = Decimal("100000")
    forced_exit_time: time = time(15, 25)

    @model_validator(mode="after")
    def validate_config(self) -> BacktestConfig:
        _validate_aware(self.window_start, "window_start")
        _validate_aware(self.window_end, "window_end")
        if self.window_start >= self.window_end:
            raise ValueError("window_start must be earlier than window_end")
        if not isinstance(self.forced_exit_time, time):
            raise TypeError("forced_exit_time must be a time")
        if self.forced_exit_time.tzinfo is not None:
            raise ValueError("forced_exit_time must be an Asia/Kolkata local clock time")
        return self


class BacktestRequestOutcome(StrEnum):
    """Exhaustive terminal outcomes for accepted backtest requests."""

    COMPLETED_ACTUAL = "COMPLETED_ACTUAL"
    COMPLETED_SHADOW = "COMPLETED_SHADOW"
    ALLOCATED_ENTRY_NOT_FILLED = "ALLOCATED_ENTRY_NOT_FILLED"
    SHADOW_ENTRY_NOT_FILLED = "SHADOW_ENTRY_NOT_FILLED"
    CAPITAL_EXHAUSTED = "CAPITAL_EXHAUSTED"


class BacktestTradeRecord(FrozenBacktestModel):
    """Completed trade plus exact cost and allocation provenance."""

    trade: Trade
    round_trip_cost_breakdown: RoundTripCostBreakdown
    cost_policy_id: NonEmptyStr
    allocation_identity: CandidateIdentity
    allocation_decision: AllocationDecision

    @model_validator(mode="after")
    def validate_provenance(self) -> BacktestTradeRecord:
        if self.trade.total_costs != self.round_trip_cost_breakdown.total:
            raise ValueError("trade total_costs must match the round-trip cost total")
        if self.cost_policy_id != self.round_trip_cost_breakdown.schedule_id:
            raise ValueError("cost policy ID must match the cost schedule ID")
        if self.allocation_identity != self.allocation_decision.candidate.identity:
            raise ValueError("allocation identity must match the allocation decision")
        if self.trade.is_shadow != self.allocation_decision.requires_shadow_tracking:
            raise ValueError("trade shadow status must match the allocation decision")
        if self.trade.ml_score != self.allocation_decision.candidate.ml_score:
            raise ValueError("trade MLScore must match the allocation decision candidate")
        if (
            self.trade.target_notional
            != self.allocation_decision.candidate.target_notional
        ):
            raise ValueError(
                "trade target_notional must match the allocation decision candidate"
            )
        return self


class BacktestRequestResult(FrozenBacktestModel):
    """One auditable terminal result for one input trade request."""

    request: BacktestTradeRequest
    outcome: BacktestRequestOutcome
    terminal_at: datetime
    allocation_decision: AllocationDecision | None = None
    trade_record: BacktestTradeRecord | None = None

    @model_validator(mode="after")
    def validate_terminal_result(self) -> BacktestRequestResult:
        _validate_aware(self.terminal_at, "terminal_at")
        if self.terminal_at < self.request.candidate.order_intent.timestamp:
            raise ValueError("terminal_at cannot precede the request order timestamp")
        completed = self.outcome in {
            BacktestRequestOutcome.COMPLETED_ACTUAL,
            BacktestRequestOutcome.COMPLETED_SHADOW,
        }
        if completed != (self.trade_record is not None):
            raise ValueError("only completed request results may contain a trade record")
        if self.outcome is BacktestRequestOutcome.CAPITAL_EXHAUSTED:
            if self.allocation_decision is not None:
                raise ValueError("capital-exhausted requests cannot have allocation decisions")
        elif self.allocation_decision is None:
            raise ValueError("allocated request results require an allocation decision")
        elif (
            self.request.candidate.identity
            != self.allocation_decision.candidate.identity
        ):
            raise ValueError("request candidate must match the allocation decision")
        elif self.outcome in {
            BacktestRequestOutcome.COMPLETED_ACTUAL,
            BacktestRequestOutcome.ALLOCATED_ENTRY_NOT_FILLED,
        } and self.allocation_decision.requires_shadow_tracking:
            raise ValueError("actual outcomes require an allocated decision")
        elif self.outcome in {
            BacktestRequestOutcome.COMPLETED_SHADOW,
            BacktestRequestOutcome.SHADOW_ENTRY_NOT_FILLED,
        } and not self.allocation_decision.requires_shadow_tracking:
            raise ValueError("shadow outcomes require a capacity-rejected decision")
        if self.trade_record is not None:
            if (
                self.trade_record.allocation_identity
                != self.request.candidate.identity
            ):
                raise ValueError("trade record identity must match the request candidate")
            if self.trade_record.allocation_decision != self.allocation_decision:
                raise ValueError("trade record must retain the terminal allocation decision")
            if self.terminal_at != self.trade_record.trade.exit_fill.timestamp:
                raise ValueError("completed terminal_at must equal the exit fill timestamp")
        return self


class BacktestRunResult(FrozenBacktestModel):
    """Deterministic completed-run output and reproducibility context."""

    run_id: NonEmptyStr
    git_commit: NonEmptyStr
    backtester_version: NonEmptyStr
    window_start: datetime
    window_end: datetime
    cost_policy_id: NonEmptyStr
    cost_policy_source_as_of_date: date
    brokerage_plan: BrokeragePlan
    starting_capital: Decimal
    ending_capital: Decimal
    capital_exhausted: bool
    actual_trade_records: tuple[BacktestTradeRecord, ...]
    shadow_trade_records: tuple[BacktestTradeRecord, ...]
    request_results: tuple[BacktestRequestResult, ...]
    ending_portfolio_state: PortfolioState | None
    symbols: tuple[str, ...]
    strategy_versions: tuple[tuple[str, str], ...]
    ml_model_versions: tuple[str, ...]
