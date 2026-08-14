"""Immutable public contracts for deterministic intraday Runtime orchestration."""

from __future__ import annotations

from datetime import date, datetime, time
from decimal import Decimal
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from algo_trader.broker import (
    BrokerOrderAcknowledgement,
    BrokerOrderSnapshot,
    BrokerTradeFill,
    BrokerTransactionAction,
)
from algo_trader.costs import BrokeragePlan
from algo_trader.domain import ExitReason, Fill, OrderType, ProtectiveExitSpec, Trade
from algo_trader.portfolio import (
    AllocationCandidate,
    AllocationDecision,
    CandidateIdentity,
    CapitalReservation,
    MarginRequirementQuote,
)

RUNTIME_ARCHITECTURE_VERSION = "1"
MARKET_TIMEZONE_NAME = "Asia/Kolkata"
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveDecimal = Annotated[Decimal, Field(gt=0, allow_inf_nan=False)]
FiniteDecimal = Annotated[Decimal, Field(allow_inf_nan=False)]


class FrozenRuntimeModel(BaseModel):
    """Shared validation policy for persisted and public Runtime records."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class RuntimeMode(StrEnum):
    """Only live-session execution modes; historical backtesting is separate."""

    PAPER = "PAPER"
    LIVE = "LIVE"


class RuntimePhase(StrEnum):
    """Explicit auditable session-safety lifecycle."""

    CREATED = "CREATED"
    PREOPEN = "PREOPEN"
    READY = "READY"
    TRADING = "TRADING"
    ENTRY_CLOSED = "ENTRY_CLOSED"
    SQUARE_OFF = "SQUARE_OFF"
    HALTED = "HALTED"
    STOPPED = "STOPPED"


class RuntimeOrderLeg(StrEnum):
    """A deterministic order identity distinguishes position entry and exit."""

    ENTRY = "ENTRY"
    EXIT = "EXIT"


class RuntimeOrderLifecycle(StrEnum):
    """Durable lifecycle for a side-effectful live broker request."""

    INTENT_RECORDED = "INTENT_RECORDED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    SUBMISSION_AMBIGUOUS = "SUBMISSION_AMBIGUOUS"
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"


class RuntimeExitLifecycle(StrEnum):
    """Duplication guard for actual and simulated position exits."""

    NONE = "NONE"
    REQUESTED = "REQUESTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    AMBIGUOUS = "AMBIGUOUS"


class RuntimeSessionTimes(FrozenRuntimeModel):
    """Configurable Asia/Kolkata intraday lifecycle defaults."""

    startup_time: time = time(8, 45)
    market_open_time: time = time(9, 15)
    entry_cutoff_time: time = time(15, 10)
    square_off_time: time = time(15, 20)
    market_close_time: time = time(15, 30)
    shutdown_time: time = time(15, 35)

    @model_validator(mode="after")
    def validate_order(self) -> RuntimeSessionTimes:
        values = (
            self.startup_time,
            self.market_open_time,
            self.entry_cutoff_time,
            self.square_off_time,
            self.market_close_time,
            self.shutdown_time,
        )
        if any(value.tzinfo is not None for value in values):
            raise ValueError("session times must be timezone-naive Asia/Kolkata wall times")
        if any(left >= right for left, right in pairwise(values)):
            raise ValueError("runtime session times must be strictly increasing")
        return self


class RuntimeConfig(FrozenRuntimeModel):
    """Non-secret deterministic operational configuration."""

    mode: RuntimeMode
    session_times: RuntimeSessionTimes = Field(default_factory=RuntimeSessionTimes)
    credential_path: Path = Path(".secrets/SmartAPI.env")
    state_db_path: Path
    brokerage_plan: BrokeragePlan
    starting_capital: PositiveDecimal
    live_order_submission_enabled: bool = False
    market_timezone: Literal["Asia/Kolkata"] = MARKET_TIMEZONE_NAME
    scheduler_misfire_grace_seconds: int = Field(default=60, strict=True, gt=0)
    stale_market_data_seconds: int = Field(default=30, strict=True, gt=0)


class RuntimeTradePlan(FrozenRuntimeModel):
    """Already-generated and already-scored opportunity supplied to Runtime."""

    candidate: AllocationCandidate
    protective_exit: ProtectiveExitSpec | None = None
    scrip_consent: bool = False


class RuntimeSessionRecord(FrozenRuntimeModel):
    """Persisted non-secret session provenance and economic-capital state."""

    runtime_session_id: NonEmptyStr
    trading_date: date
    mode: RuntimeMode
    runtime_version: Literal["1"] = RUNTIME_ARCHITECTURE_VERSION
    phase: RuntimePhase = RuntimePhase.CREATED
    starting_capital: PositiveDecimal
    current_capital: FiniteDecimal
    started_at: datetime
    ended_at: datetime | None = None
    live_order_submission_enabled: bool
    configuration_fingerprint: NonEmptyStr
    halt_reason: str | None = None

    @model_validator(mode="after")
    def validate_times(self) -> RuntimeSessionRecord:
        _require_aware(self.started_at, "started_at")
        if self.ended_at is not None:
            _require_aware(self.ended_at, "ended_at")
            if self.ended_at < self.started_at:
                raise ValueError("ended_at cannot precede started_at")
        return self


class RuntimeEvent(FrozenRuntimeModel):
    """Monotonically sequenced safe runtime event ledger entry."""

    runtime_session_id: NonEmptyStr
    sequence: int = Field(strict=True, gt=0)
    occurred_at: datetime
    event_type: NonEmptyStr
    description: str = ""

    @model_validator(mode="after")
    def validate_time(self) -> RuntimeEvent:
        _require_aware(self.occurred_at, "occurred_at")
        return self


class RuntimeOrderRecord(FrozenRuntimeModel):
    """Persist-before-side-effect live order identity and lifecycle evidence."""

    runtime_session_id: NonEmptyStr
    client_order_id: NonEmptyStr
    candidate_fingerprint: NonEmptyStr
    leg: RuntimeOrderLeg
    attempt: int = Field(default=1, strict=True, gt=0)
    symbol: NonEmptyStr
    quantity: int = Field(strict=True, gt=0)
    transaction_action: BrokerTransactionAction
    order_type: OrderType
    limit_price: Decimal | None = Field(default=None, gt=0, allow_inf_nan=False)
    intended_at: datetime
    lifecycle: RuntimeOrderLifecycle = RuntimeOrderLifecycle.INTENT_RECORDED
    broker_order_tag: str | None = None
    broker_order_id: str | None = None
    unique_order_id: str | None = None
    acknowledged_at: datetime | None = None
    cancellation_requested_at: datetime | None = None
    exit_reason: ExitReason | None = None

    @model_validator(mode="after")
    def validate_times(self) -> RuntimeOrderRecord:
        if self.leg is RuntimeOrderLeg.ENTRY and self.attempt != 1:
            raise ValueError("ENTRY order attempt must be 1")
        _require_aware(self.intended_at, "intended_at")
        if self.acknowledged_at is not None:
            _require_aware(self.acknowledged_at, "acknowledged_at")
        if self.cancellation_requested_at is not None:
            _require_aware(self.cancellation_requested_at, "cancellation_requested_at")
        return self


class RuntimeSubmissionRecord(FrozenRuntimeModel):
    """Return value for one acknowledged live order submission."""

    runtime_order: RuntimeOrderRecord
    acknowledgement: BrokerOrderAcknowledgement


class RuntimePositionRecord(FrozenRuntimeModel):
    """Persistable actual or shadow open-position state."""

    runtime_session_id: NonEmptyStr
    candidate: AllocationCandidate
    candidate_fingerprint: NonEmptyStr
    allocation_decision: AllocationDecision
    reservation: CapitalReservation | None = None
    entry_fill: Fill
    protective_exit: ProtectiveExitSpec | None = None
    is_shadow: bool = False
    mfe_return: Decimal = Field(default=Decimal("0"), ge=0, allow_inf_nan=False)
    mae_return: Decimal = Field(default=Decimal("0"), le=0, allow_inf_nan=False)
    exit_lifecycle: RuntimeExitLifecycle = RuntimeExitLifecycle.NONE
    requested_exit_reason: ExitReason | None = None
    requested_exit_at: datetime | None = None
    broker_entry_client_order_id: str | None = None
    broker_exit_client_order_ids: tuple[str, ...] = ()
    exit_filled_quantity: int = Field(default=0, strict=True, ge=0)

    @model_validator(mode="after")
    def validate_position(self) -> RuntimePositionRecord:
        if self.is_shadow and self.reservation is not None:
            raise ValueError("shadow positions cannot hold capital reservations")
        if not self.is_shadow and self.reservation is None:
            raise ValueError("actual positions require their active reservation")
        if self.requested_exit_at is not None:
            _require_aware(self.requested_exit_at, "requested_exit_at")
        if self.exit_filled_quantity > self.entry_fill.quantity:
            raise ValueError("exit_filled_quantity cannot exceed entry fill quantity")
        if len(self.broker_exit_client_order_ids) != len(
            set(self.broker_exit_client_order_ids)
        ):
            raise ValueError("broker exit client order IDs must be unique")
        return self

    @property
    def identity(self) -> CandidateIdentity:
        return self.candidate.identity


class RuntimeTradeRecord(FrozenRuntimeModel):
    """Completed actual or shadow trade plus execution/allocation provenance."""

    runtime_session_id: NonEmptyStr
    candidate_fingerprint: NonEmptyStr
    allocation_decision: AllocationDecision
    margin_quote: MarginRequirementQuote
    trade: Trade
    cost_policy_id: NonEmptyStr
    broker_entry_client_order_id: str | None = None
    broker_exit_client_order_ids: tuple[str, ...] = ()
    broker_entry_fill_ids: tuple[str, ...] = ()
    broker_exit_fill_ids: tuple[str, ...] = ()


class LiveReconciliationResult(FrozenRuntimeModel):
    """Exact normalized evidence for one persisted live order."""

    runtime_order: RuntimeOrderRecord
    broker_order: BrokerOrderSnapshot
    broker_fills: tuple[BrokerTradeFill, ...]
    aggregate_fill: Fill | None = None


class RuntimeConnectivityReport(FrozenRuntimeModel):
    """Read-only SmartAPI health report with no credentials or tokens."""

    authenticated: bool
    client_code: NonEmptyStr
    sdk_version: NonEmptyStr
    funds_read_ok: bool
    positions_read_ok: bool
    orders_read_ok: bool
    quote_read_ok: bool | None = None
    checked_at: datetime

    @model_validator(mode="after")
    def validate_time(self) -> RuntimeConnectivityReport:
        _require_aware(self.checked_at, "checked_at")
        return self


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
