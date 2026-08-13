"""Immutable portfolio allocation candidates, reservations, and outcomes."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from algo_trader.domain import MLScore, OrderIntent, OrderType, Side, Signal, SignalStatus

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
StrictPositiveDecimal = Annotated[
    Decimal,
    Field(strict=True, gt=0, allow_inf_nan=False),
]

type CandidateIdentity = tuple[
    str,
    str,
    str,
    Side,
    datetime,
    datetime,
    OrderType,
    Decimal | None,
    int,
    int,
]


class FrozenPortfolioModel(BaseModel):
    """Validation policy for immutable portfolio-owned records."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class AllocationCandidate(FrozenPortfolioModel):
    """A generated order and its pre-outcome ML sizing recommendation."""

    order_intent: OrderIntent
    ml_score: MLScore

    @model_validator(mode="after")
    def validate_notional_invariant(self) -> AllocationCandidate:
        if self.order_intent.requested_notional != self.ml_score.recommended_notional:
            raise ValueError("requested_notional must equal ML-recommended notional")
        return self

    @property
    def target_notional(self) -> int:
        """Return the exact ML-requested notional bucket without resizing."""
        return self.ml_score.recommended_notional

    @property
    def identity(self) -> CandidateIdentity:
        """Return the stable signal/order identity used throughout portfolio state."""
        signal = self.order_intent.signal
        return (
            signal.strategy_id,
            signal.strategy_version,
            signal.symbol,
            signal.side,
            signal.timestamp,
            self.order_intent.timestamp,
            self.order_intent.order_type,
            self.order_intent.limit_price,
            self.order_intent.quantity,
            self.order_intent.requested_notional,
        )


class MarginRequirementQuote(FrozenPortfolioModel):
    """Explicit capital required by one candidate according to one provider."""

    provider_id: NonEmptyStr
    required_margin: StrictPositiveDecimal


class CapitalReservation(FrozenPortfolioModel):
    """Auditable capacity reserved for one allocated candidate."""

    candidate: AllocationCandidate
    margin_quote: MarginRequirementQuote

    @property
    def identity(self) -> CandidateIdentity:
        """Return the candidate identity used for lookup and release."""
        return self.candidate.identity

    @property
    def target_notional(self) -> int:
        """Return the unchanged ML target notional retained by this reservation."""
        return self.candidate.target_notional

    @property
    def required_margin(self) -> Decimal:
        """Return the exact reserved capital amount."""
        return self.margin_quote.required_margin

    @property
    def margin_provider_id(self) -> str:
        """Return the provider that supplied the reservation requirement."""
        return self.margin_quote.provider_id


class PortfolioState(FrozenPortfolioModel):
    """Capital limit and immutable active reservations."""

    capital_limit: StrictPositiveDecimal = Decimal("100000")
    active_reservations: tuple[CapitalReservation, ...] = ()

    @model_validator(mode="after")
    def reject_duplicate_reservations(self) -> PortfolioState:
        identities = [reservation.identity for reservation in self.active_reservations]
        if len(identities) != len(set(identities)):
            raise ValueError("active reservations must have unique candidate identities")
        return self

    @property
    def reserved_margin(self) -> Decimal:
        """Return the exact total required margin of active reservations."""
        return sum(
            (reservation.required_margin for reservation in self.active_reservations),
            start=Decimal("0"),
        )

    @property
    def available_margin(self) -> Decimal:
        """Return exact capacity, including negative overcommitment if present."""
        return self.capital_limit - self.reserved_margin


class AllocationOutcome(StrEnum):
    """Portfolio-level allocation outcomes; ML never rejects a valid signal."""

    ALLOCATED = "ALLOCATED"
    CAPACITY_REJECTED = "CAPACITY_REJECTED"


class AllocationDecision(FrozenPortfolioModel):
    """Auditable allocation result for exactly one candidate."""

    candidate: AllocationCandidate
    outcome: AllocationOutcome
    margin_quote: MarginRequirementQuote
    signal: Signal
    reservation: CapitalReservation | None = None

    @model_validator(mode="after")
    def validate_outcome_contract(self) -> AllocationDecision:
        original_signal = self.candidate.order_intent.signal
        if self.outcome is AllocationOutcome.ALLOCATED:
            if self.reservation is None:
                raise ValueError("an allocated decision requires a reservation")
            if (
                self.reservation.identity != self.candidate.identity
                or self.reservation.candidate.ml_score != self.candidate.ml_score
                or self.reservation.margin_quote != self.margin_quote
            ):
                raise ValueError("reservation must retain the decision candidate and quote")
            if self.signal != original_signal or self.signal.status is not SignalStatus.GENERATED:
                raise ValueError("an allocated signal must remain GENERATED")
        else:
            if self.reservation is not None:
                raise ValueError("a capacity-rejected decision cannot reserve margin")
            expected_signal = original_signal.model_copy(
                update={"status": SignalStatus.CAPACITY_REJECTED}
            )
            if self.signal != expected_signal:
                raise ValueError(
                    "a capacity-rejected decision requires the rejected signal copy"
                )
        return self

    @property
    def requires_shadow_tracking(self) -> bool:
        """Indicate whether later orchestration must simulate a shadow outcome."""
        return self.outcome is AllocationOutcome.CAPACITY_REJECTED


class AllocationBatchResult(FrozenPortfolioModel):
    """Priority-ordered decisions and the resulting immutable portfolio state."""

    decisions: tuple[AllocationDecision, ...]
    ending_state: PortfolioState
