"""Deterministic central allocation of shared portfolio capacity."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from decimal import Decimal

from algo_trader.domain import SignalStatus
from algo_trader.portfolio.margin import MarginRequirementProvider
from algo_trader.portfolio.models import (
    AllocationBatchResult,
    AllocationCandidate,
    AllocationDecision,
    AllocationOutcome,
    CandidateIdentity,
    CapitalReservation,
    MarginRequirementQuote,
    PortfolioState,
)

type CandidateIdentitySortKey = tuple[
    str,
    str,
    str,
    str,
    datetime,
    datetime,
    str,
    int,
    Decimal,
    int,
    int,
]
type CandidatePriorityKey = tuple[
    float,
    str,
    str,
    str,
    str,
    datetime,
    datetime,
    str,
    int,
    Decimal,
    int,
    int,
]


class CapitalAllocator:
    """Allocate full candidates by ML quality while preserving shared capacity."""

    def allocate_batch(
        self,
        candidates: Iterable[AllocationCandidate],
        state: PortfolioState,
        margin_provider: MarginRequirementProvider,
    ) -> AllocationBatchResult:
        """Allocate one simultaneous-timestamp batch in deterministic priority order."""
        if not isinstance(state, PortfolioState):
            raise TypeError("state must be a PortfolioState")
        if not isinstance(margin_provider, MarginRequirementProvider):
            raise TypeError("margin_provider must implement MarginRequirementProvider")

        selected = tuple(candidates)
        if not selected:
            raise ValueError("at least one allocation candidate is required")
        if any(not isinstance(candidate, AllocationCandidate) for candidate in selected):
            raise TypeError("all candidates must be AllocationCandidate instances")

        allocation_timestamp = selected[0].order_intent.timestamp
        if any(
            candidate.order_intent.timestamp != allocation_timestamp
            for candidate in selected[1:]
        ):
            raise ValueError("all allocation candidates must have the same timestamp")

        self._validate_unique_batch(selected)
        active_identities = {reservation.identity for reservation in state.active_reservations}
        already_reserved = [
            candidate.identity
            for candidate in selected
            if candidate.identity in active_identities
        ]
        if already_reserved:
            raise ValueError("candidate is already represented by an active reservation")

        current_state = state
        decisions: list[AllocationDecision] = []
        for candidate in sorted(selected, key=_priority_key):
            quote = margin_provider.quote(candidate, current_state)
            if not isinstance(quote, MarginRequirementQuote):
                raise TypeError("margin provider must return a MarginRequirementQuote")

            if quote.required_margin <= current_state.available_margin:
                reservation = CapitalReservation(candidate=candidate, margin_quote=quote)
                current_state = PortfolioState(
                    capital_limit=current_state.capital_limit,
                    active_reservations=current_state.active_reservations + (reservation,),
                )
                decisions.append(
                    AllocationDecision(
                        candidate=candidate,
                        outcome=AllocationOutcome.ALLOCATED,
                        margin_quote=quote,
                        signal=candidate.order_intent.signal,
                        reservation=reservation,
                    )
                )
            else:
                rejected_signal = candidate.order_intent.signal.model_copy(
                    update={"status": SignalStatus.CAPACITY_REJECTED}
                )
                decisions.append(
                    AllocationDecision(
                        candidate=candidate,
                        outcome=AllocationOutcome.CAPACITY_REJECTED,
                        margin_quote=quote,
                        signal=rejected_signal,
                    )
                )

        return AllocationBatchResult(decisions=tuple(decisions), ending_state=current_state)

    def release(
        self,
        state: PortfolioState,
        reservation: CapitalReservation,
    ) -> PortfolioState:
        """Release exactly one stored reservation identified by stable order identity."""
        if not isinstance(state, PortfolioState):
            raise TypeError("state must be a PortfolioState")
        if not isinstance(reservation, CapitalReservation):
            raise TypeError("reservation must be a CapitalReservation")

        matching_indices = [
            index
            for index, active in enumerate(state.active_reservations)
            if active.identity == reservation.identity
        ]
        if not matching_indices:
            raise ValueError("reservation is not active in the portfolio state")

        release_index = matching_indices[0]
        remaining = (
            state.active_reservations[:release_index]
            + state.active_reservations[release_index + 1 :]
        )
        return PortfolioState(
            capital_limit=state.capital_limit,
            active_reservations=remaining,
        )

    @staticmethod
    def _validate_unique_batch(candidates: tuple[AllocationCandidate, ...]) -> None:
        seen: dict[CandidateIdentity, AllocationCandidate] = {}
        for candidate in candidates:
            existing = seen.get(candidate.identity)
            if existing is None:
                seen[candidate.identity] = candidate
                continue
            if existing.ml_score != candidate.ml_score:
                raise ValueError(
                    "duplicate candidate identity has conflicting MLScore information"
                )
            raise ValueError("duplicate candidate identity in allocation batch")


def _identity_sort_key(identity: CandidateIdentity) -> CandidateIdentitySortKey:
    (
        strategy_id,
        strategy_version,
        symbol,
        side,
        signal_timestamp,
        order_timestamp,
        order_type,
        limit_price,
        quantity,
        requested_notional,
    ) = identity
    return (
        strategy_id,
        strategy_version,
        symbol,
        side.value,
        signal_timestamp,
        order_timestamp,
        order_type.value,
        int(limit_price is not None),
        limit_price if limit_price is not None else Decimal("0"),
        quantity,
        requested_notional,
    )


def _priority_key(candidate: AllocationCandidate) -> CandidatePriorityKey:
    return (
        -candidate.ml_score.quality_score,
        *_identity_sort_key(candidate.identity),
    )
