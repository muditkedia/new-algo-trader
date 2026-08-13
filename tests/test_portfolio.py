from datetime import datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from algo_trader import MLScore, OrderIntent, OrderType, Side, Signal, SignalStatus
from algo_trader.portfolio import (
    AllocationCandidate,
    AllocationOutcome,
    CapitalAllocator,
    CapitalReservation,
    MarginRequirementProvider,
    MarginRequirementQuote,
    PortfolioState,
)

MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")
NOW = datetime(2025, 1, 2, 9, 20, tzinfo=MARKET_TIMEZONE)


def make_candidate(
    *,
    strategy_id: str = "strategy",
    strategy_version: str = "1",
    symbol: str = "TEST",
    side: Side = Side.LONG,
    signal_timestamp: datetime = NOW,
    order_timestamp: datetime = NOW,
    order_type: OrderType = OrderType.MARKET,
    limit_price: Decimal | None = None,
    quantity: int = 10,
    requested_notional: int = 50_000,
    recommended_notional: int | None = None,
    quality_score: float = 0.5,
    model_version: str = "meta-1",
    calibrated_probability: float = 0.5,
    predicted_net_return: float = 0.001,
) -> AllocationCandidate:
    signal = Signal(
        strategy_id=strategy_id,
        strategy_version=strategy_version,
        symbol=symbol,
        timestamp=signal_timestamp,
        side=side,
        strategy_parameters={"period": 5},
        feature_snapshot={"feature": 1.0},
    )
    order = OrderIntent(
        signal=signal,
        timestamp=order_timestamp,
        quantity=quantity,
        requested_notional=requested_notional,
        order_type=order_type,
        limit_price=limit_price,
    )
    score = MLScore(
        model_version=model_version,
        quality_score=quality_score,
        calibrated_probability=calibrated_probability,
        predicted_net_return=predicted_net_return,
        recommended_notional=(
            requested_notional
            if recommended_notional is None
            else recommended_notional
        ),
    )
    return AllocationCandidate(order_intent=order, ml_score=score)


class StubMarginProvider:
    def __init__(
        self,
        requirements: dict[str, Decimal],
        provider_id: str = "fixture-margin",
    ) -> None:
        self.requirements = requirements
        self.provider_id = provider_id
        self.calls: list[tuple[str, Decimal]] = []

    def quote(
        self,
        candidate: AllocationCandidate,
        state: PortfolioState,
    ) -> MarginRequirementQuote:
        symbol = candidate.order_intent.signal.symbol
        self.calls.append((symbol, state.available_margin))
        return MarginRequirementQuote(
            provider_id=self.provider_id,
            required_margin=self.requirements[symbol],
        )


class InvalidQuoteProvider:
    def quote(self, candidate: AllocationCandidate, state: PortfolioState) -> Decimal:
        return Decimal("1")


def make_reservation(
    candidate: AllocationCandidate,
    required_margin: str,
    provider_id: str = "existing-provider",
) -> CapitalReservation:
    return CapitalReservation(
        candidate=candidate,
        margin_quote=MarginRequirementQuote(
            provider_id=provider_id,
            required_margin=Decimal(required_margin),
        ),
    )


@pytest.mark.parametrize("notional", [50_000, 100_000])
def test_candidate_accepts_exact_domain_notional_bounds(notional: int) -> None:
    candidate = make_candidate(
        requested_notional=notional,
        recommended_notional=notional,
    )

    assert candidate.target_notional == notional


def test_candidate_rejects_requested_and_recommended_notional_mismatch() -> None:
    with pytest.raises(ValidationError, match="requested_notional must equal"):
        make_candidate(requested_notional=50_000, recommended_notional=55_000)


def test_reconstructed_candidates_have_the_same_stable_identity() -> None:
    first = make_candidate()
    reconstructed = make_candidate()

    assert first is not reconstructed
    assert first.identity == reconstructed.identity
    assert hash(first.identity) == hash(reconstructed.identity)


@pytest.mark.parametrize(
    "changes",
    [
        {"strategy_id": "other-strategy"},
        {"strategy_version": "2"},
        {"symbol": "OTHER"},
        {"side": Side.SHORT},
        {"signal_timestamp": NOW - timedelta(minutes=5)},
        {"order_timestamp": NOW + timedelta(minutes=5)},
        {
            "order_type": OrderType.LIMIT,
            "limit_price": Decimal("100"),
        },
        {"quantity": 11},
        {
            "requested_notional": 55_000,
            "recommended_notional": 55_000,
        },
    ],
)
def test_materially_different_order_identity_fields_are_distinct(changes: dict) -> None:
    assert make_candidate().identity != make_candidate(**changes).identity


def test_default_portfolio_capital_and_derived_capacity_are_exact_decimal() -> None:
    state = PortfolioState()

    assert state.capital_limit == Decimal("100000")
    assert state.reserved_margin == Decimal("0")
    assert state.available_margin == Decimal("100000")


def test_state_derives_reserved_and_zero_available_margin() -> None:
    reservation = make_reservation(make_candidate(), "100000")
    state = PortfolioState(active_reservations=(reservation,))

    assert state.reserved_margin == Decimal("100000")
    assert state.available_margin == Decimal("0")


def test_overcommitted_state_represents_negative_available_margin() -> None:
    reservation = make_reservation(make_candidate(), "110000")
    state = PortfolioState(active_reservations=(reservation,))

    assert state.available_margin == Decimal("-10000")


@pytest.mark.parametrize(
    "capital_limit",
    [0, Decimal("0"), Decimal("-1"), Decimal("NaN"), Decimal("Infinity")],
)
def test_invalid_portfolio_capital_limit_is_rejected(capital_limit) -> None:
    with pytest.raises(ValidationError):
        PortfolioState(capital_limit=capital_limit)


def test_valid_margin_quote_and_provider_identity_are_preserved() -> None:
    quote = MarginRequirementQuote(
        provider_id="historical-margin-v1",
        required_margin=Decimal("25000.50"),
    )

    assert quote.provider_id == "historical-margin-v1"
    assert quote.required_margin == Decimal("25000.50")


@pytest.mark.parametrize(
    "required_margin",
    [1, Decimal("0"), Decimal("-1"), Decimal("NaN"), Decimal("Infinity")],
)
def test_invalid_margin_requirement_is_rejected(required_margin) -> None:
    with pytest.raises(ValidationError):
        MarginRequirementQuote(
            provider_id="provider",
            required_margin=required_margin,
        )


def test_empty_margin_provider_identity_is_rejected() -> None:
    with pytest.raises(ValidationError):
        MarginRequirementQuote(provider_id="  ", required_margin=Decimal("1"))


def test_margin_provider_protocol_is_runtime_checkable() -> None:
    provider = StubMarginProvider({"TEST": Decimal("100")})

    assert isinstance(provider, MarginRequirementProvider)


def test_allocator_rejects_provider_result_that_is_not_a_margin_quote() -> None:
    with pytest.raises(TypeError, match="MarginRequirementQuote"):
        CapitalAllocator().allocate_batch(
            [make_candidate()],
            PortfolioState(),
            InvalidQuoteProvider(),  # type: ignore[arg-type]
        )


def test_low_quality_candidate_is_allocated_without_ml_rejection_or_resizing() -> None:
    candidate = make_candidate(quality_score=0.0, requested_notional=100_000)
    provider = StubMarginProvider({"TEST": Decimal("75000")})

    result = CapitalAllocator().allocate_batch([candidate], PortfolioState(), provider)

    decision = result.decisions[0]
    assert list(AllocationOutcome) == [
        AllocationOutcome.ALLOCATED,
        AllocationOutcome.CAPACITY_REJECTED,
    ]
    assert decision.outcome is AllocationOutcome.ALLOCATED
    assert decision.signal.status is SignalStatus.GENERATED
    assert decision.candidate.target_notional == 100_000
    assert decision.reservation is not None
    assert decision.reservation.target_notional == 100_000
    assert not decision.requires_shadow_tracking


def test_higher_quality_candidates_consume_capacity_first() -> None:
    high = make_candidate(symbol="HIGH", quality_score=0.9)
    medium = make_candidate(symbol="MEDIUM", quality_score=0.6)
    low = make_candidate(symbol="LOW", quality_score=0.2)
    provider = StubMarginProvider(
        {
            "HIGH": Decimal("60000"),
            "MEDIUM": Decimal("40000"),
            "LOW": Decimal("30000"),
        }
    )

    result = CapitalAllocator().allocate_batch(
        [low, high, medium],
        PortfolioState(),
        provider,
    )

    assert [decision.candidate.order_intent.signal.symbol for decision in result.decisions] == [
        "HIGH",
        "MEDIUM",
        "LOW",
    ]
    assert [decision.outcome for decision in result.decisions] == [
        AllocationOutcome.ALLOCATED,
        AllocationOutcome.ALLOCATED,
        AllocationOutcome.CAPACITY_REJECTED,
    ]
    assert result.ending_state.reserved_margin == Decimal("100000")
    assert len(result.ending_state.active_reservations) == 2
    assert provider.calls == [
        ("HIGH", Decimal("100000")),
        ("MEDIUM", Decimal("40000")),
        ("LOW", Decimal("0")),
    ]

    rejected = result.decisions[-1]
    assert rejected.reservation is None
    assert rejected.signal.status is SignalStatus.CAPACITY_REJECTED
    assert rejected.requires_shadow_tracking
    assert rejected.candidate.ml_score is low.ml_score
    assert rejected.candidate.target_notional == low.target_notional
    assert rejected.candidate.order_intent.signal.status is SignalStatus.GENERATED


def test_too_large_high_priority_candidate_does_not_block_smaller_candidate() -> None:
    high = make_candidate(symbol="HIGH", quality_score=0.9)
    low = make_candidate(symbol="LOW", quality_score=0.1)
    provider = StubMarginProvider(
        {"HIGH": Decimal("110000"), "LOW": Decimal("50000")}
    )

    result = CapitalAllocator().allocate_batch([low, high], PortfolioState(), provider)

    assert [decision.outcome for decision in result.decisions] == [
        AllocationOutcome.CAPACITY_REJECTED,
        AllocationOutcome.ALLOCATED,
    ]
    assert result.ending_state.reserved_margin == Decimal("50000")


def test_equal_quality_tie_order_is_deterministic_independent_of_input_order() -> None:
    alpha = make_candidate(symbol="ALPHA", quality_score=0.5)
    beta = make_candidate(symbol="BETA", quality_score=0.5)
    provider_one = StubMarginProvider({"ALPHA": Decimal("100"), "BETA": Decimal("100")})
    provider_two = StubMarginProvider({"ALPHA": Decimal("100"), "BETA": Decimal("100")})
    allocator = CapitalAllocator()

    first = allocator.allocate_batch([beta, alpha], PortfolioState(), provider_one)
    second = allocator.allocate_batch([alpha, beta], PortfolioState(), provider_two)

    first_order = [decision.candidate.identity for decision in first.decisions]
    second_order = [decision.candidate.identity for decision in second.decisions]
    assert first_order == second_order == [alpha.identity, beta.identity]


@pytest.mark.parametrize(
    ("first_changes", "second_changes"),
    [
        pytest.param(
            {"signal_timestamp": NOW - timedelta(minutes=10)},
            {"signal_timestamp": NOW - timedelta(minutes=5)},
            id="signal-timestamp",
        ),
        pytest.param(
            {"order_type": OrderType.MARKET},
            {"order_type": OrderType.LIMIT, "limit_price": Decimal("100")},
            id="order-type-and-limit-price",
        ),
        pytest.param(
            {"quantity": 10},
            {"quantity": 11},
            id="quantity",
        ),
        pytest.param(
            {"requested_notional": 50_000},
            {"requested_notional": 55_000},
            id="requested-notional",
        ),
    ],
)
def test_complete_identity_tie_break_is_input_order_independent_under_capacity_pressure(
    first_changes: dict,
    second_changes: dict,
) -> None:
    first_candidate = make_candidate(**first_changes)
    second_candidate = make_candidate(**second_changes)
    allocator = CapitalAllocator()

    forward = allocator.allocate_batch(
        [first_candidate, second_candidate],
        PortfolioState(),
        StubMarginProvider({"TEST": Decimal("60000")}),
    )
    reversed_result = allocator.allocate_batch(
        [second_candidate, first_candidate],
        PortfolioState(),
        StubMarginProvider({"TEST": Decimal("60000")}),
    )

    forward_decisions = [
        (decision.candidate.identity, decision.outcome)
        for decision in forward.decisions
    ]
    reversed_decisions = [
        (decision.candidate.identity, decision.outcome)
        for decision in reversed_result.decisions
    ]
    assert first_candidate.identity != second_candidate.identity
    assert forward_decisions == reversed_decisions
    assert [outcome for _, outcome in forward_decisions] == [
        AllocationOutcome.ALLOCATED,
        AllocationOutcome.CAPACITY_REJECTED,
    ]


def test_secondary_ml_metrics_do_not_influence_equal_quality_tie_breaking() -> None:
    lower_identity = make_candidate(
        quantity=10,
        model_version="z-model",
        calibrated_probability=1.0,
        predicted_net_return=1.0,
    )
    higher_identity = make_candidate(
        quantity=11,
        model_version="a-model",
        calibrated_probability=0.0,
        predicted_net_return=-1.0,
    )

    result = CapitalAllocator().allocate_batch(
        [higher_identity, lower_identity],
        PortfolioState(),
        StubMarginProvider({"TEST": Decimal("100")}),
    )

    assert [decision.candidate.identity for decision in result.decisions] == [
        lower_identity.identity,
        higher_identity.identity,
    ]


def test_mixed_allocation_timestamps_are_rejected_as_lookahead_risk() -> None:
    current = make_candidate(symbol="CURRENT")
    future = make_candidate(
        symbol="FUTURE",
        signal_timestamp=NOW + timedelta(minutes=5),
        order_timestamp=NOW + timedelta(minutes=5),
        quality_score=1.0,
    )
    provider = StubMarginProvider(
        {"CURRENT": Decimal("100"), "FUTURE": Decimal("100")}
    )

    with pytest.raises(ValueError, match="same timestamp"):
        CapitalAllocator().allocate_batch([current, future], PortfolioState(), provider)


def test_identical_duplicate_candidates_are_rejected() -> None:
    first = make_candidate()
    reconstructed = make_candidate()
    provider = StubMarginProvider({"TEST": Decimal("100")})

    with pytest.raises(ValueError, match="duplicate candidate identity"):
        CapitalAllocator().allocate_batch(
            [first, reconstructed],
            PortfolioState(),
            provider,
        )


def test_duplicate_identity_with_conflicting_ml_score_is_rejected() -> None:
    first = make_candidate(quality_score=0.9, model_version="meta-1")
    conflicting = make_candidate(quality_score=0.1, model_version="meta-2")
    provider = StubMarginProvider({"TEST": Decimal("100")})

    with pytest.raises(ValueError, match="conflicting MLScore"):
        CapitalAllocator().allocate_batch(
            [first, conflicting],
            PortfolioState(),
            provider,
        )


def test_existing_reservation_reduces_capacity_and_is_preserved() -> None:
    existing = make_reservation(make_candidate(symbol="EXISTING"), "60000")
    state = PortfolioState(active_reservations=(existing,))
    candidate = make_candidate(symbol="NEW")
    provider = StubMarginProvider({"NEW": Decimal("50000")})

    result = CapitalAllocator().allocate_batch([candidate], state, provider)

    assert result.decisions[0].outcome is AllocationOutcome.CAPACITY_REJECTED
    assert result.ending_state.active_reservations == (existing,)
    assert result.ending_state.reserved_margin == Decimal("60000")


def test_candidate_already_actively_reserved_is_rejected_by_identity() -> None:
    reserved_candidate = make_candidate()
    existing = make_reservation(reserved_candidate, "100")
    reconstructed = make_candidate()
    provider = StubMarginProvider({"TEST": Decimal("100")})

    with pytest.raises(ValueError, match="already represented"):
        CapitalAllocator().allocate_batch(
            [reconstructed],
            PortfolioState(active_reservations=(existing,)),
            provider,
        )


def test_provider_identity_is_retained_in_new_reservation() -> None:
    candidate = make_candidate()
    provider = StubMarginProvider(
        {"TEST": Decimal("25000")},
        provider_id="angel-margin-snapshot",
    )

    result = CapitalAllocator().allocate_batch([candidate], PortfolioState(), provider)

    reservation = result.ending_state.active_reservations[0]
    assert reservation.margin_provider_id == "angel-margin-snapshot"
    assert reservation.required_margin == Decimal("25000")


def test_overcommitted_state_admits_no_new_allocation() -> None:
    existing = make_reservation(make_candidate(symbol="EXISTING"), "110000")
    candidate = make_candidate(symbol="NEW")
    provider = StubMarginProvider({"NEW": Decimal("1")})

    result = CapitalAllocator().allocate_batch(
        [candidate],
        PortfolioState(active_reservations=(existing,)),
        provider,
    )

    assert result.decisions[0].outcome is AllocationOutcome.CAPACITY_REJECTED
    assert result.ending_state.available_margin == Decimal("-10000")


def test_release_restores_exact_capacity_and_preserves_other_reservations() -> None:
    first = make_reservation(make_candidate(symbol="FIRST"), "30000")
    second = make_reservation(make_candidate(symbol="SECOND"), "20000")
    state = PortfolioState(active_reservations=(first, second))

    released = CapitalAllocator().release(state, first)

    assert released.active_reservations == (second,)
    assert released.reserved_margin == Decimal("20000")
    assert released.available_margin == Decimal("80000")
    assert state.active_reservations == (first, second)


def test_release_uses_reconstructed_reservation_identity() -> None:
    stored = make_reservation(make_candidate(), "30000")
    reconstructed = make_reservation(make_candidate(), "99999", "other-provider")
    state = PortfolioState(active_reservations=(stored,))

    released = CapitalAllocator().release(state, reconstructed)

    assert released.available_margin == Decimal("100000")


def test_releasing_same_reservation_twice_fails() -> None:
    reservation = make_reservation(make_candidate(), "30000")
    allocator = CapitalAllocator()
    released = allocator.release(
        PortfolioState(active_reservations=(reservation,)),
        reservation,
    )

    with pytest.raises(ValueError, match="not active"):
        allocator.release(released, reservation)


def test_releasing_unknown_reservation_fails() -> None:
    known = make_reservation(make_candidate(symbol="KNOWN"), "30000")
    unknown = make_reservation(make_candidate(symbol="UNKNOWN"), "10000")

    with pytest.raises(ValueError, match="not active"):
        CapitalAllocator().release(
            PortfolioState(active_reservations=(known,)),
            unknown,
        )


def test_allocation_preserves_sources_and_ending_state_matches_decisions() -> None:
    existing = make_reservation(make_candidate(symbol="EXISTING"), "10000")
    state = PortfolioState(active_reservations=(existing,))
    first = make_candidate(symbol="FIRST", quality_score=0.8)
    second = make_candidate(symbol="SECOND", quality_score=0.7)
    candidates_before = [first.model_dump(), second.model_dump()]
    state_before = state.model_dump()
    provider = StubMarginProvider(
        {"FIRST": Decimal("40000"), "SECOND": Decimal("50000")}
    )

    result = CapitalAllocator().allocate_batch([second, first], state, provider)

    allocated = tuple(
        decision.reservation
        for decision in result.decisions
        if decision.reservation is not None
    )
    assert result.ending_state.active_reservations == (existing, *allocated)
    assert result.ending_state.reserved_margin == Decimal("100000")
    assert result.ending_state.available_margin == Decimal("0")
    assert [first.model_dump(), second.model_dump()] == candidates_before
    assert state.model_dump() == state_before
