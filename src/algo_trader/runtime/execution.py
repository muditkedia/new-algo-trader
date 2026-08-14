"""PAPER simulation and LIVE broker lifecycle gateways for Runtime."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal

import polars as pl

from algo_trader.broker import (
    AngelOneBroker,
    AngelOneInstrumentMaster,
    BrokerAmbiguousStateError,
    BrokerApiError,
    BrokerMarketTick,
    BrokerOrderRequest,
    BrokerOrderState,
    BrokerTradeFill,
    BrokerTransactionAction,
    broker_order_tag,
    entry_action,
    exit_action,
)
from algo_trader.domain import (
    ExitReason,
    ExitReasonDetail,
    Fill,
    OrderType,
    ProtectiveExitSpec,
    Side,
)
from algo_trader.execution import ExitResult, HistoricalExecutionSimulator
from algo_trader.execution.dynamic_exit import (
    RMultipleTrailingCoreParameters,
    RMultipleTrailingState,
    initialize_r_multiple_state,
    r_multiple_stop_detail,
)
from algo_trader.portfolio import AllocationDecision, AllocationOutcome
from algo_trader.runtime.identity import candidate_fingerprint, runtime_client_order_id
from algo_trader.runtime.models import (
    LiveReconciliationResult,
    RuntimeDynamicExitState,
    RuntimeExitLifecycle,
    RuntimeOrderLeg,
    RuntimeOrderLifecycle,
    RuntimeOrderRecord,
    RuntimePositionRecord,
    RuntimeSubmissionRecord,
    RuntimeTradePlan,
)
from algo_trader.runtime.state import RuntimeStateStore


@dataclass(frozen=True, slots=True)
class PaperTickResult:
    """Position openings and closings caused by one observed market tick."""

    opened: tuple[RuntimePositionRecord, ...] = ()
    closed: tuple[tuple[RuntimePositionRecord, ExitResult], ...] = ()


@dataclass(frozen=True, slots=True)
class _PendingPaperEntry:
    plan: RuntimeTradePlan
    decision: AllocationDecision
    candidate_fingerprint: str
    is_shadow: bool


class PaperExecutionGateway:
    """Tick-driven PAPER and shadow execution through the frozen simulator."""

    def __init__(
        self,
        runtime_session_id: str,
        state_store: RuntimeStateStore,
        simulator: HistoricalExecutionSimulator | None = None,
    ) -> None:
        self.runtime_session_id = runtime_session_id
        self.state_store = state_store
        self.simulator = simulator or HistoricalExecutionSimulator()
        self._pending: dict[str, _PendingPaperEntry] = {}
        self._positions: dict[str, RuntimePositionRecord] = {}

    @property
    def pending_fingerprints(self) -> tuple[str, ...]:
        return tuple(sorted(self._pending))

    @property
    def positions(self) -> tuple[RuntimePositionRecord, ...]:
        return tuple(self._positions[key] for key in sorted(self._positions))

    def submit_allocated_entry(
        self,
        plan: RuntimeTradePlan,
        decision: AllocationDecision,
        intended_at: datetime,
    ) -> None:
        """Track an allocated entry locally; no broker order method exists here."""
        del intended_at
        if decision.outcome is not AllocationOutcome.ALLOCATED:
            raise ValueError("allocated PAPER entry requires an ALLOCATED decision")
        self._add_pending(plan, decision, is_shadow=False)

    def submit_shadow(self, plan: RuntimeTradePlan, decision: AllocationDecision) -> None:
        """Track an allocator capacity rejection with no reservation or broker action."""
        if decision.outcome is not AllocationOutcome.CAPACITY_REJECTED:
            raise ValueError("shadow tracking requires a CAPACITY_REJECTED decision")
        self._add_pending(plan, decision, is_shadow=True)

    def on_market_tick(self, tick: BrokerMarketTick) -> PaperTickResult:
        """Drive eligible point-bar entries and exits from one immutable broker tick."""
        point_bar = point_bar_from_tick(tick)
        opened: list[RuntimePositionRecord] = []
        closed: list[tuple[RuntimePositionRecord, ExitResult]] = []
        for fingerprint, pending in tuple(sorted(self._pending.items())):
            if pending.plan.candidate.order_intent.signal.symbol != tick.instrument.symbol:
                continue
            fill = self.simulator.fill_entry_order(
                pending.plan.candidate.order_intent, point_bar
            )
            if fill is None:
                continue
            position = RuntimePositionRecord(
                runtime_session_id=self.runtime_session_id,
                candidate=pending.plan.candidate,
                candidate_fingerprint=fingerprint,
                allocation_decision=pending.decision,
                reservation=None if pending.is_shadow else pending.decision.reservation,
                entry_fill=fill,
                protective_exit=pending.plan.protective_exit,
                dynamic_exit_policy=pending.plan.dynamic_exit_policy,
                dynamic_exit_state=initialize_runtime_dynamic_exit(
                    pending.plan, fill
                ),
                is_shadow=pending.is_shadow,
            )
            self._positions[fingerprint] = position
            del self._pending[fingerprint]
            self.state_store.open_position(position)
            opened.append(position)

        for fingerprint, position in tuple(sorted(self._positions.items())):
            if position.candidate.order_intent.signal.symbol != tick.instrument.symbol:
                continue
            updated = update_position_excursion(position, tick)
            result = self._paper_exit_result(updated, point_bar)
            if result is None:
                if updated != position:
                    self._positions[fingerprint] = updated
                    self.state_store.save_position(updated)
                continue
            terminal = updated.model_copy(
                update={"exit_lifecycle": RuntimeExitLifecycle.FILLED}
            )
            closed.append((terminal, result))
        return PaperTickResult(opened=tuple(opened), closed=tuple(closed))

    def publish_completed_position(self, fingerprint: str) -> None:
        """Remove a position from memory only after durable trade finalization."""
        self._positions.pop(fingerprint, None)

    def request_exit(
        self,
        fingerprint: str,
        requested_at: datetime,
        reason: ExitReason,
    ) -> bool:
        """Record exactly one generic exit request for an active filled position."""
        position = self._positions.get(fingerprint)
        if position is None:
            raise LookupError("no active PAPER position matches candidate identity")
        if position.exit_lifecycle is not RuntimeExitLifecycle.NONE:
            return False
        if reason not in {ExitReason.STRATEGY_EXIT, ExitReason.TIME_EXIT, ExitReason.MANUAL}:
            raise ValueError("PAPER market exit reason must be generic")
        updated = position.model_copy(
            update={
                "exit_lifecycle": RuntimeExitLifecycle.REQUESTED,
                "requested_exit_reason": reason,
                "requested_exit_at": requested_at,
            }
        )
        self._positions[fingerprint] = updated
        self.state_store.save_position(updated)
        return True

    def cancel_pending_entries(self) -> tuple[AllocationDecision, ...]:
        """Terminally remove every unfilled PAPER entry."""
        decisions = tuple(
            pending.decision for _, pending in sorted(self._pending.items())
        )
        self._pending.clear()
        return decisions

    def cancel_pending_actual_entries(self) -> tuple[AllocationDecision, ...]:
        """Cancel only actual pending entries; existing shadow trackers continue."""
        selected = tuple(
            (fingerprint, pending)
            for fingerprint, pending in sorted(self._pending.items())
            if not pending.is_shadow
        )
        for fingerprint, _ in selected:
            del self._pending[fingerprint]
        return tuple(pending.decision for _, pending in selected)

    def request_strategy_exit(
        self,
        candidate_fingerprint: str,
        requested_at: datetime,
    ) -> bool:
        return self.request_exit(
            candidate_fingerprint, requested_at, ExitReason.STRATEGY_EXIT
        )

    def force_square_off(self, requested_at: datetime) -> tuple[AllocationDecision, ...]:
        cancelled = self.cancel_pending_entries()
        for position in self.positions:
            self.request_exit(
                position.candidate_fingerprint, requested_at, ExitReason.TIME_EXIT
            )
        return cancelled

    def reconcile(self, reconciled_at: datetime) -> None:
        del reconciled_at

    def restore_position(self, position: RuntimePositionRecord) -> None:
        """Restore one unambiguous persisted PAPER position during same-day recovery."""
        if position.runtime_session_id != self.runtime_session_id:
            raise ValueError("position belongs to another runtime session")
        self._positions[position.candidate_fingerprint] = position

    def update_dynamic_position(self, position: RuntimePositionRecord) -> None:
        """Replace and persist one already-open PAPER/shadow dynamic state."""
        if position.candidate_fingerprint not in self._positions:
            raise LookupError("dynamic position update requires an open PAPER position")
        self._positions[position.candidate_fingerprint] = position
        self.state_store.save_position(position)

    def _add_pending(
        self,
        plan: RuntimeTradePlan,
        decision: AllocationDecision,
        *,
        is_shadow: bool,
    ) -> None:
        fingerprint = candidate_fingerprint(plan.candidate)
        if fingerprint in self._pending or fingerprint in self._positions:
            raise ValueError("candidate already has PAPER execution state")
        self._pending[fingerprint] = _PendingPaperEntry(
            plan=plan,
            decision=decision,
            candidate_fingerprint=fingerprint,
            is_shadow=is_shadow,
        )

    def _paper_exit_result(
        self,
        position: RuntimePositionRecord,
        point_bar: pl.DataFrame,
    ) -> ExitResult | None:
        signal = position.candidate.order_intent.signal
        if position.exit_lifecycle is RuntimeExitLifecycle.REQUESTED:
            if position.requested_exit_reason is None or position.requested_exit_at is None:
                raise RuntimeError("requested PAPER exit lacks reason or timestamp")
            return self.simulator.fill_market_exit(
                side=signal.side,
                symbol=signal.symbol,
                quantity=position.entry_fill.quantity,
                requested_at=position.requested_exit_at,
                exit_reason=position.requested_exit_reason,
                candles=point_bar,
            )
        protective_exit = effective_runtime_protective_exit(position)
        if protective_exit is None:
            return None
        result = self.simulator.fill_protective_exit(
            side=signal.side,
            symbol=signal.symbol,
            quantity=position.entry_fill.quantity,
            entry_fill=position.entry_fill,
            protective_exit=protective_exit,
            candles=point_bar,
        )
        if result is None or position.dynamic_exit_state is None:
            return result
        return replace(
            result,
            exit_reason_detail=runtime_exit_detail(position, result.exit_reason),
        )


class LiveExecutionGateway:
    """Persisted exactly-once-attempt boundary around AngelOneBroker orders."""

    def __init__(
        self,
        runtime_session_id: str,
        broker: AngelOneBroker,
        instrument_master: AngelOneInstrumentMaster,
        state_store: RuntimeStateStore,
        *,
        live_order_submission_enabled: bool,
        halt_callback: Callable[[str, datetime], None],
    ) -> None:
        self.runtime_session_id = runtime_session_id
        self.broker = broker
        self.instrument_master = instrument_master
        self.state_store = state_store
        self.live_order_submission_enabled = live_order_submission_enabled
        self._halt = halt_callback

    def submit_allocated_entry(
        self,
        plan: RuntimeTradePlan,
        decision: AllocationDecision,
        intended_at: datetime,
    ) -> RuntimeSubmissionRecord:
        """Persist an entry intent, call Broker once, then persist acknowledgement."""
        if not self.live_order_submission_enabled:
            raise PermissionError("LIVE order submission safety interlock is disabled")
        if decision.outcome is not AllocationOutcome.ALLOCATED:
            raise ValueError("live entry submission requires an ALLOCATED decision")
        signal = plan.candidate.order_intent.signal
        return self._submit(
            plan=plan,
            leg=RuntimeOrderLeg.ENTRY,
            quantity=plan.candidate.order_intent.quantity,
            action=entry_action(signal.side),
            order_type=plan.candidate.order_intent.order_type,
            limit_price=plan.candidate.order_intent.limit_price,
            scrip_consent=plan.scrip_consent,
            intended_at=intended_at,
            exit_reason=None,
            attempt=1,
        )

    def submit_exit(
        self,
        position: RuntimePositionRecord,
        reason: ExitReason,
        intended_at: datetime,
        *,
        attempt: int,
    ) -> RuntimeSubmissionRecord:
        """Persist one deterministic MARKET exit attempt for proven remaining exposure."""
        if not self.live_order_submission_enabled:
            raise PermissionError("LIVE order submission safety interlock is disabled")
        signal = position.candidate.order_intent.signal
        plan = RuntimeTradePlan(
            candidate=position.candidate,
            protective_exit=position.protective_exit,
        )
        remaining_quantity = position.entry_fill.quantity - position.exit_filled_quantity
        if remaining_quantity <= 0:
            raise ValueError("no remaining position quantity is available to exit")
        return self._submit(
            plan=plan,
            leg=RuntimeOrderLeg.EXIT,
            quantity=remaining_quantity,
            action=exit_action(signal.side),
            order_type=OrderType.MARKET,
            limit_price=None,
            scrip_consent=False,
            intended_at=intended_at,
            exit_reason=reason,
            attempt=attempt,
        )

    def reconcile_order(
        self,
        order: RuntimeOrderRecord,
        reconciled_at: datetime,
    ) -> LiveReconciliationResult:
        """Reconcile one persisted exact identity and retain all broker fill evidence."""
        try:
            if order.unique_order_id:
                snapshot = self.broker.get_order(unique_order_id=order.unique_order_id)
            elif order.broker_order_id:
                snapshot = self.broker.get_order(broker_order_id=order.broker_order_id)
            else:
                snapshot = self.broker.get_order(
                    broker_order_tag=order.broker_order_tag
                    or broker_order_tag(order.client_order_id)
                )
        except (BrokerAmbiguousStateError, LookupError) as error:
            self._halt("live order reconciliation is ambiguous", reconciled_at)
            raise BrokerAmbiguousStateError(
                "persisted runtime order could not be reconciled exactly"
            ) from error
        self.state_store.record_order_snapshot(
            order.runtime_session_id, order.client_order_id, snapshot, reconciled_at
        )
        fills = tuple(
            fill
            for fill in self.broker.list_trade_fills()
            if fill.broker_order_id == snapshot.broker_order_id
        )
        for fill in fills:
            self.state_store.record_broker_fill(
                order.runtime_session_id, order.client_order_id, fill
            )
        lifecycle = _runtime_lifecycle(snapshot.state)
        updated = order.model_copy(
            update={
                "lifecycle": lifecycle,
                "broker_order_id": snapshot.broker_order_id,
                "unique_order_id": snapshot.unique_order_id,
                "broker_order_tag": snapshot.broker_order_tag or order.broker_order_tag,
            }
        )
        self.state_store.update_order(
            updated, occurred_at=reconciled_at, event_type=f"ORDER_{lifecycle.value}"
        )
        aggregate = aggregate_broker_fills(fills) if fills else None
        return LiveReconciliationResult(
            runtime_order=updated,
            broker_order=snapshot,
            broker_fills=fills,
            aggregate_fill=aggregate,
        )

    def cancel_entry_order(
        self,
        order: RuntimeOrderRecord,
        requested_at: datetime,
    ) -> LiveReconciliationResult:
        """Request cancellation once, then reconcile because acknowledgement is not final."""
        if order.leg is not RuntimeOrderLeg.ENTRY:
            raise ValueError("only an entry order may be cancelled by square-off")
        if order.cancellation_requested_at is not None:
            return self.reconcile_order(order, requested_at)
        if not order.broker_order_id:
            self._halt("entry cancellation lacks a broker order ID", requested_at)
            raise BrokerAmbiguousStateError("cannot cancel an entry without broker order ID")
        cancellation = order.model_copy(
            update={"cancellation_requested_at": requested_at}
        )
        self.state_store.update_order(
            cancellation,
            occurred_at=requested_at,
            event_type="ORDER_CANCELLATION_REQUESTED",
        )
        try:
            self.broker.cancel_order(order.broker_order_id, requested_at)
        except Exception as error:
            self._halt("entry cancellation outcome is ambiguous", requested_at)
            raise BrokerApiError(
                "entry cancellation outcome is ambiguous; no retry attempted"
            ) from error
        self.state_store.append_event(
            order.runtime_session_id,
            requested_at,
            "ORDER_CANCELLATION_ACKNOWLEDGED",
            f"client_order_id={order.client_order_id}",
        )
        return self.reconcile_order(cancellation, requested_at)

    def on_market_tick(self, tick: BrokerMarketTick) -> None:
        """LIVE fills are broker evidence; observed-tick policy remains in RuntimeService."""
        if not isinstance(tick, BrokerMarketTick):
            raise TypeError("tick must be a BrokerMarketTick")

    def request_strategy_exit(
        self,
        position: RuntimePositionRecord,
        requested_at: datetime,
    ) -> RuntimeSubmissionRecord | None:
        attempt = self._next_exit_attempt(position.candidate_fingerprint)
        if attempt is None:
            return None
        return self.submit_exit(position, ExitReason.STRATEGY_EXIT, requested_at, attempt=attempt)

    def force_square_off(
        self,
        positions: Iterable[RuntimePositionRecord],
        requested_at: datetime,
    ) -> tuple[RuntimeSubmissionRecord, ...]:
        submissions = []
        for position in positions:
            attempt = self._next_exit_attempt(position.candidate_fingerprint)
            if attempt is not None:
                submissions.append(
                    self.submit_exit(
                        position, ExitReason.TIME_EXIT, requested_at, attempt=attempt
                    )
                )
        return tuple(submissions)

    def reconcile(self, reconciled_at: datetime) -> tuple[LiveReconciliationResult, ...]:
        return tuple(
            self.reconcile_order(order, reconciled_at)
            for order in self.state_store.list_orders(self.runtime_session_id)
        )

    def _next_exit_attempt(self, fingerprint: str) -> int | None:
        orders = sorted(
            (
                order
                for order in self.state_store.list_orders(self.runtime_session_id)
                if order.leg is RuntimeOrderLeg.EXIT
                and order.candidate_fingerprint == fingerprint
            ),
            key=lambda order: order.attempt,
        )
        if not orders:
            return 1
        latest = orders[-1]
        if latest.lifecycle in {
            RuntimeOrderLifecycle.CANCELLED,
            RuntimeOrderLifecycle.REJECTED,
        }:
            return latest.attempt + 1
        return None

    def _submit(
        self,
        *,
        plan: RuntimeTradePlan,
        leg: RuntimeOrderLeg,
        quantity: int,
        action: BrokerTransactionAction,
        order_type: OrderType,
        limit_price: Decimal | None,
        scrip_consent: bool,
        intended_at: datetime,
        exit_reason: ExitReason | None,
        attempt: int,
    ) -> RuntimeSubmissionRecord:
        candidate = plan.candidate
        fingerprint = candidate_fingerprint(candidate)
        client_order_id = runtime_client_order_id(
            self.runtime_session_id, candidate.identity, leg, attempt
        )
        instrument = self.instrument_master.resolve(candidate.order_intent.signal.symbol)
        request = BrokerOrderRequest(
            client_order_id=client_order_id,
            instrument=instrument,
            transaction_action=action,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            submitted_at=intended_at,
            scrip_consent=scrip_consent,
        )
        record = RuntimeOrderRecord(
            runtime_session_id=self.runtime_session_id,
            client_order_id=client_order_id,
            candidate_fingerprint=fingerprint,
            leg=leg,
            attempt=attempt,
            symbol=instrument.symbol,
            quantity=quantity,
            transaction_action=action,
            order_type=order_type,
            limit_price=limit_price,
            intended_at=intended_at,
            broker_order_tag=broker_order_tag(client_order_id),
            exit_reason=exit_reason,
        )
        self.state_store.record_order_intent(record)
        try:
            acknowledgement = self.broker.place_order(request, intended_at)
        except Exception as error:
            ambiguous = record.model_copy(
                update={"lifecycle": RuntimeOrderLifecycle.SUBMISSION_AMBIGUOUS}
            )
            self.state_store.update_order(
                ambiguous,
                occurred_at=intended_at,
                event_type="SUBMISSION_AMBIGUOUS",
            )
            self._halt("broker submission outcome is ambiguous", intended_at)
            raise BrokerApiError(
                "broker submission outcome is ambiguous; no retry attempted"
            ) from error
        acknowledged = record.model_copy(
            update={
                "lifecycle": RuntimeOrderLifecycle.ACKNOWLEDGED,
                "broker_order_tag": acknowledgement.broker_order_tag,
                "broker_order_id": acknowledgement.broker_order_id,
                "unique_order_id": acknowledgement.unique_order_id,
                "acknowledged_at": acknowledgement.acknowledged_at,
            }
        )
        self.state_store.update_order(
            acknowledged,
            occurred_at=acknowledgement.acknowledged_at,
            event_type="ORDER_ACKNOWLEDGED",
        )
        return RuntimeSubmissionRecord(
            runtime_order=acknowledged,
            acknowledgement=acknowledgement,
        )


def point_bar_from_tick(tick: BrokerMarketTick) -> pl.DataFrame:
    """Create an ephemeral canonical point-bar from one exact observed tick."""
    if not isinstance(tick, BrokerMarketTick):
        raise TypeError("tick must be a BrokerMarketTick")
    price = float(tick.last_traded_price)
    return pl.DataFrame(
        {
            "timestamp": [tick.exchange_timestamp],
            "open": [price],
            "high": [price],
            "low": [price],
            "close": [price],
            "volume": [float(tick.cumulative_volume or 0)],
            "symbol": [tick.instrument.symbol],
        },
        schema={
            "timestamp": pl.Datetime(time_unit="us", time_zone="Asia/Kolkata"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "symbol": pl.String,
        },
    )


def aggregate_broker_fills(fills: Iterable[BrokerTradeFill]) -> Fill:
    """Aggregate compatible broker evidence without deleting its source rows."""
    selected = tuple(fills)
    if not selected:
        raise ValueError("at least one broker fill is required")
    first = selected[0]
    if any(
        fill.broker_order_id != first.broker_order_id
        or fill.instrument != first.instrument
        or fill.transaction_action is not first.transaction_action
        for fill in selected[1:]
    ):
        raise ValueError("broker fills must share order, instrument, and action")
    total_quantity = sum(fill.quantity for fill in selected)
    weighted_price = sum(
        (fill.fill_price * fill.quantity for fill in selected), Decimal("0")
    ) / total_quantity
    return Fill(
        timestamp=max(fill.fill_timestamp for fill in selected),
        price=weighted_price,
        quantity=total_quantity,
        slippage_per_unit=Decimal("0"),
        is_simulated=False,
    )


def live_protective_reason(
    position: RuntimePositionRecord,
    tick: BrokerMarketTick,
) -> ExitReason | None:
    """Return the threshold proven by one observed live LTP, if any."""
    protective = effective_runtime_protective_exit(position)
    if (
        protective is None
        or tick.instrument.symbol != position.candidate.order_intent.signal.symbol
    ):
        return None
    side = position.candidate.order_intent.signal.side
    price = tick.last_traded_price
    if side is Side.LONG:
        if protective.stop_price is not None and price <= protective.stop_price:
            return ExitReason.STOP_LOSS
        if protective.target_price is not None and price >= protective.target_price:
            return ExitReason.TARGET_REACHED
    else:
        if protective.stop_price is not None and price >= protective.stop_price:
            return ExitReason.STOP_LOSS
        if protective.target_price is not None and price <= protective.target_price:
            return ExitReason.TARGET_REACHED
    return None


def initialize_runtime_dynamic_exit(
    plan: RuntimeTradePlan,
    entry_fill: Fill,
) -> RuntimeDynamicExitState | None:
    policy = plan.dynamic_exit_policy
    if policy is None:
        return None
    side = plan.candidate.order_intent.signal.side
    core = initialize_r_multiple_state(side, entry_fill.price, policy.initial_stop_price)
    target = (
        entry_fill.price + policy.hard_target_r * core.risk
        if side is Side.LONG
        else entry_fill.price - policy.hard_target_r * core.risk
    )
    if target <= 0:
        raise ValueError("runtime dynamic exit produced a nonpositive hard target")
    return RuntimeDynamicExitState(
        entry_price=core.entry_price,
        initial_stop=core.initial_stop,
        risk=core.risk,
        current_stop=core.current_stop,
        best_favorable=core.best_favorable,
        hard_target=target,
    )


def effective_runtime_protective_exit(
    position: RuntimePositionRecord,
) -> ProtectiveExitSpec | None:
    if position.dynamic_exit_state is None:
        return position.protective_exit
    return ProtectiveExitSpec(
        stop_price=position.dynamic_exit_state.current_stop,
        target_price=position.dynamic_exit_state.hard_target,
    )


def runtime_exit_detail(
    position: RuntimePositionRecord,
    reason: ExitReason,
) -> ExitReasonDetail | None:
    state = position.dynamic_exit_state
    policy = position.dynamic_exit_policy
    if state is None or policy is None:
        return None
    if reason is ExitReason.TARGET_REACHED:
        return ExitReasonDetail.HARD_TARGET
    if reason is not ExitReason.STOP_LOSS:
        return None
    return r_multiple_stop_detail(
        RMultipleTrailingState(
            entry_price=state.entry_price,
            initial_stop=state.initial_stop,
            risk=state.risk,
            current_stop=state.current_stop,
            best_favorable=state.best_favorable,
        ),
        position.candidate.order_intent.signal.side,
        RMultipleTrailingCoreParameters(
            breakeven_trigger_r=policy.breakeven_trigger_r,
            breakeven_stop_r=policy.breakeven_stop_r,
            profit_lock_trigger_r=policy.profit_lock_trigger_r,
            profit_lock_stop_r=policy.profit_lock_stop_r,
            trailing_distance_r=policy.trailing_distance_r,
        ),
    )


def update_position_excursion(
    position: RuntimePositionRecord, tick: BrokerMarketTick
) -> RuntimePositionRecord:
    if tick.exchange_timestamp <= position.entry_fill.timestamp:
        return position
    entry = position.entry_fill.price
    price = tick.last_traded_price
    side = position.candidate.order_intent.signal.side
    observed_return = (price - entry) / entry
    if side is Side.SHORT:
        observed_return = -observed_return
    return position.model_copy(
        update={
            "mfe_return": max(position.mfe_return, observed_return, Decimal("0")),
            "mae_return": min(position.mae_return, observed_return, Decimal("0")),
        }
    )


def _runtime_lifecycle(state: BrokerOrderState) -> RuntimeOrderLifecycle:
    mapping = {
        BrokerOrderState.SUBMITTED: RuntimeOrderLifecycle.ACKNOWLEDGED,
        BrokerOrderState.OPEN: RuntimeOrderLifecycle.OPEN,
        BrokerOrderState.PENDING: RuntimeOrderLifecycle.OPEN,
        BrokerOrderState.PARTIALLY_FILLED: RuntimeOrderLifecycle.PARTIALLY_FILLED,
        BrokerOrderState.FILLED: RuntimeOrderLifecycle.FILLED,
        BrokerOrderState.CANCELLED: RuntimeOrderLifecycle.CANCELLED,
        BrokerOrderState.REJECTED: RuntimeOrderLifecycle.REJECTED,
        BrokerOrderState.UNKNOWN: RuntimeOrderLifecycle.UNKNOWN,
    }
    return mapping[state]
