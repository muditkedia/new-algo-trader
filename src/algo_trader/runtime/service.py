"""Central Runtime lifecycle, allocation, safety, and mode orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from threading import RLock, Thread
from zoneinfo import ZoneInfo

import polars as pl

from algo_trader.broker import (
    AngelOneBroker,
    AngelOneInstrumentMaster,
    BrokerAmbiguousStateError,
    BrokerMarketTick,
    BrokerOrderState,
    BrokerTradeFill,
)
from algo_trader.costs import calculate_round_trip_costs, get_fixed_current_backtest_cost_policy
from algo_trader.data import bar_available_at
from algo_trader.domain import ExitReason, Fill, Side, SignalStatus, Trade
from algo_trader.execution import (
    ExitResult,
    FixedBasisPointsSlippage,
    HistoricalExecutionSimulator,
    RMultipleTrailingCoreParameters,
    RMultipleTrailingState,
    advance_r_multiple_state,
)
from algo_trader.portfolio import (
    AllocationBatchResult,
    AllocationDecision,
    AllocationOutcome,
    CapitalAllocator,
    MarginRequirementProvider,
    PortfolioState,
)
from algo_trader.runtime.calendar import TradingDayProvider
from algo_trader.runtime.clock import Clock
from algo_trader.runtime.execution import (
    LiveExecutionGateway,
    PaperExecutionGateway,
    initialize_runtime_dynamic_exit,
    live_protective_reason,
    runtime_exit_detail,
    update_position_excursion,
)
from algo_trader.runtime.identity import candidate_fingerprint, runtime_config_fingerprint
from algo_trader.runtime.models import (
    LiveReconciliationResult,
    RuntimeConfig,
    RuntimeDynamicExitState,
    RuntimeExitLifecycle,
    RuntimeMode,
    RuntimeOrderLeg,
    RuntimeOrderLifecycle,
    RuntimeOrderRecord,
    RuntimePhase,
    RuntimePositionRecord,
    RuntimeSessionRecord,
    RuntimeTradePlan,
    RuntimeTradeRecord,
)
from algo_trader.runtime.state import RuntimeStateStore

MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")


class RuntimeService:
    """Single-process safety coordinator around frozen platform components."""

    def __init__(
        self,
        *,
        runtime_session_id: str,
        trading_date: date,
        config: RuntimeConfig,
        clock: Clock,
        trading_calendar: TradingDayProvider,
        state_store: RuntimeStateStore,
        margin_provider: MarginRequirementProvider,
        capital_allocator: CapitalAllocator | None = None,
        broker: AngelOneBroker | None = None,
        instrument_master: AngelOneInstrumentMaster | None = None,
        simulator: HistoricalExecutionSimulator | None = None,
        stream: object | None = None,
        logout_callback: object | None = None,
    ) -> None:
        if not isinstance(runtime_session_id, str) or not runtime_session_id.strip():
            raise ValueError("runtime_session_id must be a non-empty string")
        if not isinstance(config, RuntimeConfig):
            raise TypeError("config must be a RuntimeConfig")
        if not isinstance(clock, Clock):
            raise TypeError("clock must implement Clock")
        if not isinstance(trading_calendar, TradingDayProvider):
            raise TypeError("trading_calendar must implement TradingDayProvider")
        if not isinstance(margin_provider, MarginRequirementProvider):
            raise TypeError("margin_provider must implement MarginRequirementProvider")
        self.runtime_session_id = runtime_session_id.strip()
        self.trading_date = trading_date
        self.config = config
        self.clock = clock
        self.trading_calendar = trading_calendar
        self.state_store = state_store
        self.margin_provider = margin_provider
        self.capital_allocator = capital_allocator or CapitalAllocator()
        self.broker = broker
        self.instrument_master = instrument_master
        self.stream = stream
        self.logout_callback = logout_callback
        self._lock = RLock()
        self._session: RuntimeSessionRecord | None = None
        self._economic_capital = config.starting_capital
        self._portfolio_state = PortfolioState(capital_limit=config.starting_capital)
        self._last_tick_time: dict[str, datetime] = {}
        self._live_positions: dict[str, RuntimePositionRecord] = {}
        self._stream_connected = False
        self._stream_thread: Thread | None = None
        self._subscribed_instruments: tuple[object, ...] = ()

        paper_simulator = simulator or HistoricalExecutionSimulator(
            slippage_model=FixedBasisPointsSlippage(config.paper_slippage_bps)
        )
        self._paper_gateway = PaperExecutionGateway(
            self.runtime_session_id, state_store, paper_simulator
        )
        self._live_gateway: LiveExecutionGateway | None = None
        if config.mode is RuntimeMode.LIVE:
            if broker is None or instrument_master is None:
                raise ValueError("LIVE Runtime requires broker and instrument_master")
            self._live_gateway = LiveExecutionGateway(
                self.runtime_session_id,
                broker,
                instrument_master,
                state_store,
                live_order_submission_enabled=config.live_order_submission_enabled,
                halt_callback=self.halt,
            )

    @property
    def phase(self) -> RuntimePhase:
        return RuntimePhase.CREATED if self._session is None else self._session.phase

    @property
    def portfolio_state(self) -> PortfolioState:
        return self._portfolio_state

    @property
    def economic_capital(self) -> Decimal:
        return self._economic_capital

    @property
    def positions(self) -> tuple[RuntimePositionRecord, ...]:
        if self.config.mode is RuntimeMode.PAPER:
            return self._paper_gateway.positions
        return tuple(self._live_positions[key] for key in sorted(self._live_positions))

    def start(self) -> RuntimePhase:
        """Create or recover one session, reconcile, and derive phase from injected time."""
        with self._lock:
            now = self._now()
            unfinished = self.state_store.list_unfinished_sessions(self.trading_date)
            if len(unfinished) > 1:
                identities = ", ".join(
                    session.runtime_session_id for session in unfinished
                )
                raise RuntimeError(
                    f"multiple unfinished runtime sessions exist for {self.trading_date}: "
                    f"{identities}"
                )
            if unfinished and unfinished[0].runtime_session_id != self.runtime_session_id:
                raise RuntimeError(
                    "unfinished same-day runtime session must be recovered with its exact "
                    f"runtime_session_id: {unfinished[0].runtime_session_id}"
                )
            persisted = unfinished[0] if unfinished else None
            if persisted is None:
                self._session = RuntimeSessionRecord(
                    runtime_session_id=self.runtime_session_id,
                    trading_date=self.trading_date,
                    mode=self.config.mode,
                    starting_capital=self.config.starting_capital,
                    current_capital=self.config.starting_capital,
                    started_at=now,
                    live_order_submission_enabled=self.config.live_order_submission_enabled,
                    configuration_fingerprint=runtime_config_fingerprint(self.config),
                )
                self.state_store.create_session(self._session)
            else:
                self._validate_recovery_session(persisted)
                self._session = persisted
                self._economic_capital = persisted.current_capital
                if self._economic_capital <= 0:
                    self.halt("persisted economic capital is non-positive", now)
                    return self.phase
                reservations = self.state_store.load_reservations(self.runtime_session_id)
                self._portfolio_state = PortfolioState(
                    capital_limit=self._economic_capital,
                    active_reservations=reservations,
                )
                self._restore_execution_state()

            if not self.trading_calendar.is_trading_day(self.trading_date):
                self.halt("configured date is not an explicit trading day", now)
                return self.phase
            if self.config.mode is RuntimeMode.LIVE:
                try:
                    self.reconcile(now)
                except Exception:
                    if self.phase is not RuntimePhase.HALTED:
                        self.halt("startup broker reconciliation failed", now)
                    return self.phase
            self._derive_phase(now)
            return self.phase

    def market_open(self, occurred_at: datetime | None = None) -> None:
        with self._lock:
            now = self._at(occurred_at)
            if self.phase is RuntimePhase.HALTED:
                return
            if now >= self._session_at(self.config.session_times.entry_cutoff_time):
                self.close_entries(now)
                return
            if now < self._session_at(self.config.session_times.market_open_time):
                raise ValueError("market_open cannot precede configured market open")
            self._transition(RuntimePhase.TRADING, now, "MARKET_OPEN")

    def close_entries(self, occurred_at: datetime | None = None) -> None:
        with self._lock:
            now = self._at(occurred_at)
            if self.phase in {RuntimePhase.SQUARE_OFF, RuntimePhase.STOPPED}:
                return
            was_halted = self.phase is RuntimePhase.HALTED
            if was_halted:
                self.state_store.append_event(
                    self.runtime_session_id,
                    now,
                    "ENTRY_CUTOFF_SAFETY_WHILE_HALTED",
                )
            else:
                self._transition(RuntimePhase.ENTRY_CLOSED, now, "ENTRY_CUTOFF")
            if self.config.mode is RuntimeMode.PAPER:
                self._cancel_paper_pending(
                    self._paper_gateway.cancel_pending_actual_entries()
                )
            else:
                self._cancel_live_entry_remainders(now)
            if was_halted and self.phase is not RuntimePhase.HALTED:
                raise RuntimeError("entry-cutoff cleanup must preserve HALTED phase")

    def process_plans(
        self,
        plans: Sequence[RuntimeTradePlan],
        *,
        decision_at: datetime | None = None,
    ) -> AllocationBatchResult:
        """Apply hard entry safety, then delegate one batch unchanged to CapitalAllocator."""
        with self._lock:
            now = self._at(decision_at)
            selected = tuple(plans)
            if not selected:
                raise ValueError("at least one RuntimeTradePlan is required")
            if any(not isinstance(plan, RuntimeTradePlan) for plan in selected):
                raise TypeError("all plans must be RuntimeTradePlan instances")
            self._require_entry_enabled(now)
            seen_candidates = {}
            for plan in selected:
                existing = seen_candidates.get(plan.candidate.identity)
                if existing is None:
                    seen_candidates[plan.candidate.identity] = plan.candidate
                    continue
                if existing.ml_score != plan.candidate.ml_score:
                    raise ValueError(
                        "duplicate RuntimeTradePlan identity has conflicting MLScore information"
                    )
                raise ValueError("duplicate RuntimeTradePlan candidate identity")
            try:
                result = self.capital_allocator.allocate_batch(
                    (plan.candidate for plan in selected),
                    self._portfolio_state,
                    self.margin_provider,
                )
            except Exception:
                self.halt("margin/allocation batch failed without fallback", now)
                raise
            if self.config.mode is RuntimeMode.LIVE:
                if self.broker is None:
                    raise RuntimeError("LIVE broker funds safety requires a broker")
                funds = self.broker.get_funds()
                external_ceiling = (
                    funds.available_limit_margin
                    if funds.available_limit_margin is not None
                    else funds.available_cash
                )
                if (
                    external_ceiling < 0
                    or result.ending_state.reserved_margin > external_ceiling
                ):
                    self.halt("broker funds ceiling would be exceeded", now)
                    raise RuntimeError(
                        "LIVE allocation exceeds the independent broker funds ceiling"
                    )
            plans_by_fingerprint = {
                candidate_fingerprint(plan.candidate): plan for plan in selected
            }
            persistence_rows = []
            for decision in result.decisions:
                fingerprint = candidate_fingerprint(decision.candidate)
                plan = plans_by_fingerprint[fingerprint]
                persistence_rows.append((fingerprint, plan, decision))
                if decision.outcome is AllocationOutcome.ALLOCATED:
                    if decision.reservation is None:
                        raise RuntimeError("allocator returned allocation without reservation")
            try:
                self.state_store.save_allocation_batch(
                    self.runtime_session_id, persistence_rows, now
                )
            except Exception:
                self.halt("allocation batch persistence failed before execution", now)
                raise
            self._portfolio_state = result.ending_state
            for decision in result.decisions:
                fingerprint = candidate_fingerprint(decision.candidate)
                plan = plans_by_fingerprint[fingerprint]
                if decision.outcome is AllocationOutcome.ALLOCATED:
                    if self.config.mode is RuntimeMode.PAPER:
                        self._paper_gateway.submit_allocated_entry(plan, decision, now)
                    else:
                        if self._live_gateway is None:
                            raise RuntimeError("LIVE gateway is unavailable")
                        self._live_gateway.submit_allocated_entry(plan, decision, now)
                else:
                    self._paper_gateway.submit_shadow(plan, decision)
            return result

    def on_market_tick(self, tick: BrokerMarketTick) -> tuple[RuntimeTradeRecord, ...]:
        """Update health/excursion and drive PAPER/shadow or LIVE protective exits."""
        with self._lock:
            self._last_tick_time[tick.instrument.symbol] = tick.exchange_timestamp
            completed: list[RuntimeTradeRecord] = []
            paper_result = self._paper_gateway.on_market_tick(tick)
            for position, exit_result in paper_result.closed:
                completed.append(self._complete_trade(position, exit_result))
            if self.config.mode is RuntimeMode.LIVE:
                for fingerprint, position in tuple(sorted(self._live_positions.items())):
                    if position.candidate.order_intent.signal.symbol != tick.instrument.symbol:
                        continue
                    updated = update_position_excursion(position, tick)
                    reason = live_protective_reason(updated, tick)
                    if reason is not None and updated.exit_lifecycle is RuntimeExitLifecycle.NONE:
                        updated = self._request_live_exit(updated, reason, tick.exchange_timestamp)
                    self._live_positions[fingerprint] = updated
                    self.state_store.save_position(updated)
            return tuple(completed)

    def advance_dynamic_exits(
        self,
        completed_candles: Mapping[str, pl.DataFrame],
        occurred_at: datetime | None = None,
    ) -> None:
        """Apply completed-bar trailing updates for the next bar and persist them."""
        with self._lock:
            now = self._at(occurred_at)
            positions = self.positions
            if self.config.mode is RuntimeMode.LIVE:
                positions += tuple(
                    position
                    for position in self._paper_gateway.positions
                    if position.is_shadow
                )
            for position in positions:
                policy = position.dynamic_exit_policy
                persisted = position.dynamic_exit_state
                if policy is None or persisted is None:
                    continue
                symbol = position.candidate.order_intent.signal.symbol
                candles = completed_candles.get(symbol)
                if candles is None:
                    raise RuntimeError(f"dynamic exit update lacks candles for {symbol}")
                side = position.candidate.order_intent.signal.side
                core = RMultipleTrailingState(
                    entry_price=persisted.entry_price,
                    initial_stop=persisted.initial_stop,
                    risk=persisted.risk,
                    current_stop=persisted.current_stop,
                    best_favorable=persisted.best_favorable,
                )
                parameters = RMultipleTrailingCoreParameters(
                    breakeven_trigger_r=policy.breakeven_trigger_r,
                    breakeven_stop_r=policy.breakeven_stop_r,
                    profit_lock_trigger_r=policy.profit_lock_trigger_r,
                    profit_lock_stop_r=policy.profit_lock_stop_r,
                    trailing_distance_r=policy.trailing_distance_r,
                )
                last_start = persisted.last_completed_bar_start
                changed = False
                for row in candles.iter_rows(named=True):
                    bar_start = row["timestamp"]
                    if bar_start < position.entry_fill.timestamp:
                        continue
                    if last_start is not None and bar_start <= last_start:
                        continue
                    if bar_available_at(bar_start) > now:
                        continue
                    favorable = Decimal(
                        str(row["high"] if side is Side.LONG else row["low"])
                    )
                    core = advance_r_multiple_state(
                        core, side, favorable, parameters
                    )
                    last_start = bar_start
                    changed = True
                if not changed:
                    continue
                updated = position.model_copy(
                    update={
                        "dynamic_exit_state": RuntimeDynamicExitState(
                            entry_price=core.entry_price,
                            initial_stop=core.initial_stop,
                            risk=core.risk,
                            current_stop=core.current_stop,
                            best_favorable=core.best_favorable,
                            hard_target=persisted.hard_target,
                            last_completed_bar_start=last_start,
                        )
                    }
                )
                if self.config.mode is RuntimeMode.PAPER or position.is_shadow:
                    self._paper_gateway.update_dynamic_position(updated)
                else:
                    self._live_positions[position.candidate_fingerprint] = updated
                    self.state_store.save_position(updated)

    def request_strategy_exit(
        self,
        candidate_fingerprint_value: str,
        requested_at: datetime | None = None,
    ) -> bool:
        """Request exactly one STRATEGY_EXIT for an active filled position."""
        with self._lock:
            now = self._at(requested_at)
            if self.config.mode is RuntimeMode.PAPER:
                return self._paper_gateway.request_exit(
                    candidate_fingerprint_value, now, ExitReason.STRATEGY_EXIT
                )
            position = self._live_positions.get(candidate_fingerprint_value)
            if position is None:
                raise LookupError("no active LIVE position matches candidate identity")
            if position.exit_lifecycle is not RuntimeExitLifecycle.NONE:
                return False
            self._live_positions[candidate_fingerprint_value] = self._request_live_exit(
                position, ExitReason.STRATEGY_EXIT, now
            )
            return True

    def force_square_off(self, occurred_at: datetime | None = None) -> None:
        """Close entries, reconcile partials, and target only actual filled exposure."""
        with self._lock:
            now = self._at(occurred_at)
            if self.phase is not RuntimePhase.HALTED:
                self._transition(RuntimePhase.SQUARE_OFF, now, "SQUARE_OFF_STARTED")
            if self.config.mode is RuntimeMode.PAPER:
                self._cancel_paper_pending(self._paper_gateway.cancel_pending_entries())
                for position in self._paper_gateway.positions:
                    self._paper_gateway.request_exit(
                        position.candidate_fingerprint, now, ExitReason.TIME_EXIT
                    )
                return
            if self._live_gateway is None:
                raise RuntimeError("LIVE gateway is unavailable")
            self._cancel_live_entry_remainders(now)
            self.reconcile(now)
            for fingerprint, position in tuple(sorted(self._live_positions.items())):
                self._live_positions[fingerprint] = self._request_live_exit(
                    position, ExitReason.TIME_EXIT, now
                )

    def reconcile(self, reconciled_at: datetime | None = None) -> None:
        """Fail-closed exact LIVE reconciliation; PAPER restores only persisted local state."""
        with self._lock:
            now = self._at(reconciled_at)
            if self.config.mode is RuntimeMode.PAPER:
                return
            if self._live_gateway is None or self.broker is None:
                raise RuntimeError("LIVE reconciliation requires broker gateway")
            orders = sorted(
                self.state_store.list_orders(self.runtime_session_id),
                key=lambda item: (
                    item.leg is RuntimeOrderLeg.EXIT,
                    item.candidate_fingerprint,
                    item.attempt,
                ),
            )
            for order in orders:
                result = self._live_gateway.reconcile_order(order, now)
                self._apply_live_reconciliation(result, now)
            self._reconcile_external_orders(now)
            self._reconcile_external_positions(now)

    def check_market_data_health(
        self,
        checked_at: datetime | None = None,
        *,
        required_symbols: Sequence[str] = (),
    ) -> bool:
        """Halt entries when required symbols have no sufficiently recent tick."""
        with self._lock:
            now = self._at(checked_at)
            required = {
                position.candidate.order_intent.signal.symbol for position in self.positions
            }
            required.update(required_symbols)
            required.update(
                plan.candidate.order_intent.signal.symbol
                for _, plan, _, _ in self.state_store.load_allocations(
                    self.runtime_session_id, status="PENDING"
                )
            )
            stale = [
                symbol
                for symbol in sorted(required)
                if symbol not in self._last_tick_time
                or (now - self._last_tick_time[symbol]).total_seconds()
                > self.config.stale_market_data_seconds
            ]
            if stale:
                self.halt(f"stale market data: {', '.join(stale)}", now)
                return False
            return True

    def on_stream_error(self, error: object, occurred_at: datetime | None = None) -> None:
        """Fail closed on stream errors without automatic reconnect or liquidation."""
        del error
        self.halt("market-data stream error", self._at(occurred_at))

    def connect_stream(self, instruments: Sequence[object]) -> None:
        """Configure once, then own one thread for the SDK's blocking connect call."""
        with self._lock:
            if self.stream is None:
                raise RuntimeError("no market-data stream is configured")
            if self._stream_connected:
                raise RuntimeError("market-data stream is already connected")
            self._subscribed_instruments = tuple(instruments)
            self.stream.configure_initial_subscription(self._subscribed_instruments)
            self._stream_connected = True
            self._stream_thread = Thread(
                target=self._run_stream_connect,
                name=f"runtime-stream-{self.runtime_session_id}",
                daemon=True,
            )
            self._stream_thread.start()

    def halt(self, reason: str, occurred_at: datetime) -> None:
        """Persist a one-way new-entry safety halt without erasing open exposure."""
        with self._lock:
            if self._session is None:
                raise RuntimeError("Runtime session has not started")
            if self.phase is RuntimePhase.HALTED:
                return
            self._session = self._session.model_copy(
                update={"phase": RuntimePhase.HALTED, "halt_reason": reason}
            )
            self.state_store.update_session(
                self._session,
                occurred_at=occurred_at,
                event_type="SESSION_HALTED",
                description=reason,
            )

    def market_close_check(self, occurred_at: datetime | None = None) -> bool:
        with self._lock:
            now = self._at(occurred_at)
            if now < self._session_at(self.config.session_times.market_close_time):
                raise ValueError("market-close safety check is premature")
            if self._actual_positions():
                self.halt("runtime-managed position remains open at market close", now)
                return False
            return True

    def shutdown(self, occurred_at: datetime | None = None) -> RuntimePhase:
        """Explicitly close external resources; unsafe exposure retains HALTED phase."""
        now = self._at(occurred_at)
        stream_thread: Thread | None = None
        stream_shutdown_timed_out = False
        if self.stream is not None and self._stream_connected:
            try:
                if self._subscribed_instruments:
                    self.stream.unsubscribe(self._subscribed_instruments)
            except Exception:
                pass
            self.stream.close()
            stream_thread = self._stream_thread
            if stream_thread is not None:
                stream_thread.join(timeout=self.config.stream_shutdown_timeout_seconds)
                if stream_thread.is_alive():
                    stream_shutdown_timed_out = True
        with self._lock:
            if stream_shutdown_timed_out:
                if self.phase is not RuntimePhase.HALTED:
                    self.halt("market-data stream thread shutdown timed out", now)
                self.state_store.append_event(
                    self.runtime_session_id,
                    now,
                    "STREAM_THREAD_SHUTDOWN_TIMEOUT",
                    "stream thread remained alive after bounded shutdown wait",
                )
            if self.config.mode is RuntimeMode.LIVE:
                try:
                    self.reconcile(now)
                except Exception:
                    if self.phase is not RuntimePhase.HALTED:
                        self.halt("final reconciliation failed", now)
            if self._actual_positions() and self.phase is not RuntimePhase.HALTED:
                self.halt("shutdown attempted with open runtime position", now)
            unresolved = self._unresolved_actual_execution_state()
            if unresolved and self.phase is not RuntimePhase.HALTED:
                self.halt(
                    "shutdown blocked by unresolved actual execution state: "
                    + "; ".join(unresolved),
                    now,
                )
            self._stream_connected = False
            if callable(self.logout_callback):
                self.logout_callback()
            if self._session is None:
                raise RuntimeError("Runtime session has not started")
            terminal_phase = (
                RuntimePhase.HALTED if self.phase is RuntimePhase.HALTED else RuntimePhase.STOPPED
            )
            self._session = self._session.model_copy(
                update={"phase": terminal_phase, "ended_at": now}
            )
            self.state_store.update_session(
                self._session,
                occurred_at=now,
                event_type=(
                    "SESSION_HALTED_END"
                    if terminal_phase is RuntimePhase.HALTED
                    else "SESSION_STOPPED"
                ),
            )
            self.state_store.close()
            return terminal_phase

    def _derive_phase(self, now: datetime) -> None:
        times = self.config.session_times
        if now < self._session_at(times.market_open_time):
            self._transition(RuntimePhase.PREOPEN, now, "PREOPEN_STARTED")
            self._transition(RuntimePhase.READY, now, "STARTUP_CHECKS_PASSED")
        elif now < self._session_at(times.entry_cutoff_time):
            self._transition(RuntimePhase.TRADING, now, "LATE_START_TRADING")
        elif now < self._session_at(times.square_off_time):
            self.close_entries(now)
        elif now < self._session_at(times.market_close_time):
            self._transition(RuntimePhase.SQUARE_OFF, now, "LATE_START_SQUARE_OFF")
            self.force_square_off(now)
        else:
            self.close_entries(now)
            if self.positions:
                self.halt("open position found after market close", now)
            unresolved = self._unresolved_actual_execution_state()
            if unresolved:
                self.halt(
                    "unresolved entry execution state found after market close: "
                    + "; ".join(unresolved),
                    now,
                )

    def _transition(self, phase: RuntimePhase, occurred_at: datetime, event_type: str) -> None:
        if self._session is None:
            raise RuntimeError("Runtime session has not started")
        if self.phase is RuntimePhase.HALTED and phase is not RuntimePhase.HALTED:
            return
        if self.phase is phase:
            return
        self._session = self._session.model_copy(update={"phase": phase})
        self.state_store.update_session(
            self._session, occurred_at=occurred_at, event_type=event_type
        )

    def _require_entry_enabled(self, now: datetime) -> None:
        if self.phase is not RuntimePhase.TRADING:
            raise RuntimeError(f"new entries require TRADING phase, not {self.phase.value}")
        if not self.trading_calendar.is_trading_day(self.trading_date):
            raise RuntimeError("new entries require an explicit trading day")
        if now >= self._session_at(self.config.session_times.entry_cutoff_time):
            self.close_entries(now)
            raise RuntimeError("new entries are prohibited at or after entry cutoff")
        if self.config.mode is RuntimeMode.LIVE and not self.config.live_order_submission_enabled:
            raise PermissionError("LIVE order submission safety interlock is disabled")

    def _request_live_exit(
        self,
        position: RuntimePositionRecord,
        reason: ExitReason,
        requested_at: datetime,
    ) -> RuntimePositionRecord:
        position = self._stabilize_live_entry(position, requested_at)
        attempt = self._next_exit_attempt(position, requested_at)
        if attempt is None:
            return position
        if self._live_gateway is None:
            raise RuntimeError("LIVE gateway is unavailable")
        try:
            submission = self._live_gateway.submit_exit(
                position, reason, requested_at, attempt=attempt
            )
        except Exception:
            latest_ids = tuple(
                order.client_order_id
                for order in self._exit_orders(position.candidate_fingerprint)
            )
            ambiguous = position.model_copy(
                update={
                    "exit_lifecycle": RuntimeExitLifecycle.AMBIGUOUS,
                    "broker_exit_client_order_ids": latest_ids,
                }
            )
            self.state_store.save_position(ambiguous)
            return ambiguous
        requested_reason = position.requested_exit_reason or reason
        updated = position.model_copy(
            update={
                "exit_lifecycle": RuntimeExitLifecycle.ACKNOWLEDGED,
                "requested_exit_reason": requested_reason,
                "requested_exit_at": requested_at,
                "broker_exit_client_order_ids": (
                    *position.broker_exit_client_order_ids,
                    submission.runtime_order.client_order_id,
                ),
            }
        )
        self.state_store.append_event(
            self.runtime_session_id,
            requested_at,
            "EXIT_TRIGGERED",
            f"candidate={position.candidate_fingerprint};reason={reason.value}",
        )
        self.state_store.save_position(updated)
        return updated

    def _run_stream_connect(self) -> None:
        try:
            self.stream.connect()
        except Exception as error:
            with self._lock:
                active = self._session is not None and self._session.ended_at is None
            if active:
                self.on_stream_error(error)

    def _cancel_paper_pending(
        self, decisions: Sequence[AllocationDecision]
    ) -> None:
        for decision in decisions:
            fingerprint = candidate_fingerprint(decision.candidate)
            self.state_store.update_allocation_status(
                self.runtime_session_id, fingerprint, "CANCELLED"
            )
            if decision.outcome is AllocationOutcome.ALLOCATED:
                self._release_decision(decision)

    def _cancel_live_entry_remainders(self, requested_at: datetime) -> None:
        if self._live_gateway is None:
            raise RuntimeError("LIVE gateway is unavailable")
        terminal = {
            RuntimeOrderLifecycle.FILLED,
            RuntimeOrderLifecycle.CANCELLED,
            RuntimeOrderLifecycle.REJECTED,
        }
        entries = sorted(
            (
                order
                for order in self.state_store.list_orders(self.runtime_session_id)
                if order.leg is RuntimeOrderLeg.ENTRY
            ),
            key=lambda order: order.client_order_id,
        )
        for order in entries:
            if order.lifecycle in terminal:
                continue
            try:
                result = self._live_gateway.cancel_entry_order(order, requested_at)
                self._apply_live_reconciliation(result, requested_at)
            except Exception:
                if self.phase is not RuntimePhase.HALTED:
                    self.halt("entry remainder cancellation is ambiguous", requested_at)
                return

    def _unresolved_actual_execution_state(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self._actual_positions():
            issues.append("actual Runtime position remains open")
        if self.config.mode is RuntimeMode.LIVE:
            nonterminal = {
                RuntimeOrderLifecycle.INTENT_RECORDED,
                RuntimeOrderLifecycle.ACKNOWLEDGED,
                RuntimeOrderLifecycle.OPEN,
                RuntimeOrderLifecycle.PARTIALLY_FILLED,
                RuntimeOrderLifecycle.SUBMISSION_AMBIGUOUS,
                RuntimeOrderLifecycle.UNKNOWN,
            }
            orders = tuple(
                order
                for order in self.state_store.list_orders(self.runtime_session_id)
                if order.lifecycle in nonterminal
            )
            if orders:
                issues.append(
                    "nonterminal LIVE order(s): "
                    + ",".join(order.client_order_id for order in orders)
                )
        pending_actual = tuple(
            fingerprint
            for fingerprint, _, decision, status in self.state_store.load_allocations(
                self.runtime_session_id
            )
            if status == "PENDING"
            and decision.outcome is AllocationOutcome.ALLOCATED
        )
        if pending_actual:
            issues.append("pending actual allocation(s): " + ",".join(pending_actual))
        reservations = self.state_store.load_reservations(self.runtime_session_id)
        if reservations:
            issues.append(f"active actual reservation count={len(reservations)}")
        return tuple(issues)

    def _stabilize_live_entry(
        self, position: RuntimePositionRecord, requested_at: datetime
    ) -> RuntimePositionRecord:
        if self._live_gateway is None:
            raise RuntimeError("LIVE gateway is unavailable")
        if position.broker_entry_client_order_id is None:
            self.halt("live position lacks entry order provenance", requested_at)
            raise BrokerAmbiguousStateError("live entry order provenance is missing")
        entry_order = self.state_store.get_order(position.broker_entry_client_order_id)
        if entry_order.lifecycle not in {
            RuntimeOrderLifecycle.FILLED,
            RuntimeOrderLifecycle.CANCELLED,
            RuntimeOrderLifecycle.REJECTED,
        }:
            try:
                result = self._live_gateway.cancel_entry_order(entry_order, requested_at)
                self._apply_live_reconciliation(result, requested_at)
            except Exception as error:
                if self.phase is not RuntimePhase.HALTED:
                    self.halt("entry quantity could not be stabilized before exit", requested_at)
                raise BrokerAmbiguousStateError(
                    "entry quantity could not be stabilized before exit"
                ) from error
            entry_order = result.runtime_order
        if entry_order.lifecycle not in {
            RuntimeOrderLifecycle.FILLED,
            RuntimeOrderLifecycle.CANCELLED,
            RuntimeOrderLifecycle.REJECTED,
        }:
            self.halt("entry cancellation did not establish a terminal state", requested_at)
            raise BrokerAmbiguousStateError(
                "entry cancellation did not establish a terminal state"
            )
        stable = self._live_positions.get(position.candidate_fingerprint)
        if stable is None or stable.entry_fill.quantity <= 0:
            self.halt("no proven entry quantity is available to exit", requested_at)
            raise BrokerAmbiguousStateError("no proven entry quantity is available to exit")
        return stable

    def _exit_orders(self, fingerprint: str) -> tuple[RuntimeOrderRecord, ...]:
        return tuple(
            sorted(
                (
                    order
                    for order in self.state_store.list_orders(self.runtime_session_id)
                    if order.leg is RuntimeOrderLeg.EXIT
                    and order.candidate_fingerprint == fingerprint
                ),
                key=lambda order: order.attempt,
            )
        )

    def _next_exit_attempt(
        self, position: RuntimePositionRecord, requested_at: datetime
    ) -> int | None:
        if position.exit_filled_quantity >= position.entry_fill.quantity:
            return None
        orders = self._exit_orders(position.candidate_fingerprint)
        if not orders:
            return 1
        latest = orders[-1]
        if latest.lifecycle is RuntimeOrderLifecycle.UNKNOWN:
            self.halt("unknown broker exit state prevents another attempt", requested_at)
            return None
        if latest.lifecycle in {
            RuntimeOrderLifecycle.INTENT_RECORDED,
            RuntimeOrderLifecycle.ACKNOWLEDGED,
            RuntimeOrderLifecycle.OPEN,
            RuntimeOrderLifecycle.PARTIALLY_FILLED,
            RuntimeOrderLifecycle.SUBMISSION_AMBIGUOUS,
        }:
            return None
        if latest.lifecycle is RuntimeOrderLifecycle.FILLED:
            return None
        return latest.attempt + 1

    @staticmethod
    def _aggregate_position_exit_fills(
        fills: Sequence[BrokerTradeFill],
    ) -> Fill:
        selected = tuple(fills)
        if not selected:
            raise ValueError("at least one exit broker fill is required")
        first = selected[0]
        if any(
            fill.instrument != first.instrument
            or fill.transaction_action is not first.transaction_action
            for fill in selected[1:]
        ):
            raise ValueError("exit fills must share instrument and transaction action")
        quantity = sum(fill.quantity for fill in selected)
        weighted_price = sum(
            (fill.fill_price * fill.quantity for fill in selected), Decimal("0")
        ) / quantity
        return Fill(
            timestamp=max(fill.fill_timestamp for fill in selected),
            price=weighted_price,
            quantity=quantity,
            slippage_per_unit=Decimal("0"),
            is_simulated=False,
        )

    def _apply_live_reconciliation(
        self, result: LiveReconciliationResult, reconciled_at: datetime
    ) -> None:
        order = result.runtime_order
        plan, decision, allocation_status = self._allocation(
            order.candidate_fingerprint
        )
        if allocation_status in {"CLOSED", "CANCELLED"}:
            return
        if result.broker_order.state is BrokerOrderState.UNKNOWN:
            self.halt("unknown broker order status during reconciliation", reconciled_at)
            return
        if order.leg is RuntimeOrderLeg.ENTRY:
            if result.aggregate_fill is not None:
                existing = self._live_positions.get(order.candidate_fingerprint)
                if plan.protective_exit is not None:
                    try:
                        self._validate_live_protective_geometry(plan, result.aggregate_fill)
                    except (TypeError, ValueError):
                        self.halt(
                            "protective exit geometry is invalid for live entry fill",
                            reconciled_at,
                        )
                        raise
                position = RuntimePositionRecord(
                    runtime_session_id=self.runtime_session_id,
                    candidate=plan.candidate,
                    candidate_fingerprint=order.candidate_fingerprint,
                    allocation_decision=decision,
                    reservation=decision.reservation,
                    entry_fill=result.aggregate_fill,
                    protective_exit=plan.protective_exit,
                    dynamic_exit_policy=plan.dynamic_exit_policy,
                    dynamic_exit_state=(
                        existing.dynamic_exit_state
                        if existing
                        else initialize_runtime_dynamic_exit(
                            plan, result.aggregate_fill
                        )
                    ),
                    broker_entry_client_order_id=order.client_order_id,
                    mfe_return=(existing.mfe_return if existing else Decimal("0")),
                    mae_return=(existing.mae_return if existing else Decimal("0")),
                    exit_lifecycle=(
                        existing.exit_lifecycle
                        if existing
                        else RuntimeExitLifecycle.NONE
                    ),
                    requested_exit_reason=(
                        existing.requested_exit_reason if existing else None
                    ),
                    requested_exit_at=(existing.requested_exit_at if existing else None),
                    broker_exit_client_order_ids=(
                        existing.broker_exit_client_order_ids if existing else ()
                    ),
                    exit_filled_quantity=(
                        existing.exit_filled_quantity if existing else 0
                    ),
                )
                self._live_positions[order.candidate_fingerprint] = position
                self.state_store.open_position(position)
            elif result.broker_order.filled_quantity > 0:
                self.halt("broker reports fills but trade evidence is missing", reconciled_at)
            elif result.broker_order.state in {
                BrokerOrderState.REJECTED,
                BrokerOrderState.CANCELLED,
            }:
                self._release_decision(decision)
                self.state_store.update_allocation_status(
                    self.runtime_session_id, order.candidate_fingerprint, "CANCELLED"
                )
        else:
            position = self._live_positions.get(order.candidate_fingerprint)
            if position is None:
                if result.aggregate_fill is not None:
                    self.halt("exit fill has no persisted runtime position", reconciled_at)
                return
            exit_orders = self._exit_orders(order.candidate_fingerprint)
            all_fills = tuple(
                fill
                for exit_order in exit_orders
                for fill in self.state_store.list_broker_fills(exit_order.client_order_id)
            )
            all_fills = tuple(
                sorted(all_fills, key=lambda fill: (fill.fill_timestamp, fill.fill_id))
            )
            cumulative_quantity = sum(fill.quantity for fill in all_fills)
            if cumulative_quantity > position.entry_fill.quantity:
                self.halt("cumulative exit fills exceed entry quantity", reconciled_at)
                return
            latest_order = exit_orders[-1]
            if cumulative_quantity == position.entry_fill.quantity:
                lifecycle = RuntimeExitLifecycle.FILLED
            elif latest_order.lifecycle in {
                RuntimeOrderLifecycle.CANCELLED,
                RuntimeOrderLifecycle.REJECTED,
            }:
                lifecycle = RuntimeExitLifecycle.NONE
            elif latest_order.lifecycle in {
                RuntimeOrderLifecycle.SUBMISSION_AMBIGUOUS,
                RuntimeOrderLifecycle.UNKNOWN,
            }:
                lifecycle = RuntimeExitLifecycle.AMBIGUOUS
            elif cumulative_quantity:
                lifecycle = RuntimeExitLifecycle.PARTIALLY_FILLED
            else:
                lifecycle = RuntimeExitLifecycle.ACKNOWLEDGED
            updated = position.model_copy(
                update={
                    "exit_lifecycle": lifecycle,
                    "exit_filled_quantity": cumulative_quantity,
                    "broker_exit_client_order_ids": tuple(
                        exit_order.client_order_id for exit_order in exit_orders
                    ),
                }
            )
            self._live_positions[order.candidate_fingerprint] = updated
            self.state_store.save_position(updated)
            if lifecycle is RuntimeExitLifecycle.FILLED:
                exit_fill = self._aggregate_position_exit_fills(all_fills)
                reason = (
                    updated.requested_exit_reason
                    or order.exit_reason
                    or ExitReason.MANUAL
                )
                self._complete_trade(
                    updated,
                    ExitResult(
                        fill=exit_fill,
                        exit_reason=reason,
                        exit_reason_detail=runtime_exit_detail(position, reason),
                    ),
                    broker_exit_fill_ids=tuple(fill.fill_id for fill in all_fills),
                )

    def _reconcile_external_positions(self, reconciled_at: datetime) -> None:
        if self.broker is None:
            return
        actual: dict[str, int] = {}
        for position in self.broker.list_positions():
            if position.net_quantity:
                actual[position.instrument.symbol] = (
                    actual.get(position.instrument.symbol, 0) + position.net_quantity
                )
        expected: dict[str, int] = {}
        for position in self._live_positions.values():
            signal = position.candidate.order_intent.signal
            remaining_quantity = (
                position.entry_fill.quantity - position.exit_filled_quantity
            )
            signed = (
                remaining_quantity
                if signal.side is Side.LONG
                else -remaining_quantity
            )
            expected[signal.symbol] = expected.get(signal.symbol, 0) + signed
        if actual != expected:
            self.halt(
                f"unexpected external broker position; expected={expected};actual={actual}",
                reconciled_at,
            )

    def _reconcile_external_orders(self, reconciled_at: datetime) -> None:
        """Reject active Broker orders with no exact persisted Runtime identity."""
        if self.broker is None:
            return
        persisted = self.state_store.list_orders(self.runtime_session_id)
        known_order_ids = {order.broker_order_id for order in persisted if order.broker_order_id}
        known_tags = {order.broker_order_tag for order in persisted if order.broker_order_tag}
        active_states = {
            BrokerOrderState.SUBMITTED,
            BrokerOrderState.OPEN,
            BrokerOrderState.PENDING,
            BrokerOrderState.PARTIALLY_FILLED,
            BrokerOrderState.UNKNOWN,
        }
        unknown = [
            order
            for order in self.broker.list_orders()
            if order.state in active_states
            and order.broker_order_id not in known_order_ids
            and order.broker_order_tag not in known_tags
        ]
        if unknown:
            identities = ",".join(sorted(order.broker_order_id for order in unknown))
            self.halt(f"unexpected active broker order(s): {identities}", reconciled_at)

    def _complete_trade(
        self,
        position: RuntimePositionRecord,
        exit_result: ExitResult,
        *,
        broker_exit_fill_ids: tuple[str, ...] = (),
    ) -> RuntimeTradeRecord:
        signal = position.candidate.order_intent.signal
        quantity = position.entry_fill.quantity
        if signal.side is Side.LONG:
            gross_pnl = (exit_result.fill.price - position.entry_fill.price) * quantity
        else:
            gross_pnl = (position.entry_fill.price - exit_result.fill.price) * quantity
        policy = get_fixed_current_backtest_cost_policy(self.config.brokerage_plan)
        costs = calculate_round_trip_costs(
            side=signal.side,
            entry_fill=position.entry_fill,
            exit_fill=exit_result.fill,
            schedule=policy.schedule,
        )
        completed_signal = (
            position.allocation_decision.signal
            if position.is_shadow
            else signal.model_copy(update={"status": SignalStatus.EXECUTED})
        )
        trade = Trade(
            signal=completed_signal,
            ml_score=position.candidate.ml_score,
            target_notional=position.candidate.target_notional,
            entry_fill=position.entry_fill,
            exit_fill=exit_result.fill,
            gross_pnl=gross_pnl,
            total_costs=costs.total,
            net_pnl=gross_pnl - costs.total,
            mfe_return=position.mfe_return,
            mae_return=position.mae_return,
            exit_reason=exit_result.exit_reason,
            exit_reason_detail=exit_result.exit_reason_detail,
            is_shadow=position.is_shadow,
        )
        entry_fill_ids = ()
        if position.broker_entry_client_order_id:
            entry_fill_ids = tuple(
                fill.fill_id
                for fill in self.state_store.list_broker_fills(
                    position.broker_entry_client_order_id
                )
            )
        record = RuntimeTradeRecord(
            runtime_session_id=self.runtime_session_id,
            candidate_fingerprint=position.candidate_fingerprint,
            allocation_decision=position.allocation_decision,
            margin_quote=position.allocation_decision.margin_quote,
            trade=trade,
            cost_policy_id=policy.policy_id,
            broker_entry_client_order_id=position.broker_entry_client_order_id,
            broker_exit_client_order_ids=position.broker_exit_client_order_ids,
            broker_entry_fill_ids=entry_fill_ids,
            broker_exit_fill_ids=broker_exit_fill_ids,
        )
        terminal_position = position.model_copy(
            update={"exit_lifecycle": RuntimeExitLifecycle.FILLED}
        )
        if self._session is None:
            raise RuntimeError("Runtime session has not started")
        proposed_capital = self._economic_capital
        proposed_session = self._session
        proposed_portfolio_state = self._portfolio_state
        try:
            if not position.is_shadow:
                proposed_capital += trade.net_pnl
                proposed_session = self._session.model_copy(
                    update={"current_capital": proposed_capital}
                )
                released_state = self.capital_allocator.release(
                    self._portfolio_state, position.reservation
                )
                proposed_portfolio_state = released_state
                if proposed_capital > 0:
                    proposed_portfolio_state = PortfolioState(
                        capital_limit=proposed_capital,
                        active_reservations=released_state.active_reservations,
                    )
            self.state_store.finalize_completed_trade(
                terminal_position,
                record,
                proposed_session,
                exit_result.fill.timestamp,
            )
        except Exception as error:
            try:
                self.halt(
                    "completed-trade atomic finalization failed",
                    exit_result.fill.timestamp,
                )
            except Exception as halt_error:
                error.add_note(
                    "Runtime HALT persistence also failed: "
                    f"{type(halt_error).__name__}: {halt_error}"
                )
            raise

        if position.is_shadow or self.config.mode is RuntimeMode.PAPER:
            self._paper_gateway.publish_completed_position(
                position.candidate_fingerprint
            )
        else:
            self._live_positions.pop(position.candidate_fingerprint, None)
        if position.is_shadow:
            return record
        self._portfolio_state = proposed_portfolio_state
        self._economic_capital = proposed_capital
        self._session = proposed_session
        if proposed_capital <= 0:
            self.halt("modeled economic capital is exhausted", exit_result.fill.timestamp)
        return record

    def _release_decision(self, decision: AllocationDecision) -> None:
        if decision.reservation is None:
            return
        if any(
            active.identity == decision.reservation.identity
            for active in self._portfolio_state.active_reservations
        ):
            self._portfolio_state = self.capital_allocator.release(
                self._portfolio_state, decision.reservation
            )
        self.state_store.delete_reservation(
            self.runtime_session_id, candidate_fingerprint(decision.candidate)
        )

    def _allocation(
        self,
        fingerprint: str,
    ) -> tuple[RuntimeTradePlan, AllocationDecision, str]:
        matching = [
            (plan, decision, status)
            for stored_fingerprint, plan, decision, status in self.state_store.load_allocations(
                self.runtime_session_id
            )
            if stored_fingerprint == fingerprint
        ]
        if len(matching) != 1:
            raise BrokerAmbiguousStateError("runtime allocation provenance is missing or ambiguous")
        return matching[0]

    def _validate_live_protective_geometry(
        self,
        plan: RuntimeTradePlan,
        entry_fill: Fill,
    ) -> None:
        """Reuse the frozen simulator's exact side/entry protective validation."""
        if plan.protective_exit is None:
            return
        signal = plan.candidate.order_intent.signal
        empty = pl.DataFrame(
            schema={
                "timestamp": pl.Datetime("us", "Asia/Kolkata"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "symbol": pl.String,
            }
        )
        self._paper_gateway.simulator.fill_protective_exit(
            side=signal.side,
            symbol=signal.symbol,
            quantity=entry_fill.quantity,
            entry_fill=entry_fill,
            protective_exit=plan.protective_exit,
            candles=empty,
        )

    def _actual_positions(self) -> tuple[RuntimePositionRecord, ...]:
        return tuple(position for position in self.positions if not position.is_shadow)

    def _restore_execution_state(self) -> None:
        positions = self.state_store.load_positions(self.runtime_session_id)
        if self.config.mode is RuntimeMode.PAPER:
            for position in positions:
                self._paper_gateway.restore_position(position)
            for _, plan, decision, status in self.state_store.load_allocations(
                self.runtime_session_id
            ):
                if status != "PENDING":
                    continue
                if decision.outcome is AllocationOutcome.ALLOCATED:
                    self._paper_gateway.submit_allocated_entry(
                        plan, decision, self._now()
                    )
                else:
                    self._paper_gateway.submit_shadow(plan, decision)
        else:
            self._live_positions = {
                position.candidate_fingerprint: position for position in positions
            }

    def _validate_recovery_session(self, session: RuntimeSessionRecord) -> None:
        if session.trading_date != self.trading_date or session.mode is not self.config.mode:
            raise ValueError("persisted session date/mode does not match Runtime configuration")
        if session.ended_at is not None:
            raise ValueError("cannot restart a completed runtime session")
        if session.configuration_fingerprint != runtime_config_fingerprint(self.config):
            raise ValueError("persisted session configuration fingerprint does not match")

    def _session_at(self, wall_time: object) -> datetime:
        return datetime.combine(self.trading_date, wall_time, tzinfo=MARKET_TIMEZONE)

    def _now(self) -> datetime:
        return self._at(self.clock.now())

    def _at(self, value: datetime | None) -> datetime:
        selected = self.clock.now() if value is None else value
        if selected.tzinfo is None or selected.utcoffset() is None:
            raise ValueError("Runtime timestamps must be timezone-aware")
        return selected.astimezone(MARKET_TIMEZONE)
