"""Event-driven historical backtest orchestration over frozen architecture layers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import cast
from zoneinfo import ZoneInfo

import polars as pl

from algo_trader.backtest.exit_policies import (
    BacktestExitPolicyResolver,
    freeze_exit_policy_registry,
)
from algo_trader.backtest.models import (
    BacktestConfig,
    BacktestIntegrityError,
    BacktestRequestOutcome,
    BacktestRequestResult,
    BacktestRunResult,
    BacktestTradeRecord,
    BacktestTradeRequest,
)
from algo_trader.costs import (
    BacktestCostPolicy,
    calculate_round_trip_costs,
    get_fixed_current_backtest_cost_policy,
)
from algo_trader.data import ParquetMarketDataStore
from algo_trader.domain import ExitReason, Fill, Side, SignalStatus, Trade
from algo_trader.execution import ExitResult, HistoricalExecutionSimulator
from algo_trader.portfolio import (
    AllocationDecision,
    AllocationOutcome,
    CandidateIdentity,
    CapitalAllocator,
    CapitalReservation,
    MarginRequirementProvider,
    PortfolioState,
)

BACKTESTER_VERSION = "2"
MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")


@dataclass(frozen=True, slots=True)
class _RealizationEvent:
    timestamp: datetime
    identity: CandidateIdentity
    reservation: CapitalReservation
    net_pnl: Decimal


class HistoricalBacktester:
    """Coordinate data, allocation, execution, costs, and event realization."""

    def __init__(
        self,
        market_data_store: ParquetMarketDataStore,
        margin_provider: MarginRequirementProvider,
        execution_simulator: HistoricalExecutionSimulator | None = None,
        exit_policy_resolvers: Iterable[BacktestExitPolicyResolver] = (),
    ) -> None:
        if not isinstance(market_data_store, ParquetMarketDataStore):
            raise TypeError("market_data_store must be a ParquetMarketDataStore")
        if not isinstance(margin_provider, MarginRequirementProvider):
            raise TypeError("margin_provider must implement MarginRequirementProvider")
        if execution_simulator is not None and not isinstance(
            execution_simulator, HistoricalExecutionSimulator
        ):
            raise TypeError(
                "execution_simulator must be a HistoricalExecutionSimulator"
            )
        self.market_data_store = market_data_store
        self.margin_provider = margin_provider
        self.execution_simulator = (
            execution_simulator or HistoricalExecutionSimulator()
        )
        self.exit_policy_resolvers = freeze_exit_policy_registry(
            tuple(exit_policy_resolvers)
        )
        self.allocator = CapitalAllocator()

    def run(
        self,
        config: BacktestConfig,
        requests: Iterable[BacktestTradeRequest],
    ) -> BacktestRunResult:
        """Run a deterministic event-driven simulation over ``[start, end)``."""
        if not isinstance(config, BacktestConfig):
            raise TypeError("config must be a BacktestConfig")
        selected = tuple(requests)
        if any(not isinstance(request, BacktestTradeRequest) for request in selected):
            raise TypeError("all requests must be BacktestTradeRequest instances")
        self._validate_request_window(selected, config)
        self._validate_requested_exit_policies(selected)

        policy = get_fixed_current_backtest_cost_policy(config.brokerage_plan)
        ordered = tuple(
            sorted(
                selected,
                key=lambda request: (
                    request.candidate.order_intent.timestamp,
                    _identity_sort_key(request.candidate.identity),
                ),
            )
        )
        candle_cache: dict[tuple[str, date], pl.DataFrame] = {}
        state = PortfolioState(capital_limit=config.initial_capital)
        economic_capital = config.initial_capital
        capital_exhausted = False
        events: list[_RealizationEvent] = []
        actual_records: list[BacktestTradeRecord] = []
        shadow_records: list[BacktestTradeRecord] = []
        request_results: list[BacktestRequestResult] = []

        def process_events(*, before: datetime, inclusive: bool) -> None:
            nonlocal state, economic_capital, capital_exhausted
            due = sorted(
                (
                    event
                    for event in events
                    if event.timestamp < before
                    or (inclusive and event.timestamp == before)
                ),
                key=lambda event: (
                    event.timestamp,
                    _identity_sort_key(event.identity),
                ),
            )
            index = 0
            while index < len(due):
                event_time = due[index].timestamp
                group_end = index + 1
                while group_end < len(due) and due[group_end].timestamp == event_time:
                    group_end += 1
                group = due[index:group_end]
                group_pnl = sum(
                    (event.net_pnl for event in group),
                    start=Decimal("0"),
                )
                for event in group:
                    state = self.allocator.release(state, event.reservation)
                    events.remove(event)
                economic_capital += group_pnl
                if economic_capital <= 0:
                    capital_exhausted = True
                if not capital_exhausted:
                    state = PortfolioState(
                        capital_limit=economic_capital,
                        active_reservations=state.active_reservations,
                    )
                index = group_end

        index = 0
        while index < len(ordered):
            allocation_time = ordered[index].candidate.order_intent.timestamp
            batch_end = index + 1
            while (
                batch_end < len(ordered)
                and ordered[batch_end].candidate.order_intent.timestamp == allocation_time
            ):
                batch_end += 1
            batch = ordered[index:batch_end]

            process_events(before=allocation_time, inclusive=False)
            if capital_exhausted:
                request_results.extend(
                    BacktestRequestResult(
                        request=request,
                        outcome=BacktestRequestOutcome.CAPITAL_EXHAUSTED,
                        terminal_at=allocation_time,
                    )
                    for request in batch
                )
            else:
                allocation = self.allocator.allocate_batch(
                    [request.candidate for request in batch],
                    state,
                    self.margin_provider,
                )
                state = allocation.ending_state
                requests_by_identity = {
                    request.candidate.identity: request for request in batch
                }
                for decision in allocation.decisions:
                    request = requests_by_identity[decision.candidate.identity]
                    candles = self._daily_candles(
                        request,
                        config,
                        candle_cache,
                    )
                    terminal_result, event = self._simulate_decision(
                        request=request,
                        decision=decision,
                        candles=candles,
                        config=config,
                        policy=policy,
                    )
                    request_results.append(terminal_result)
                    if terminal_result.trade_record is not None:
                        if terminal_result.trade_record.trade.is_shadow:
                            shadow_records.append(terminal_result.trade_record)
                        else:
                            actual_records.append(terminal_result.trade_record)
                    if event is not None:
                        events.append(event)

            process_events(before=allocation_time, inclusive=True)
            index = batch_end

        for event_time in sorted({event.timestamp for event in events}):
            process_events(before=event_time, inclusive=True)

        ending_state = (
            PortfolioState(
                capital_limit=economic_capital,
                active_reservations=state.active_reservations,
            )
            if economic_capital > 0
            else None
        )
        return BacktestRunResult(
            run_id=config.run_id,
            git_commit=config.git_commit,
            backtester_version=BACKTESTER_VERSION,
            window_start=config.window_start,
            window_end=config.window_end,
            cost_policy_id=policy.policy_id,
            cost_policy_source_as_of_date=policy.source_as_of_date,
            brokerage_plan=config.brokerage_plan,
            starting_capital=config.initial_capital,
            ending_capital=economic_capital,
            capital_exhausted=capital_exhausted,
            actual_trade_records=tuple(sorted(actual_records, key=_record_sort_key)),
            shadow_trade_records=tuple(sorted(shadow_records, key=_record_sort_key)),
            request_results=tuple(sorted(request_results, key=_request_result_sort_key)),
            ending_portfolio_state=ending_state,
            symbols=tuple(
                sorted({request.candidate.order_intent.signal.symbol for request in selected})
            ),
            strategy_versions=tuple(
                sorted(
                    {
                        (
                            request.candidate.order_intent.signal.strategy_id,
                            request.candidate.order_intent.signal.strategy_version,
                        )
                        for request in selected
                    }
                )
            ),
            ml_model_versions=tuple(
                sorted({request.candidate.ml_score.model_version for request in selected})
            ),
        )

    def _simulate_decision(
        self,
        *,
        request: BacktestTradeRequest,
        decision: AllocationDecision,
        candles: pl.DataFrame,
        config: BacktestConfig,
        policy: BacktestCostPolicy,
    ) -> tuple[BacktestRequestResult, _RealizationEvent | None]:
        candidate = request.candidate
        cutoff = _market_datetime(
            candidate.order_intent.timestamp,
            config.forced_exit_time,
        )
        entry_deadline = min(
            cutoff,
            request.strategy_exit_at
            if request.strategy_exit_at is not None
            else cutoff,
        )
        entry_candles = candles.filter(pl.col("timestamp") < entry_deadline)
        entry_fill = self.execution_simulator.fill_entry_order(
            candidate.order_intent,
            entry_candles,
        )
        if entry_fill is None or entry_fill.timestamp >= entry_deadline:
            if decision.outcome is AllocationOutcome.ALLOCATED:
                reservation = cast(CapitalReservation, decision.reservation)
                return (
                    BacktestRequestResult(
                        request=request,
                        outcome=BacktestRequestOutcome.ALLOCATED_ENTRY_NOT_FILLED,
                        terminal_at=entry_deadline,
                        allocation_decision=decision,
                    ),
                    _RealizationEvent(
                        timestamp=entry_deadline,
                        identity=candidate.identity,
                        reservation=reservation,
                        net_pnl=Decimal("0"),
                    ),
                )
            return (
                BacktestRequestResult(
                    request=request,
                    outcome=BacktestRequestOutcome.SHADOW_ENTRY_NOT_FILLED,
                    terminal_at=entry_deadline,
                    allocation_decision=decision,
                ),
                None,
            )

        exit_result = self._resolve_exit(
            request=request,
            entry_fill=entry_fill,
            candles=candles,
            cutoff=cutoff,
        )
        record = self._build_trade_record(
            request=request,
            decision=decision,
            entry_fill=entry_fill,
            exit_result=exit_result,
            candles=candles,
            policy=policy,
        )
        if decision.outcome is AllocationOutcome.ALLOCATED:
            reservation = cast(CapitalReservation, decision.reservation)
            return (
                BacktestRequestResult(
                    request=request,
                    outcome=BacktestRequestOutcome.COMPLETED_ACTUAL,
                    terminal_at=exit_result.fill.timestamp,
                    allocation_decision=decision,
                    trade_record=record,
                ),
                _RealizationEvent(
                    timestamp=exit_result.fill.timestamp,
                    identity=candidate.identity,
                    reservation=reservation,
                    net_pnl=record.trade.net_pnl,
                ),
            )
        return (
            BacktestRequestResult(
                request=request,
                outcome=BacktestRequestOutcome.COMPLETED_SHADOW,
                terminal_at=exit_result.fill.timestamp,
                allocation_decision=decision,
                trade_record=record,
            ),
            None,
        )

    def _resolve_exit(
        self,
        *,
        request: BacktestTradeRequest,
        entry_fill: Fill,
        candles: pl.DataFrame,
        cutoff: datetime,
    ) -> ExitResult:
        candidate = request.candidate
        side = candidate.order_intent.signal.side
        symbol = candidate.order_intent.signal.symbol
        quantity = candidate.order_intent.quantity
        possibilities: list[tuple[datetime, int, ExitResult]] = []
        exit_candles = candles.filter(
            (pl.col("timestamp") >= entry_fill.timestamp)
            & (pl.col("timestamp") <= cutoff)
        )

        if request.dynamic_exit_policy is not None:
            resolver = self.exit_policy_resolvers[
                request.dynamic_exit_policy.policy_id
            ]
            resolved = resolver.resolve(
                request.dynamic_exit_policy,
                side=side,
                symbol=symbol,
                quantity=quantity,
                entry_fill=entry_fill,
                candles=exit_candles,
                execution_simulator=self.execution_simulator,
                strategy_exit_at=request.strategy_exit_at,
                forced_cutoff=cutoff,
            )
            if not isinstance(resolved, ExitResult):
                raise TypeError("exit policy resolver must return an ExitResult")
            if resolved.fill.quantity != entry_fill.quantity:
                raise BacktestIntegrityError(
                    "exit policy resolver returned a quantity different from the entry fill"
                )
            if not (entry_fill.timestamp <= resolved.fill.timestamp <= cutoff):
                raise BacktestIntegrityError(
                    "exit policy resolver returned a fill outside the open-position window"
                )
            return resolved

        if request.protective_exit is not None:
            protective = self.execution_simulator.fill_protective_exit(
                side=side,
                symbol=symbol,
                quantity=quantity,
                entry_fill=entry_fill,
                protective_exit=request.protective_exit,
                candles=exit_candles,
            )
            if protective is not None and protective.fill.timestamp <= cutoff:
                possibilities.append((protective.fill.timestamp, 0, protective))

        if (
            request.strategy_exit_at is not None
            and request.strategy_exit_at <= cutoff
        ):
            strategy = self.execution_simulator.fill_market_exit(
                side=side,
                symbol=symbol,
                quantity=quantity,
                requested_at=request.strategy_exit_at,
                exit_reason=ExitReason.STRATEGY_EXIT,
                candles=exit_candles,
            )
            if (
                strategy is not None
                and entry_fill.timestamp <= strategy.fill.timestamp <= cutoff
            ):
                possibilities.append((strategy.fill.timestamp, 1, strategy))

        if possibilities:
            return min(possibilities, key=lambda item: (item[0], item[1]))[2]

        time_exit = self.execution_simulator.fill_market_exit(
            side=side,
            symbol=symbol,
            quantity=quantity,
            requested_at=cutoff,
            exit_reason=ExitReason.TIME_EXIT,
            candles=exit_candles,
        )
        if time_exit is not None and time_exit.fill.timestamp == cutoff:
            return time_exit
        raise BacktestIntegrityError(
            f"no valid mandatory same-day exit exactly at cutoff or earlier; "
            f"cutoff is {cutoff.isoformat()} "
            f"for {symbol!r} after entry at {entry_fill.timestamp.isoformat()}"
        )

    @staticmethod
    def _build_trade_record(
        *,
        request: BacktestTradeRequest,
        decision: AllocationDecision,
        entry_fill: Fill,
        exit_result: ExitResult,
        candles: pl.DataFrame,
        policy: BacktestCostPolicy,
    ) -> BacktestTradeRecord:
        side = request.candidate.order_intent.signal.side
        quantity = request.candidate.order_intent.quantity
        gross_pnl = _gross_pnl(side, entry_fill, exit_result.fill, quantity)
        costs = calculate_round_trip_costs(
            side=side,
            entry_fill=entry_fill,
            exit_fill=exit_result.fill,
            schedule=policy.schedule,
        )
        mfe_return, mae_return = _excursion_returns(
            side=side,
            entry_fill=entry_fill,
            exit_fill=exit_result.fill,
            candles=candles,
        )
        signal_status = (
            SignalStatus.CAPACITY_REJECTED
            if decision.outcome is AllocationOutcome.CAPACITY_REJECTED
            else SignalStatus.EXECUTED
        )
        trade = Trade(
            signal=decision.signal.model_copy(update={"status": signal_status}),
            ml_score=request.candidate.ml_score,
            target_notional=request.candidate.target_notional,
            entry_fill=entry_fill,
            exit_fill=exit_result.fill,
            gross_pnl=gross_pnl,
            total_costs=costs.total,
            net_pnl=gross_pnl - costs.total,
            mfe_return=mfe_return,
            mae_return=mae_return,
            exit_reason=exit_result.exit_reason,
            is_shadow=decision.outcome is AllocationOutcome.CAPACITY_REJECTED,
        )
        return BacktestTradeRecord(
            trade=trade,
            round_trip_cost_breakdown=costs,
            cost_policy_id=policy.policy_id,
            allocation_identity=request.candidate.identity,
            allocation_decision=decision,
        )

    def _daily_candles(
        self,
        request: BacktestTradeRequest,
        config: BacktestConfig,
        cache: dict[tuple[str, date], pl.DataFrame],
    ) -> pl.DataFrame:
        order = request.candidate.order_intent
        symbol = order.signal.symbol
        trading_date = order.timestamp.astimezone(MARKET_TIMEZONE).date()
        key = (symbol, trading_date)
        if key not in cache:
            day_start = datetime.combine(trading_date, time.min, MARKET_TIMEZONE)
            day_end = day_start + timedelta(days=1)
            start = max(day_start, config.window_start.astimezone(MARKET_TIMEZONE))
            end = min(day_end, config.window_end.astimezone(MARKET_TIMEZONE))
            cache[key] = self.market_data_store.load_candles(symbol, start, end)
        return cache[key]

    def _validate_requested_exit_policies(
        self,
        requests: tuple[BacktestTradeRequest, ...],
    ) -> None:
        requested = sorted(
            {
                request.dynamic_exit_policy.policy_id
                for request in requests
                if request.dynamic_exit_policy is not None
            }
        )
        unknown = [
            policy_id
            for policy_id in requested
            if policy_id not in self.exit_policy_resolvers
        ]
        if unknown:
            raise ValueError(
                "unknown dynamic exit policy_id(s): " + ", ".join(unknown)
            )

    @staticmethod
    def _validate_request_window(
        requests: tuple[BacktestTradeRequest, ...],
        config: BacktestConfig,
    ) -> None:
        for request in requests:
            timestamp = request.candidate.order_intent.timestamp
            if not (config.window_start <= timestamp < config.window_end):
                raise ValueError(
                    "order_intent.timestamp must be inside the backtest window"
                )
            cutoff = _market_datetime(timestamp, config.forced_exit_time)
            if timestamp > cutoff:
                raise ValueError(
                    "order_intent.timestamp is later than the configured "
                    "forced-exit cutoff"
                )


def _market_datetime(value: datetime, local_time: time) -> datetime:
    trading_date = value.astimezone(MARKET_TIMEZONE).date()
    return datetime.combine(trading_date, local_time, MARKET_TIMEZONE)


def _gross_pnl(side: Side, entry: Fill, exit_fill: Fill, quantity: int) -> Decimal:
    if side is Side.LONG:
        return (exit_fill.price - entry.price) * quantity
    return (entry.price - exit_fill.price) * quantity


def _excursion_returns(
    *,
    side: Side,
    entry_fill: Fill,
    exit_fill: Fill,
    candles: pl.DataFrame,
) -> tuple[Decimal, Decimal]:
    eligible = candles.filter(
        (pl.col("timestamp") >= entry_fill.timestamp)
        & (pl.col("timestamp") < exit_fill.timestamp)
    )
    if eligible.is_empty():
        return Decimal("0"), Decimal("0")

    max_high = Decimal(str(eligible["high"].max()))
    min_low = Decimal(str(eligible["low"].min()))
    entry_price = entry_fill.price
    if side is Side.LONG:
        mfe = (max_high - entry_price) / entry_price
        mae = (min_low - entry_price) / entry_price
    else:
        mfe = (entry_price - min_low) / entry_price
        mae = (entry_price - max_high) / entry_price
    return max(mfe, Decimal("0")), min(mae, Decimal("0"))


def _identity_sort_key(identity: CandidateIdentity) -> tuple:
    return (
        identity[0],
        identity[1],
        identity[2],
        identity[3].value,
        identity[4],
        identity[5],
        identity[6].value,
        int(identity[7] is not None),
        identity[7] if identity[7] is not None else Decimal("0"),
        identity[8],
        identity[9],
    )


def _record_sort_key(record: BacktestTradeRecord) -> tuple:
    return (
        record.trade.exit_fill.timestamp,
        _identity_sort_key(record.allocation_identity),
    )


def _request_result_sort_key(result: BacktestRequestResult) -> tuple:
    return (
        result.terminal_at,
        _identity_sort_key(result.request.candidate.identity),
    )
