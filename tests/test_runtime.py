import copy
import os
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from threading import Condition, Event, Thread
from zoneinfo import ZoneInfo

import polars as pl
import pytest
from pydantic import SecretStr, ValidationError

from algo_trader.broker import (
    AngelOneCredentials,
    AngelOneSession,
    BrokerAmbiguousStateError,
    BrokerCancellationAcknowledgement,
    BrokerFunds,
    BrokerMarketTick,
    BrokerOrderAcknowledgement,
    BrokerOrderSnapshot,
    BrokerOrderState,
    BrokerPosition,
    BrokerQuote,
    BrokerTradeFill,
    BrokerTransactionAction,
    parse_instrument_master,
)
from algo_trader.costs import BrokeragePlan
from algo_trader.data import CANONICAL_CANDLE_COLUMNS
from algo_trader.domain import (
    ExitReason,
    MLScore,
    OrderIntent,
    OrderType,
    ProtectiveExitSpec,
    Side,
    Signal,
    SignalStatus,
)
from algo_trader.execution import FixedBasisPointsSlippage, HistoricalExecutionSimulator
from algo_trader.portfolio import (
    AllocationCandidate,
    AllocationOutcome,
    MarginRequirementQuote,
    PortfolioState,
)
from algo_trader.runtime import (
    DEFAULT_SMARTAPI_ENV_PATH,
    RUNTIME_ARCHITECTURE_VERSION,
    Clock,
    ExplicitTradingDayCalendar,
    FiveMinuteStrategyCycle,
    LiveExecutionGateway,
    PaperExecutionGateway,
    RuntimeConfig,
    RuntimeDynamicExitPolicy,
    RuntimeExecutionGateway,
    RuntimeExitLifecycle,
    RuntimeMode,
    RuntimeOrderLeg,
    RuntimeOrderLifecycle,
    RuntimeOrderRecord,
    RuntimePhase,
    RuntimeScheduler,
    RuntimeService,
    RuntimeSessionRecord,
    RuntimeSessionTimes,
    RuntimeStateStore,
    RuntimeTradePlan,
    SystemClock,
    TradingDayProvider,
    aggregate_broker_fills,
    candidate_fingerprint,
    compose_runtime_application,
    get_completed_five_minute_candles,
    load_smartapi_credentials,
    load_trading_day_calendar,
    point_bar_from_tick,
    run_smartapi_connectivity_check,
    runtime_client_order_id,
)

IST = ZoneInfo("Asia/Kolkata")
TRADING_DATE = date(2026, 8, 14)


def at(hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(2026, 8, 14, hour, minute, second, tzinfo=IST)


class FakeClock:
    def __init__(self, current: datetime) -> None:
        self.current = current

    def now(self) -> datetime:
        return self.current


class FixedMarginProvider:
    def __init__(self, required_margin: Decimal = Decimal("20000")) -> None:
        self.required_margin = required_margin
        self.calls = []

    def quote(self, candidate, state):
        self.calls.append((candidate, state))
        return MarginRequirementQuote(
            provider_id="TEST:MARGIN:V1",
            required_margin=self.required_margin,
        )


def master():
    rows = []
    for name, token in (("AAA", "101"), ("BBB", "102")):
        rows.append(
            {
                "token": token,
                "symbol": f"{name}-EQ",
                "name": name,
                "expiry": "",
                "strike": "-1",
                "lotsize": "1",
                "instrumenttype": "",
                "exch_seg": "NSE",
                "tick_size": "5",
            }
        )
    return parse_instrument_master(rows)


def candidate(
    *,
    symbol: str = "AAA",
    side: Side = Side.LONG,
    quality: float = 0.8,
    notional: int = 50_000,
    order_type: OrderType = OrderType.MARKET,
    limit_price: Decimal | None = None,
    minute: int = 16,
) -> AllocationCandidate:
    signal = Signal(
        strategy_id="runtime-test",
        strategy_version="1",
        symbol=symbol,
        timestamp=at(9, 15),
        side=side,
    )
    order = OrderIntent(
        signal=signal,
        timestamp=at(9, minute),
        quantity=10,
        requested_notional=notional,
        order_type=order_type,
        limit_price=limit_price,
    )
    score = MLScore(
        model_version="meta-v1",
        quality_score=quality,
        calibrated_probability=0.6,
        predicted_net_return=0.01,
        recommended_notional=notional,
    )
    return AllocationCandidate(order_intent=order, ml_score=score)


def plan(**kwargs) -> RuntimeTradePlan:
    protective = kwargs.pop(
        "protective_exit",
        ProtectiveExitSpec(stop_price=Decimal("90"), target_price=Decimal("110")),
    )
    return RuntimeTradePlan(candidate=candidate(**kwargs), protective_exit=protective)


def tick(
    price: str,
    *,
    symbol: str = "AAA",
    timestamp: datetime | None = None,
) -> BrokerMarketTick:
    return BrokerMarketTick(
        instrument=master().resolve(symbol),
        exchange_timestamp=timestamp or at(9, 16),
        last_traded_price=Decimal(price),
        cumulative_volume=100,
    )


def config(
    tmp_path: Path,
    *,
    mode: RuntimeMode = RuntimeMode.PAPER,
    capital: Decimal = Decimal("100000"),
    live_enabled: bool = False,
) -> RuntimeConfig:
    return RuntimeConfig(
        mode=mode,
        credential_path=tmp_path / "credentials.env",
        state_db_path=tmp_path / "runtime.duckdb",
        brokerage_plan=BrokeragePlan.PLUS,
        starting_capital=capital,
        paper_slippage_bps=Decimal("0"),
        live_order_submission_enabled=live_enabled,
    )


def service(
    tmp_path: Path,
    *,
    mode: RuntimeMode = RuntimeMode.PAPER,
    capital: Decimal = Decimal("100000"),
    margin: Decimal = Decimal("20000"),
    current: datetime | None = None,
    live_enabled: bool = False,
    broker=None,
    simulator=None,
    runtime_session_id: str = "session-1",
    state_store_type=RuntimeStateStore,
):
    selected_config = config(
        tmp_path, mode=mode, capital=capital, live_enabled=live_enabled
    )
    clock = FakeClock(current or at(9, 16))
    store = state_store_type(selected_config.state_db_path)
    runtime = RuntimeService(
        runtime_session_id=runtime_session_id,
        trading_date=TRADING_DATE,
        config=selected_config,
        clock=clock,
        trading_calendar=ExplicitTradingDayCalendar([TRADING_DATE]),
        state_store=store,
        margin_provider=FixedMarginProvider(margin),
        broker=broker,
        instrument_master=master() if mode is RuntimeMode.LIVE else None,
        simulator=simulator,
    )
    return runtime, store, clock


class FailingFinalizationStore(RuntimeStateStore):
    """Inject a deterministic failure after one finalization mutation."""

    failure_stage = ""

    def _fail_after(self, stage: str) -> None:
        if self.failure_stage == stage:
            raise OSError(f"injected finalization failure after {stage}")

    def _close_position_in_transaction(self, position) -> None:
        super()._close_position_in_transaction(position)
        self._fail_after("position")

    def _insert_trade_in_transaction(self, trade) -> None:
        super()._insert_trade_in_transaction(trade)
        self._fail_after("trade")

    def _close_allocation_in_transaction(self, position) -> None:
        super()._close_allocation_in_transaction(position)
        self._fail_after("allocation")

    def _delete_reservation_in_transaction(self, position) -> None:
        super()._delete_reservation_in_transaction(position)
        self._fail_after("reservation")

    def _update_session_capital_in_transaction(self, updated_session) -> None:
        super()._update_session_capital_in_transaction(updated_session)
        self._fail_after("capital")

    def _append_completion_events_in_transaction(
        self, position, trade, updated_session, occurred_at
    ) -> None:
        super()._append_completion_events_in_transaction(
            position, trade, updated_session, occurred_at
        )
        self._fail_after("events")


def test_credentials_use_exact_file_and_never_mutate_environment(tmp_path: Path) -> None:
    assert DEFAULT_SMARTAPI_ENV_PATH == Path(".secrets/SmartAPI.env")
    env_path = tmp_path / "SmartAPI.env"
    env_path.write_text(
        "SMARTAPI_API_KEY=api\n"
        "SMARTAPI_CLIENT_CODE=client\n"
        "SMARTAPI_MPIN=1234\n"
        "SMARTAPI_TOTP_SECRET=seed\n",
        encoding="utf-8",
    )
    before = dict(os.environ)
    loaded = load_smartapi_credentials(env_path)
    assert loaded.client_code == "client"
    assert loaded.api_key.get_secret_value() == "api"
    assert loaded.pin.get_secret_value() == "1234"
    assert loaded.totp_secret.get_secret_value() == "seed"
    assert isinstance(loaded.api_key, SecretStr)
    assert dict(os.environ) == before
    assert "SecretStr('**********')" in repr(loaded)


@pytest.mark.parametrize(
    "contents,match",
    [
        ("", "missing"),
        (
            "SMARTAPI_API_KEY=api\nSMARTAPI_CLIENT_CODE=client\n"
            "SMARTAPI_MPIN=1234\n",
            "missing",
        ),
        (
            "SMARTAPI_API_KEY=\nSMARTAPI_CLIENT_CODE=client\n"
            "SMARTAPI_MPIN=1234\nSMARTAPI_TOTP_SECRET=seed\n",
            "non-blank",
        ),
        (
            "SMARTAPI_API_KEY=api\nSMARTAPI_CLIENT_CODE=client\n"
            "SMARTAPI_MPIN=1234\nSMARTAPI_TOTP_SECRET=seed\nTYPO=x\n",
            "unexpected",
        ),
        (
            "SMARTAPI_API_KEY=api\nSMARTAPI_API_KEY=other\n"
            "SMARTAPI_CLIENT_CODE=client\nSMARTAPI_MPIN=1234\n"
            "SMARTAPI_TOTP_SECRET=seed\n",
            "duplicate",
        ),
    ],
)
def test_credentials_reject_missing_blank_unexpected_or_duplicate(
    tmp_path: Path, contents: str, match: str
) -> None:
    selected = tmp_path / "SmartAPI.env"
    selected.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_smartapi_credentials(selected)
    with pytest.raises(FileNotFoundError):
        load_smartapi_credentials(tmp_path / "missing.env")
    with pytest.raises(ValueError, match="not a file"):
        load_smartapi_credentials(tmp_path)


def test_clock_calendar_session_times_and_config_contract(tmp_path: Path) -> None:
    current = SystemClock().now()
    assert current.tzinfo == IST
    fixed = FakeClock(at(10))
    assert isinstance(fixed, Clock)
    assert fixed.now() == at(10)
    calendar = ExplicitTradingDayCalendar([TRADING_DATE])
    assert isinstance(calendar, TradingDayProvider)
    assert calendar.is_trading_day(TRADING_DATE)
    assert not calendar.is_trading_day(date(2026, 8, 15))
    times = RuntimeSessionTimes()
    assert [
        times.startup_time.isoformat(timespec="minutes"),
        times.market_open_time.isoformat(timespec="minutes"),
        times.entry_cutoff_time.isoformat(timespec="minutes"),
        times.square_off_time.isoformat(timespec="minutes"),
        times.market_close_time.isoformat(timespec="minutes"),
        times.shutdown_time.isoformat(timespec="minutes"),
    ] == ["08:45", "09:15", "15:10", "15:20", "15:30", "15:35"]
    with pytest.raises(ValidationError, match="strictly increasing"):
        RuntimeSessionTimes(entry_cutoff_time=times.market_open_time)
    selected = config(tmp_path, mode=RuntimeMode.LIVE)
    assert RUNTIME_ARCHITECTURE_VERSION == "3"
    assert selected.live_order_submission_enabled is False
    assert "secret" not in repr(selected).lower()
    with pytest.raises(ValidationError):
        config(tmp_path, capital=Decimal("0"))
    with pytest.raises(ValidationError):
        RuntimeConfig(
            mode="BACKTEST",
            state_db_path=tmp_path / "x",
            brokerage_plan=BrokeragePlan.PLUS,
            starting_capital=Decimal("1"),
            paper_slippage_bps=Decimal("5"),
        )
    for invalid_timeout in (True, 0, -1, float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            RuntimeConfig(
                mode=RuntimeMode.PAPER,
                state_db_path=tmp_path / "timeout.duckdb",
                brokerage_plan=BrokeragePlan.PLUS,
                starting_capital=Decimal("1"),
                paper_slippage_bps=Decimal("5"),
                stream_shutdown_timeout_seconds=invalid_timeout,
            )
    assert selected.model_dump()["stream_shutdown_timeout_seconds"] == 10.0
    with pytest.raises(ValidationError, match="paper_slippage_bps"):
        RuntimeConfig(
            mode=RuntimeMode.PAPER,
            state_db_path=tmp_path / "missing-slippage.duckdb",
            brokerage_plan=BrokeragePlan.PLUS,
            starting_capital=Decimal("100000"),
        )
    assert selected.paper_slippage_bps == 0

    calendar_path = tmp_path / "calendar.txt"
    with pytest.raises(FileNotFoundError, match="required"):
        load_trading_day_calendar(calendar_path)
    calendar_path.write_text("# verified NSE dates\n2026-08-14\n", encoding="utf-8")
    assert load_trading_day_calendar(calendar_path).trading_dates == frozenset(
        {TRADING_DATE}
    )


def test_deterministic_runtime_identity_uses_session_candidate_and_leg() -> None:
    selected = candidate()
    entry = runtime_client_order_id("session-1", selected.identity, RuntimeOrderLeg.ENTRY)
    assert entry == runtime_client_order_id(
        "session-1", selected.identity, RuntimeOrderLeg.ENTRY
    )
    assert entry != runtime_client_order_id(
        "session-1", selected.identity, RuntimeOrderLeg.EXIT
    )
    assert entry != runtime_client_order_id(
        "session-2", selected.identity, RuntimeOrderLeg.ENTRY
    )
    reconstructed = AllocationCandidate.model_validate(selected.model_dump())
    assert candidate_fingerprint(reconstructed) == candidate_fingerprint(selected)


def test_state_store_persists_session_events_order_reservation_and_reopens(
    tmp_path: Path,
) -> None:
    db = tmp_path / "runtime.duckdb"
    store = RuntimeStateStore(db)
    session = RuntimeSessionRecord(
        runtime_session_id="state-session",
        trading_date=TRADING_DATE,
        mode=RuntimeMode.PAPER,
        starting_capital=Decimal("100000"),
        current_capital=Decimal("100000"),
        started_at=at(8, 45),
        live_order_submission_enabled=False,
        configuration_fingerprint="fingerprint",
    )
    store.create_session(session)
    store.append_event("state-session", at(9), "CHECK", "safe")
    events = store.list_events("state-session")
    assert [event.sequence for event in events] == [1, 2]
    assert events[0].event_type == "SESSION_STARTED"
    with pytest.raises(ValueError, match="duplicate active"):
        store.create_session(session)

    selected_plan = plan()
    margin = FixedMarginProvider().quote(selected_plan.candidate, PortfolioState())
    from algo_trader.portfolio import AllocationDecision, CapitalReservation

    reservation = CapitalReservation(candidate=selected_plan.candidate, margin_quote=margin)
    decision = AllocationDecision(
        candidate=selected_plan.candidate,
        outcome=AllocationOutcome.ALLOCATED,
        margin_quote=margin,
        signal=selected_plan.candidate.order_intent.signal,
        reservation=reservation,
    )
    fingerprint = candidate_fingerprint(selected_plan.candidate)
    store.save_allocation("state-session", fingerprint, selected_plan, decision)
    store.save_reservation("state-session", fingerprint, reservation)
    order = RuntimeOrderRecord(
        runtime_session_id="state-session",
        client_order_id="client-order",
        candidate_fingerprint=fingerprint,
        leg=RuntimeOrderLeg.ENTRY,
        symbol="AAA",
        quantity=10,
        transaction_action="BUY",
        order_type="MARKET",
        intended_at=at(9, 16),
    )
    store.record_order_intent(order)
    assert store.get_order("client-order").lifecycle is RuntimeOrderLifecycle.INTENT_RECORDED
    assert store.load_reservations("state-session") == (reservation,)
    store.close()
    reopened = RuntimeStateStore(db)
    assert reopened.get_session("state-session") == session
    assert reopened.get_order("client-order") == order
    assert reopened.load_allocations("state-session")[0][1] == selected_plan
    reopened.close()
    database_bytes = db.read_bytes()
    for secret in (b"api-secret", b"totp-secret", b"jwt-secret"):
        assert secret not in database_bytes


def test_point_bar_and_broker_fill_aggregation_are_exact() -> None:
    selected_tick = tick("100.25")
    frame = point_bar_from_tick(selected_tick)
    assert frame.columns == list(CANONICAL_CANDLE_COLUMNS)
    assert frame.row(0, named=True) == {
        "timestamp": at(9, 16),
        "open": 100.25,
        "high": 100.25,
        "low": 100.25,
        "close": 100.25,
        "volume": 100.0,
        "symbol": "AAA",
    }
    instrument = master().resolve("AAA")
    fills = (
        BrokerTradeFill(
            broker_order_id="B1",
            fill_id="F1",
            instrument=instrument,
            transaction_action=BrokerTransactionAction.BUY,
            fill_timestamp=at(9, 16),
            fill_price=Decimal("100"),
            quantity=4,
        ),
        BrokerTradeFill(
            broker_order_id="B1",
            fill_id="F2",
            instrument=instrument,
            transaction_action=BrokerTransactionAction.BUY,
            fill_timestamp=at(9, 17),
            fill_price=Decimal("110"),
            quantity=6,
        ),
    )
    aggregate = aggregate_broker_fills(fills)
    assert aggregate.quantity == 10
    assert aggregate.price == Decimal("106")
    assert aggregate.timestamp == at(9, 17)
    assert aggregate.is_simulated is False
    assert aggregate.slippage_per_unit == 0


def test_execution_gateways_satisfy_public_structural_contract(tmp_path: Path) -> None:
    store = RuntimeStateStore(tmp_path / "gateways.duckdb")
    paper = PaperExecutionGateway("session", store)
    live = LiveExecutionGateway(
        "session",
        FakeBroker(store),
        master(),
        store,
        live_order_submission_enabled=True,
        halt_callback=lambda reason, occurred_at: None,
    )
    assert isinstance(paper, RuntimeExecutionGateway)
    assert isinstance(live, RuntimeExecutionGateway)
    store.close()


def test_paper_service_allocates_fills_protects_costs_and_updates_capital(
    tmp_path: Path,
) -> None:
    runtime, store, _ = service(
        tmp_path,
        simulator=HistoricalExecutionSimulator(
            slippage_model=FixedBasisPointsSlippage(Decimal("10"))
        ),
    )
    assert runtime.start() is RuntimePhase.TRADING
    before = runtime.economic_capital
    result = runtime.process_plans((plan(),), decision_at=at(9, 16))
    assert result.decisions[0].outcome is AllocationOutcome.ALLOCATED
    assert runtime.portfolio_state.reserved_margin == Decimal("20000")
    assert runtime.on_market_tick(tick("100", timestamp=at(9, 16))) == ()
    position = runtime.positions[0]
    assert position.entry_fill.is_simulated
    assert position.entry_fill.price == Decimal("100.1")
    completed = runtime.on_market_tick(tick("111", timestamp=at(9, 17)))
    assert len(completed) == 1
    trade = completed[0].trade
    assert trade.signal.status is SignalStatus.EXECUTED
    assert trade.exit_reason is ExitReason.TARGET_REACHED
    assert trade.is_shadow is False
    assert trade.total_costs > 0
    assert trade.mfe_return > 0
    assert trade.mae_return == 0
    assert runtime.economic_capital == before + trade.net_pnl
    assert runtime.portfolio_state.active_reservations == ()
    assert store.load_trades("session-1") == completed
    store.close()


def test_runtime_dynamic_trailing_state_is_completed_bar_causal_and_persisted(
    tmp_path: Path,
) -> None:
    runtime, store, clock = service(tmp_path)
    runtime.start()
    dynamic_plan = RuntimeTradePlan(
        candidate=candidate(),
        dynamic_exit_policy=RuntimeDynamicExitPolicy(
            initial_stop_price=Decimal("90"),
            hard_target_r=Decimal("1.25"),
            breakeven_trigger_r=Decimal("0.75"),
            breakeven_stop_r=Decimal("0"),
            profit_lock_trigger_r=Decimal("1"),
            profit_lock_stop_r=Decimal("0.25"),
            trailing_distance_r=Decimal("0.5"),
        ),
    )
    runtime.process_plans((dynamic_plan,), decision_at=at(9, 16))
    runtime.on_market_tick(tick("100", timestamp=at(9, 16)))
    assert runtime.positions[0].dynamic_exit_state.current_stop == Decimal("90")
    completed = pl.DataFrame(
        {
            "timestamp": [at(9, 20)],
            "open": [100.0],
            "high": [108.0],
            "low": [99.0],
            "close": [107.0],
            "volume": [100.0],
            "symbol": ["AAA"],
        }
    )
    runtime.advance_dynamic_exits({"AAA": completed}, at(9, 25))
    assert runtime.positions[0].dynamic_exit_state.current_stop == Decimal("100")
    runtime.advance_dynamic_exits({"AAA": completed}, at(9, 25))
    assert runtime.positions[0].dynamic_exit_state.current_stop == Decimal("100")
    store.close()

    reopened = RuntimeStateStore(config(tmp_path).state_db_path)
    recovered = RuntimeService(
        runtime_session_id="session-1",
        trading_date=TRADING_DATE,
        config=config(tmp_path),
        clock=clock,
        trading_calendar=ExplicitTradingDayCalendar([TRADING_DATE]),
        state_store=reopened,
        margin_provider=FixedMarginProvider(),
    )
    recovered.start()
    assert recovered.positions[0].dynamic_exit_state.current_stop == Decimal("100")
    reopened.close()


def test_live_shadow_dynamic_trailing_advances_from_completed_bars(tmp_path: Path) -> None:
    broker = FakeBroker()
    runtime, store, _ = service(
        tmp_path,
        mode=RuntimeMode.LIVE,
        capital=Decimal("50000"),
        margin=Decimal("40000"),
        live_enabled=True,
        broker=broker,
    )
    broker.state_store = store
    runtime.start()
    policy = RuntimeDynamicExitPolicy(
        initial_stop_price=Decimal("90"),
        hard_target_r=Decimal("1.25"),
        breakeven_trigger_r=Decimal("0.75"),
        breakeven_stop_r=Decimal("0"),
        profit_lock_trigger_r=Decimal("1"),
        profit_lock_stop_r=Decimal("0.25"),
        trailing_distance_r=Decimal("0.5"),
    )
    actual = RuntimeTradePlan(candidate=candidate(symbol="AAA"), dynamic_exit_policy=policy)
    shadow = RuntimeTradePlan(candidate=candidate(symbol="BBB"), dynamic_exit_policy=policy)
    runtime.process_plans((actual, shadow), decision_at=at(9, 16))
    runtime.on_market_tick(tick("100", symbol="BBB", timestamp=at(9, 16)))

    runtime.advance_dynamic_exits(
        {
            "BBB": pl.DataFrame(
                {
                    "timestamp": [at(9, 20)],
                    "open": [100.0],
                    "high": [108.0],
                    "low": [99.0],
                    "close": [107.0],
                    "volume": [100.0],
                    "symbol": ["BBB"],
                }
            )
        },
        at(9, 25),
    )

    persisted = store.load_positions("session-1")
    assert len(persisted) == 1
    assert persisted[0].is_shadow
    assert persisted[0].dynamic_exit_state.current_stop == Decimal("100")
    store.close()


def test_completed_actual_trade_is_atomically_durable_across_restart(
    tmp_path: Path,
) -> None:
    runtime, store, clock = service(tmp_path)
    runtime.start()
    selected = plan()
    runtime.process_plans((selected,), decision_at=at(9, 16))
    runtime.on_market_tick(tick("100", timestamp=at(9, 16)))
    completed = runtime.on_market_tick(tick("111", timestamp=at(9, 17)))
    expected_capital = Decimal("100000") + completed[0].trade.net_pnl
    fingerprint = candidate_fingerprint(selected.candidate)

    assert store.load_positions("session-1") == ()
    assert store.load_trades("session-1") == completed
    assert store.load_allocations("session-1")[0][3] == "CLOSED"
    assert store.load_reservations("session-1") == ()
    assert store.get_session("session-1").current_capital == expected_capital
    event_types = [event.event_type for event in store.list_events("session-1")]
    assert event_types.count("TRADE_CLOSED") == 1
    assert event_types.count("CAPITAL_UPDATED") == 1
    store.close()

    reopened = RuntimeStateStore(config(tmp_path).state_db_path)
    assert reopened.load_positions("session-1") == ()
    assert len(reopened.load_trades("session-1")) == 1
    assert reopened.load_allocations("session-1")[0][3] == "CLOSED"
    assert reopened.load_reservations("session-1") == ()
    assert reopened.get_session("session-1").current_capital == expected_capital
    recovered = RuntimeService(
        runtime_session_id="session-1",
        trading_date=TRADING_DATE,
        config=config(tmp_path),
        clock=clock,
        trading_calendar=ExplicitTradingDayCalendar([TRADING_DATE]),
        state_store=reopened,
        margin_provider=FixedMarginProvider(),
    )
    assert recovered.start() is RuntimePhase.TRADING
    assert recovered.positions == ()
    assert recovered.portfolio_state.active_reservations == ()
    assert recovered.economic_capital == expected_capital
    assert candidate_fingerprint(selected.candidate) == fingerprint
    reopened.close()


def test_paper_exit_evidence_does_not_close_position_before_service_commit(
    tmp_path: Path,
) -> None:
    runtime, store, _ = service(tmp_path)
    runtime.start()
    runtime.process_plans((plan(),), decision_at=at(9, 16))
    gateway = runtime._paper_gateway
    opened = gateway.on_market_tick(tick("100", timestamp=at(9, 16)))
    assert len(opened.opened) == 1

    detected = gateway.on_market_tick(tick("111", timestamp=at(9, 17)))
    assert len(detected.closed) == 1
    assert detected.closed[0][0].exit_lifecycle is RuntimeExitLifecycle.FILLED
    assert len(gateway.positions) == 1
    assert len(store.load_positions("session-1")) == 1
    assert store.load_trades("session-1") == ()
    assert store.load_allocations("session-1")[0][3] == "OPEN"
    assert len(store.load_reservations("session-1")) == 1
    store.close()


@pytest.mark.parametrize(
    "failure_stage",
    ["position", "trade", "allocation", "reservation", "capital", "events"],
)
def test_completed_trade_finalization_rolls_back_every_durable_effect(
    tmp_path: Path,
    failure_stage: str,
) -> None:
    class SelectedFailingStore(FailingFinalizationStore):
        pass

    SelectedFailingStore.failure_stage = failure_stage
    runtime, store, _ = service(tmp_path, state_store_type=SelectedFailingStore)
    runtime.start()
    selected = plan()
    runtime.process_plans((selected,), decision_at=at(9, 16))
    runtime.on_market_tick(tick("100", timestamp=at(9, 16)))
    before_capital = runtime.economic_capital

    with pytest.raises(OSError, match=f"after {failure_stage}"):
        runtime.on_market_tick(tick("111", timestamp=at(9, 17)))

    assert runtime.phase is RuntimePhase.HALTED
    assert runtime.economic_capital == before_capital
    assert len(runtime.positions) == 1
    assert len(runtime.portfolio_state.active_reservations) == 1
    store.close()

    reopened = RuntimeStateStore(config(tmp_path).state_db_path)
    assert len(reopened.load_positions("session-1")) == 1
    assert reopened.load_trades("session-1") == ()
    assert reopened.load_allocations("session-1")[0][3] == "OPEN"
    assert len(reopened.load_reservations("session-1")) == 1
    assert reopened.get_session("session-1").current_capital == before_capital
    event_types = [event.event_type for event in reopened.list_events("session-1")]
    assert "TRADE_CLOSED" not in event_types
    assert "CAPITAL_UPDATED" not in event_types
    assert event_types[-1] == "SESSION_HALTED"
    assert [event.sequence for event in reopened.list_events("session-1")] == list(
        range(1, len(event_types) + 1)
    )
    reopened.close()


def test_failed_finalization_keeps_runtime_memory_precommit_and_rejects_duplicate(
    tmp_path: Path,
) -> None:
    class TradeFailingStore(FailingFinalizationStore):
        failure_stage = "trade"

    runtime, store, _ = service(tmp_path, state_store_type=TradeFailingStore)
    runtime.start()
    selected = plan()
    runtime.process_plans((selected,), decision_at=at(9, 16))
    runtime.on_market_tick(tick("100", timestamp=at(9, 16)))
    open_position = runtime.positions[0]
    before_state = runtime.portfolio_state
    before_capital = runtime.economic_capital

    with pytest.raises(OSError, match="after trade"):
        runtime.on_market_tick(tick("111", timestamp=at(9, 17)))

    assert runtime.phase is RuntimePhase.HALTED
    assert runtime.economic_capital == before_capital
    assert runtime.portfolio_state == before_state
    assert runtime.positions == (open_position,)
    assert store.load_trades("session-1") == ()
    store.close()

    successful, successful_store, _ = service(tmp_path / "duplicate")
    successful.start()
    successful.process_plans((selected,), decision_at=at(9, 16))
    successful.on_market_tick(tick("100", timestamp=at(9, 16)))
    persisted_position = successful.positions[0]
    completed = successful.on_market_tick(tick("111", timestamp=at(9, 17)))[0]
    with pytest.raises(ValueError, match="already has a completed trade"):
        successful_store.finalize_completed_trade(
            persisted_position.model_copy(
                update={"exit_lifecycle": RuntimeExitLifecycle.FILLED}
            ),
            completed,
            successful_store.get_session("session-1"),
            at(9, 17),
        )
    assert len(successful_store.load_trades("session-1")) == 1
    successful_store.close()


def test_paper_limit_strategy_exit_and_pending_square_off_release(tmp_path: Path) -> None:
    runtime, store, _ = service(tmp_path)
    runtime.start()
    limit_plan = plan(
        order_type=OrderType.LIMIT,
        limit_price=Decimal("99"),
        protective_exit=None,
    )
    runtime.process_plans((limit_plan,), decision_at=at(9, 16))
    runtime.on_market_tick(tick("100", timestamp=at(9, 16)))
    assert runtime.positions == ()
    runtime.on_market_tick(tick("99", timestamp=at(9, 17)))
    fingerprint = candidate_fingerprint(limit_plan.candidate)
    assert runtime.request_strategy_exit(fingerprint, at(9, 18))
    assert not runtime.request_strategy_exit(fingerprint, at(9, 18))
    completed = runtime.on_market_tick(tick("101", timestamp=at(9, 18)))
    assert completed[0].trade.exit_reason is ExitReason.STRATEGY_EXIT

    another = plan(symbol="BBB", minute=19)
    runtime.process_plans((another,), decision_at=at(9, 19))
    assert runtime.portfolio_state.active_reservations
    runtime.force_square_off(at(15, 20))
    assert runtime.phase is RuntimePhase.SQUARE_OFF
    assert runtime.portfolio_state.active_reservations == ()
    store.close()


def test_capacity_shadow_is_separate_and_never_changes_capital(tmp_path: Path) -> None:
    runtime, store, _ = service(tmp_path, capital=Decimal("50000"), margin=Decimal("40000"))
    runtime.start()
    first = plan(symbol="AAA", quality=0.9)
    second = plan(symbol="BBB", quality=0.2)
    result = runtime.process_plans((second, first), decision_at=at(9, 16))
    assert [decision.outcome for decision in result.decisions] == [
        AllocationOutcome.ALLOCATED,
        AllocationOutcome.CAPACITY_REJECTED,
    ]
    starting = runtime.economic_capital
    runtime.on_market_tick(tick("100", symbol="BBB", timestamp=at(9, 16)))
    shadow = next(position for position in runtime.positions if position.is_shadow)
    assert shadow.allocation_decision.signal.status is SignalStatus.CAPACITY_REJECTED
    completed = runtime.on_market_tick(tick("111", symbol="BBB", timestamp=at(9, 17)))
    assert completed[0].trade.is_shadow
    assert completed[0].trade.signal.status is SignalStatus.CAPACITY_REJECTED
    assert completed[0].trade.total_costs > 0
    assert runtime.economic_capital == starting
    assert len(runtime.portfolio_state.active_reservations) == 1
    store.close()


def test_shadow_completion_is_atomic_and_capital_neutral_across_restart(
    tmp_path: Path,
) -> None:
    runtime, store, clock = service(
        tmp_path,
        capital=Decimal("50000"),
        margin=Decimal("60000"),
    )
    runtime.start()
    selected = plan()
    decision = runtime.process_plans((selected,), decision_at=at(9, 16)).decisions[0]
    assert decision.outcome is AllocationOutcome.CAPACITY_REJECTED
    runtime.on_market_tick(tick("100", timestamp=at(9, 16)))
    shadow_position = runtime.positions[0]
    completed = runtime.on_market_tick(tick("111", timestamp=at(9, 17)))

    assert completed[0].trade.is_shadow
    assert runtime.economic_capital == Decimal("50000")
    assert runtime.portfolio_state.active_reservations == ()
    assert store.load_positions("session-1") == ()
    assert store.load_trades("session-1") == completed
    assert store.load_allocations("session-1")[0][3] == "CLOSED"
    assert store.load_reservations("session-1") == ()
    assert store.get_session("session-1").current_capital == Decimal("50000")
    event_types = [event.event_type for event in store.list_events("session-1")]
    assert event_types.count("TRADE_CLOSED") == 1
    assert "CAPITAL_UPDATED" not in event_types

    with pytest.raises(ValueError, match="already has a completed trade"):
        store.finalize_completed_trade(
            shadow_position.model_copy(
                update={"exit_lifecycle": RuntimeExitLifecycle.FILLED}
            ),
            completed[0],
            store.get_session("session-1"),
            at(9, 17),
        )
    assert len(store.load_trades("session-1")) == 1
    assert [event.event_type for event in store.list_events("session-1")].count(
        "TRADE_CLOSED"
    ) == 1
    store.close()

    reopened = RuntimeStateStore(config(tmp_path, capital=Decimal("50000")).state_db_path)
    assert reopened.load_positions("session-1") == ()
    assert len(reopened.load_trades("session-1")) == 1
    assert reopened.load_allocations("session-1")[0][3] == "CLOSED"
    assert reopened.load_reservations("session-1") == ()
    assert reopened.get_session("session-1").current_capital == Decimal("50000")
    recovered = RuntimeService(
        runtime_session_id="session-1",
        trading_date=TRADING_DATE,
        config=config(tmp_path, capital=Decimal("50000")),
        clock=clock,
        trading_calendar=ExplicitTradingDayCalendar([TRADING_DATE]),
        state_store=reopened,
        margin_provider=FixedMarginProvider(Decimal("60000")),
    )
    assert recovered.start() is RuntimePhase.TRADING
    assert recovered.positions == ()
    assert recovered.economic_capital == Decimal("50000")
    assert recovered.portfolio_state.active_reservations == ()
    reopened.close()


@pytest.mark.parametrize(
    "failure_stage",
    ["position", "trade", "allocation", "events"],
)
def test_shadow_finalization_failure_rolls_back_storage_and_memory(
    tmp_path: Path,
    failure_stage: str,
) -> None:
    class SelectedFailingStore(FailingFinalizationStore):
        pass

    SelectedFailingStore.failure_stage = failure_stage
    runtime, store, _ = service(
        tmp_path,
        capital=Decimal("50000"),
        margin=Decimal("60000"),
        state_store_type=SelectedFailingStore,
    )
    runtime.start()
    selected = plan()
    decision = runtime.process_plans((selected,), decision_at=at(9, 16)).decisions[0]
    assert decision.outcome is AllocationOutcome.CAPACITY_REJECTED
    runtime.on_market_tick(tick("100", timestamp=at(9, 16)))
    shadow_position = runtime.positions[0]
    before_state = runtime.portfolio_state
    before_capital = runtime.economic_capital

    with pytest.raises(OSError, match=f"after {failure_stage}"):
        runtime.on_market_tick(tick("111", timestamp=at(9, 17)))

    assert runtime.phase is RuntimePhase.HALTED
    assert runtime.positions == (shadow_position,)
    assert runtime.portfolio_state == before_state
    assert runtime.economic_capital == before_capital
    store.close()

    reopened = RuntimeStateStore(config(tmp_path, capital=Decimal("50000")).state_db_path)
    assert len(reopened.load_positions("session-1")) == 1
    assert reopened.load_positions("session-1")[0].is_shadow
    assert reopened.load_trades("session-1") == ()
    assert reopened.load_allocations("session-1")[0][3] == "OPEN"
    assert reopened.load_reservations("session-1") == ()
    assert reopened.get_session("session-1").current_capital == before_capital
    event_types = [event.event_type for event in reopened.list_events("session-1")]
    assert "TRADE_CLOSED" not in event_types
    assert "CAPITAL_UPDATED" not in event_types
    assert event_types[-1] == "SESSION_HALTED"
    reopened.close()


def test_short_excursions_and_forced_time_exit_have_correct_signs(tmp_path: Path) -> None:
    runtime, store, _ = service(tmp_path)
    runtime.start()
    selected = plan(
        side=Side.SHORT,
        protective_exit=ProtectiveExitSpec(
            stop_price=Decimal("110"), target_price=Decimal("90")
        ),
    )
    runtime.process_plans((selected,), decision_at=at(9, 16))
    runtime.on_market_tick(tick("100", timestamp=at(9, 16)))
    runtime.on_market_tick(tick("95", timestamp=at(9, 17)))
    runtime.on_market_tick(tick("105", timestamp=at(9, 18)))
    position = runtime.positions[0]
    assert position.mfe_return == Decimal("0.05")
    assert position.mae_return == Decimal("-0.05")
    runtime.force_square_off(at(15, 20))
    completed = runtime.on_market_tick(tick("100", timestamp=at(15, 20)))
    assert completed[0].trade.exit_reason is ExitReason.TIME_EXIT
    assert completed[0].trade.mfe_return >= 0
    assert completed[0].trade.mae_return <= 0
    store.close()


class FakeBroker:
    def __init__(self, state_store: RuntimeStateStore | None = None) -> None:
        self.state_store = state_store
        self.calls = []
        self.order_snapshots: dict[str, BrokerOrderSnapshot] = {}
        self.fills: tuple[BrokerTradeFill, ...] = ()
        self.positions: tuple[BrokerPosition, ...] = ()
        self.fail_place = False
        self.fail_cancel = False
        self.place_count = 0
        self.before_place = None

    def place_order(self, request, acknowledged_at):
        self.place_count += 1
        self.calls.append(("place_order", copy.deepcopy(request)))
        if self.before_place is not None:
            self.before_place(request)
        if self.state_store is not None:
            persisted = self.state_store.get_order(request.client_order_id)
            assert persisted.lifecycle is RuntimeOrderLifecycle.INTENT_RECORDED
        if self.fail_place:
            raise OSError("connection lost after request")
        return BrokerOrderAcknowledgement(
            client_order_id=request.client_order_id,
            broker_order_tag=f"TAG-{self.place_count}",
            broker_order_id=f"ORDER-{self.place_count}",
            unique_order_id=f"UNIQUE-{self.place_count}",
            acknowledged_at=acknowledged_at,
            raw_status=True,
            raw_message="SUCCESS",
        )

    def get_order(self, **identity):
        self.calls.append(("get_order", identity))
        value = next(iter(identity.values()))
        matching = [
            row
            for row in self.order_snapshots.values()
            if value in {row.unique_order_id, row.broker_order_id, row.broker_order_tag}
        ]
        if len(matching) != 1:
            raise LookupError("not exact")
        return matching[0]

    def list_trade_fills(self):
        self.calls.append(("list_trade_fills", None))
        return self.fills

    def list_orders(self):
        self.calls.append(("list_orders", None))
        return tuple(self.order_snapshots.values())

    def cancel_order(self, order_id, acknowledged_at):
        self.calls.append(("cancel_order", order_id))
        if self.fail_cancel:
            raise OSError("connection lost after cancellation request")
        snapshot = self.order_snapshots.get(order_id)
        if snapshot is not None:
            self.order_snapshots[order_id] = snapshot.model_copy(
                update={"state": BrokerOrderState.CANCELLED}
            )
        return BrokerCancellationAcknowledgement(
            broker_order_id=order_id,
            acknowledged_at=acknowledged_at,
            raw_status=True,
        )

    def list_positions(self):
        self.calls.append(("list_positions", None))
        return self.positions

    def get_funds(self):
        self.calls.append(("get_funds", None))
        return BrokerFunds(
            net=Decimal("1000000"),
            available_cash=Decimal("1000000"),
            available_limit_margin=Decimal("1000000"),
        )


def order_snapshot(
    *,
    broker_order_id: str = "ORDER-1",
    unique_order_id: str = "UNIQUE-1",
    tag: str = "TAG-1",
    action: BrokerTransactionAction = BrokerTransactionAction.BUY,
    state: BrokerOrderState = BrokerOrderState.OPEN,
    filled: int = 0,
    remaining: int = 10,
    requested: int = 10,
) -> BrokerOrderSnapshot:
    return BrokerOrderSnapshot(
        broker_order_id=broker_order_id,
        unique_order_id=unique_order_id,
        broker_order_tag=tag,
        instrument=master().resolve("AAA"),
        transaction_action=action,
        order_type=OrderType.MARKET,
        requested_quantity=requested,
        filled_quantity=filled,
        remaining_quantity=remaining,
        average_price=Decimal("100") if filled else Decimal("0"),
        state=state,
        raw_status=state.value,
        raw_order_status=state.value,
        raw_text=None,
        updated_at=at(9, 17),
        exchange_timestamp=at(9, 17),
    )


def broker_fill(
    fill_id: str,
    quantity: int,
    price: str,
    *,
    order_id: str = "ORDER-1",
    action: BrokerTransactionAction = BrokerTransactionAction.BUY,
    timestamp: datetime | None = None,
) -> BrokerTradeFill:
    return BrokerTradeFill(
        broker_order_id=order_id,
        fill_id=fill_id,
        instrument=master().resolve("AAA"),
        transaction_action=action,
        fill_timestamp=timestamp or at(9, 17),
        fill_price=Decimal(price),
        quantity=quantity,
    )


def test_live_interlock_persist_before_place_ack_not_fill_and_no_retry(tmp_path: Path) -> None:
    disabled_broker = FakeBroker()
    disabled, disabled_store, _ = service(
        tmp_path / "disabled",
        mode=RuntimeMode.LIVE,
        live_enabled=False,
        broker=disabled_broker,
    )
    disabled.start()
    with pytest.raises(PermissionError, match="interlock"):
        disabled.process_plans((plan(),), decision_at=at(9, 16))
    assert not any(name == "place_order" for name, _ in disabled_broker.calls)
    disabled_store.close()

    runtime, store, _ = service(
        tmp_path / "enabled",
        mode=RuntimeMode.LIVE,
        live_enabled=True,
        broker=FakeBroker(),
    )
    broker = runtime.broker
    broker.state_store = store
    runtime.start()
    runtime.process_plans((plan(),), decision_at=at(9, 16))
    orders = store.list_orders("session-1")
    assert len(orders) == 1
    assert orders[0].lifecycle is RuntimeOrderLifecycle.ACKNOWLEDGED
    assert store.load_positions("session-1") == ()
    assert [name for name, _ in broker.calls].count("place_order") == 1
    assert runtime.portfolio_state.reserved_margin == Decimal("20000")
    store.close()


def test_live_broker_funds_are_independent_upper_ceiling(tmp_path: Path) -> None:
    class LowFundsBroker(FakeBroker):
        def get_funds(self):
            return BrokerFunds(
                net=Decimal("1000"),
                available_cash=Decimal("1000"),
                available_limit_margin=Decimal("1000"),
            )

    broker = LowFundsBroker()
    runtime, store, _ = service(
        tmp_path,
        mode=RuntimeMode.LIVE,
        live_enabled=True,
        broker=broker,
    )
    broker.state_store = store
    runtime.start()
    with pytest.raises(RuntimeError, match="broker funds ceiling"):
        runtime.process_plans((plan(),), decision_at=at(9, 16))
    assert runtime.phase is RuntimePhase.HALTED
    assert not any(name == "place_order" for name, _ in broker.calls)
    store.close()


def test_ambiguous_live_submission_halts_and_never_retries(tmp_path: Path) -> None:
    broker = FakeBroker()
    runtime, store, _ = service(
        tmp_path,
        mode=RuntimeMode.LIVE,
        live_enabled=True,
        broker=broker,
    )
    broker.state_store = store
    runtime.start()
    broker.fail_place = True
    with pytest.raises(Exception, match="ambiguous"):
        runtime.process_plans((plan(),), decision_at=at(9, 16))
    assert runtime.phase is RuntimePhase.HALTED
    persisted = store.list_orders("session-1")[0]
    assert persisted.lifecycle is RuntimeOrderLifecycle.SUBMISSION_AMBIGUOUS
    assert [name for name, _ in broker.calls].count("place_order") == 1
    with pytest.raises(RuntimeError, match="TRADING"):
        runtime.process_plans((plan(symbol="BBB"),), decision_at=at(9, 17))
    store.close()


def test_live_partial_fill_reconciliation_retains_evidence_and_reservation(
    tmp_path: Path,
) -> None:
    broker = FakeBroker()
    runtime, store, _ = service(
        tmp_path,
        mode=RuntimeMode.LIVE,
        live_enabled=True,
        broker=broker,
    )
    broker.state_store = store
    runtime.start()
    runtime.process_plans((plan(),), decision_at=at(9, 16))
    broker.order_snapshots["ORDER-1"] = order_snapshot(
        state=BrokerOrderState.PARTIALLY_FILLED, filled=4, remaining=6
    )
    broker.fills = (broker_fill("F1", 4, "100"),)
    broker.positions = (
        BrokerPosition(
            instrument=master().resolve("AAA"),
            product_type="INTRADAY",
            net_quantity=4,
            buy_quantity=4,
            sell_quantity=0,
            buy_average_price=Decimal("100"),
            sell_average_price=Decimal("0"),
            net_average_price=Decimal("100"),
        ),
    )
    runtime.reconcile(at(9, 17))
    assert runtime.positions[0].entry_fill.quantity == 4
    assert runtime.positions[0].entry_fill.is_simulated is False
    assert runtime.portfolio_state.active_reservations
    assert store.list_broker_fills(store.list_orders("session-1")[0].client_order_id) == (
        broker.fills[0],
    )
    runtime.reconcile(at(9, 18))
    assert len(store.list_broker_fills(store.list_orders("session-1")[0].client_order_id)) == 1
    store.close()


def test_live_zero_fill_rejection_releases_without_capacity_rejection(tmp_path: Path) -> None:
    broker = FakeBroker()
    runtime, store, _ = service(
        tmp_path,
        mode=RuntimeMode.LIVE,
        live_enabled=True,
        broker=broker,
    )
    broker.state_store = store
    runtime.start()
    selected = plan()
    runtime.process_plans((selected,), decision_at=at(9, 16))
    broker.order_snapshots["ORDER-1"] = order_snapshot(
        state=BrokerOrderState.REJECTED, filled=0, remaining=10
    )
    runtime.reconcile(at(9, 17))
    assert runtime.portfolio_state.active_reservations == ()
    assert selected.candidate.order_intent.signal.status is SignalStatus.GENERATED
    assert store.load_trades("session-1") == ()
    store.close()


def test_live_square_off_cancels_once_and_exits_only_partial_entry_quantity(
    tmp_path: Path,
) -> None:
    broker = FakeBroker()
    runtime, store, _ = service(
        tmp_path,
        mode=RuntimeMode.LIVE,
        live_enabled=True,
        broker=broker,
    )
    broker.state_store = store
    runtime.start()
    runtime.process_plans((plan(),), decision_at=at(9, 16))
    broker.order_snapshots["ORDER-1"] = order_snapshot(
        state=BrokerOrderState.PARTIALLY_FILLED, filled=4, remaining=6
    )
    broker.fills = (broker_fill("ENTRY-FILL", 4, "100"),)
    broker.positions = (
        BrokerPosition(
            instrument=master().resolve("AAA"),
            product_type="INTRADAY",
            net_quantity=4,
            buy_quantity=4,
            sell_quantity=0,
            buy_average_price=Decimal("100"),
            sell_average_price=Decimal("0"),
            net_average_price=Decimal("100"),
        ),
    )
    runtime.force_square_off(at(15, 20))
    place_calls = [request for name, request in broker.calls if name == "place_order"]
    assert [request.quantity for request in place_calls] == [10, 4]
    assert [name for name, _ in broker.calls].count("cancel_order") == 1

    exit_order = next(
        order for order in store.list_orders("session-1") if order.leg is RuntimeOrderLeg.EXIT
    )
    broker.order_snapshots[exit_order.broker_order_id] = order_snapshot(
        broker_order_id=exit_order.broker_order_id,
        unique_order_id=exit_order.unique_order_id,
        tag=exit_order.broker_order_tag,
        action=BrokerTransactionAction.SELL,
        state=BrokerOrderState.OPEN,
        filled=0,
        remaining=10,
    ).model_copy(update={"requested_quantity": 4, "remaining_quantity": 4})
    runtime.force_square_off(at(15, 21))
    assert [name for name, _ in broker.calls].count("cancel_order") == 1
    assert [name for name, _ in broker.calls].count("place_order") == 2
    assert runtime.positions[0].entry_fill.quantity == 4
    assert runtime.positions[0].exit_lifecycle is RuntimeExitLifecycle.ACKNOWLEDGED
    store.close()


def test_live_protective_exit_is_opposite_once_and_completion_updates_capital(
    tmp_path: Path,
) -> None:
    broker = FakeBroker()
    runtime, store, _ = service(
        tmp_path,
        mode=RuntimeMode.LIVE,
        live_enabled=True,
        broker=broker,
    )
    broker.state_store = store
    runtime.start()
    runtime.process_plans((plan(),), decision_at=at(9, 16))
    broker.order_snapshots["ORDER-1"] = order_snapshot(
        state=BrokerOrderState.FILLED, filled=10, remaining=0
    )
    broker.fills = (broker_fill("ENTRY-FILL", 10, "100"),)
    broker.positions = (
        BrokerPosition(
            instrument=master().resolve("AAA"),
            product_type="INTRADAY",
            net_quantity=10,
            buy_quantity=10,
            sell_quantity=0,
            buy_average_price=Decimal("100"),
            sell_average_price=Decimal("0"),
            net_average_price=Decimal("100"),
        ),
    )
    runtime.reconcile(at(9, 17))
    starting = runtime.economic_capital
    runtime.on_market_tick(tick("111", timestamp=at(9, 18)))
    runtime.on_market_tick(tick("112", timestamp=at(9, 18, 1)))
    place_calls = [request for name, request in broker.calls if name == "place_order"]
    assert len(place_calls) == 2
    assert place_calls[1].transaction_action is BrokerTransactionAction.SELL
    assert place_calls[1].order_type is OrderType.MARKET
    position = runtime.positions[0]
    assert position.exit_lifecycle is RuntimeExitLifecycle.ACKNOWLEDGED

    exit_order = next(
        order for order in store.list_orders("session-1") if order.leg is RuntimeOrderLeg.EXIT
    )
    broker.order_snapshots[exit_order.broker_order_id] = order_snapshot(
        broker_order_id=exit_order.broker_order_id,
        unique_order_id=exit_order.unique_order_id,
        tag=exit_order.broker_order_tag,
        action=BrokerTransactionAction.SELL,
        state=BrokerOrderState.FILLED,
        filled=10,
        remaining=0,
    )
    broker.fills = broker.fills + (
        broker_fill(
            "EXIT-FILL",
            10,
            "111",
            order_id=exit_order.broker_order_id,
            action=BrokerTransactionAction.SELL,
            timestamp=at(9, 19),
        ),
    )
    broker.positions = ()
    runtime.reconcile(at(9, 19))
    completed = store.load_trades("session-1")
    assert completed[0].trade.exit_reason is ExitReason.TARGET_REACHED
    assert completed[0].trade.is_shadow is False
    assert completed[0].broker_entry_fill_ids == ("ENTRY-FILL",)
    assert completed[0].broker_exit_fill_ids == ("EXIT-FILL",)
    assert runtime.positions == ()
    assert runtime.economic_capital == starting + completed[0].trade.net_pnl
    assert runtime.portfolio_state.active_reservations == ()
    runtime.reconcile(at(9, 20))
    assert runtime.positions == ()
    assert store.load_trades("session-1") == completed
    store.close()


def test_live_finalization_failure_retains_broker_evidence_and_restarts_halted(
    tmp_path: Path,
) -> None:
    class TradeFailingStore(FailingFinalizationStore):
        failure_stage = "trade"

    broker = FakeBroker()
    runtime, store, _ = service(
        tmp_path,
        mode=RuntimeMode.LIVE,
        live_enabled=True,
        broker=broker,
        state_store_type=TradeFailingStore,
    )
    broker.state_store = store
    runtime.start()
    runtime.process_plans((plan(),), decision_at=at(9, 16))
    broker.order_snapshots["ORDER-1"] = order_snapshot(
        state=BrokerOrderState.FILLED, filled=10, remaining=0
    )
    broker.fills = (broker_fill("ENTRY-FILL", 10, "100"),)
    broker.positions = (
        BrokerPosition(
            instrument=master().resolve("AAA"),
            product_type="INTRADAY",
            net_quantity=10,
            buy_quantity=10,
            sell_quantity=0,
            buy_average_price=Decimal("100"),
            sell_average_price=Decimal("0"),
            net_average_price=Decimal("100"),
        ),
    )
    runtime.reconcile(at(9, 17))
    starting_capital = runtime.economic_capital
    runtime.on_market_tick(tick("111", timestamp=at(9, 18)))
    exit_order = next(
        order
        for order in store.list_orders("session-1")
        if order.leg is RuntimeOrderLeg.EXIT
    )
    broker.order_snapshots[exit_order.broker_order_id] = order_snapshot(
        broker_order_id=exit_order.broker_order_id,
        unique_order_id=exit_order.unique_order_id,
        tag=exit_order.broker_order_tag,
        action=BrokerTransactionAction.SELL,
        state=BrokerOrderState.FILLED,
        filled=10,
        remaining=0,
    )
    exit_fill = broker_fill(
        "EXIT-FILL",
        10,
        "111",
        order_id=exit_order.broker_order_id,
        action=BrokerTransactionAction.SELL,
        timestamp=at(9, 19),
    )
    broker.fills += (exit_fill,)
    broker.positions = ()

    with pytest.raises(OSError, match="after trade"):
        runtime.reconcile(at(9, 19))

    assert runtime.phase is RuntimePhase.HALTED
    assert runtime.economic_capital == starting_capital
    assert len(runtime.positions) == 1
    assert runtime.positions[0].exit_lifecycle is RuntimeExitLifecycle.FILLED
    assert store.load_trades("session-1") == ()
    assert store.load_allocations("session-1")[0][3] == "OPEN"
    assert len(store.load_reservations("session-1")) == 1
    assert store.list_broker_fills(exit_order.client_order_id) == (exit_fill,)
    place_count = len([call for call in broker.calls if call[0] == "place_order"])
    store.close()

    reopened = TradeFailingStore(
        config(tmp_path, mode=RuntimeMode.LIVE, live_enabled=True).state_db_path
    )
    recovered = RuntimeService(
        runtime_session_id="session-1",
        trading_date=TRADING_DATE,
        config=config(tmp_path, mode=RuntimeMode.LIVE, live_enabled=True),
        clock=FakeClock(at(9, 20)),
        trading_calendar=ExplicitTradingDayCalendar([TRADING_DATE]),
        state_store=reopened,
        margin_provider=FixedMarginProvider(),
        broker=broker,
        instrument_master=master(),
    )
    broker.state_store = reopened
    assert recovered.start() is RuntimePhase.HALTED
    assert reopened.load_trades("session-1") == ()
    assert len(reopened.load_positions("session-1")) == 1
    assert len(reopened.load_reservations("session-1")) == 1
    assert reopened.get_session("session-1").current_capital == starting_capital
    assert reopened.list_broker_fills(exit_order.client_order_id) == (exit_fill,)
    assert len([call for call in broker.calls if call[0] == "place_order"]) == place_count
    reopened.close()


def _filled_live_runtime(tmp_path: Path):
    broker = FakeBroker()
    runtime, store, _ = service(
        tmp_path,
        mode=RuntimeMode.LIVE,
        live_enabled=True,
        broker=broker,
    )
    broker.state_store = store
    runtime.start()
    selected = plan()
    runtime.process_plans((selected,), decision_at=at(9, 16))
    broker.order_snapshots["ORDER-1"] = order_snapshot(
        state=BrokerOrderState.FILLED, filled=10, remaining=0
    )
    broker.fills = (broker_fill("ENTRY", 10, "100"),)
    broker.positions = (
        BrokerPosition(
            instrument=master().resolve("AAA"),
            product_type="INTRADAY",
            net_quantity=10,
            buy_quantity=10,
            sell_quantity=0,
            buy_average_price=Decimal("100"),
            sell_average_price=Decimal("0"),
            net_average_price=Decimal("100"),
        ),
    )
    runtime.reconcile(at(9, 17))
    return runtime, store, broker, selected


def test_exit_attempt_identity_is_deterministic_and_attempt_specific() -> None:
    selected = candidate()
    first = runtime_client_order_id(
        "session-1", selected.identity, RuntimeOrderLeg.EXIT, 1
    )
    assert first == runtime_client_order_id(
        "session-1", selected.identity, RuntimeOrderLeg.EXIT, 1
    )
    assert first != runtime_client_order_id(
        "session-1", selected.identity, RuntimeOrderLeg.EXIT, 2
    )
    with pytest.raises(ValueError, match="ENTRY order attempt"):
        runtime_client_order_id(
            "session-1", selected.identity, RuntimeOrderLeg.ENTRY, 2
        )


def test_terminal_partial_exit_retries_remainder_and_aggregates_final_trade(
    tmp_path: Path,
) -> None:
    runtime, store, broker, _ = _filled_live_runtime(tmp_path)
    starting = runtime.economic_capital
    runtime.on_market_tick(tick("111", timestamp=at(9, 18)))
    first_exit = next(
        order for order in store.list_orders("session-1") if order.leg is RuntimeOrderLeg.EXIT
    )
    broker.order_snapshots[first_exit.broker_order_id] = order_snapshot(
        broker_order_id=first_exit.broker_order_id,
        unique_order_id=first_exit.unique_order_id,
        tag=first_exit.broker_order_tag,
        action=BrokerTransactionAction.SELL,
        state=BrokerOrderState.CANCELLED,
        filled=4,
        remaining=6,
    )
    broker.fills += (
        broker_fill(
            "EXIT-1",
            4,
            "110",
            order_id=first_exit.broker_order_id,
            action=BrokerTransactionAction.SELL,
            timestamp=at(9, 19),
        ),
    )
    broker.positions = (
        BrokerPosition(
            instrument=master().resolve("AAA"),
            product_type="INTRADAY",
            net_quantity=6,
            buy_quantity=10,
            sell_quantity=4,
            buy_average_price=Decimal("100"),
            sell_average_price=Decimal("110"),
            net_average_price=Decimal("100"),
        ),
    )
    runtime.reconcile(at(9, 19))
    assert runtime.positions[0].exit_filled_quantity == 4
    assert runtime.positions[0].exit_lifecycle is RuntimeExitLifecycle.NONE

    runtime.on_market_tick(tick("112", timestamp=at(9, 20)))
    exit_orders = sorted(
        (order for order in store.list_orders("session-1") if order.leg is RuntimeOrderLeg.EXIT),
        key=lambda order: order.attempt,
    )
    assert [order.attempt for order in exit_orders] == [1, 2]
    assert exit_orders[0].client_order_id != exit_orders[1].client_order_id
    assert exit_orders[1].quantity == 6
    assert [request.quantity for name, request in broker.calls if name == "place_order"] == [
        10,
        10,
        6,
    ]

    second_exit = exit_orders[1]
    broker.order_snapshots[second_exit.broker_order_id] = order_snapshot(
        broker_order_id=second_exit.broker_order_id,
        unique_order_id=second_exit.unique_order_id,
        tag=second_exit.broker_order_tag,
        action=BrokerTransactionAction.SELL,
        state=BrokerOrderState.FILLED,
        filled=6,
        remaining=0,
        requested=6,
    )
    broker.fills += (
        broker_fill(
            "EXIT-2",
            6,
            "120",
            order_id=second_exit.broker_order_id,
            action=BrokerTransactionAction.SELL,
            timestamp=at(9, 21),
        ),
    )
    broker.positions = ()
    runtime.reconcile(at(9, 21))
    completed = store.load_trades("session-1")
    assert len(completed) == 1
    assert completed[0].trade.exit_fill.quantity == 10
    assert completed[0].trade.exit_fill.price == Decimal("116")
    assert completed[0].trade.exit_fill.timestamp == at(9, 21)
    assert completed[0].broker_exit_client_order_ids == tuple(
        order.client_order_id for order in exit_orders
    )
    assert completed[0].broker_exit_fill_ids == ("EXIT-1", "EXIT-2")
    assert runtime.portfolio_state.active_reservations == ()
    capital = runtime.economic_capital
    assert capital == starting + completed[0].trade.net_pnl
    runtime.reconcile(at(9, 22))
    assert runtime.economic_capital == capital
    assert store.load_trades("session-1") == completed
    store.close()


@pytest.mark.parametrize(
    "state",
    [
        BrokerOrderState.OPEN,
        BrokerOrderState.PARTIALLY_FILLED,
    ],
)
def test_active_partial_exit_never_creates_second_attempt(
    tmp_path: Path, state: BrokerOrderState
) -> None:
    runtime, store, broker, _ = _filled_live_runtime(tmp_path)
    runtime.on_market_tick(tick("111", timestamp=at(9, 18)))
    exit_order = next(
        order for order in store.list_orders("session-1") if order.leg is RuntimeOrderLeg.EXIT
    )
    filled = 4 if state is BrokerOrderState.PARTIALLY_FILLED else 0
    broker.order_snapshots[exit_order.broker_order_id] = order_snapshot(
        broker_order_id=exit_order.broker_order_id,
        unique_order_id=exit_order.unique_order_id,
        tag=exit_order.broker_order_tag,
        action=BrokerTransactionAction.SELL,
        state=state,
        filled=filled,
        remaining=10 - filled,
    )
    if filled:
        broker.fills += (
            broker_fill(
                "PARTIAL",
                filled,
                "111",
                order_id=exit_order.broker_order_id,
                action=BrokerTransactionAction.SELL,
            ),
        )
        broker.positions = (
            broker.positions[0].model_copy(
                update={"net_quantity": 6, "sell_quantity": 4}
            ),
        )
    runtime.reconcile(at(9, 19))
    runtime.force_square_off(at(15, 20))
    exits = [
        order for order in store.list_orders("session-1") if order.leg is RuntimeOrderLeg.EXIT
    ]
    assert len(exits) == 1
    store.close()


def test_ambiguous_and_unknown_exit_states_fail_closed_without_new_attempt(
    tmp_path: Path,
) -> None:
    runtime, store, broker, _ = _filled_live_runtime(tmp_path / "ambiguous")
    broker.fail_place = True
    runtime.on_market_tick(tick("111", timestamp=at(9, 18)))
    assert runtime.phase is RuntimePhase.HALTED
    assert next(
        order
        for order in store.list_orders("session-1")
        if order.leg is RuntimeOrderLeg.EXIT
    ).lifecycle is RuntimeOrderLifecycle.SUBMISSION_AMBIGUOUS
    with pytest.raises(BrokerAmbiguousStateError):
        runtime.force_square_off(at(15, 20))
    assert len([o for o in store.list_orders("session-1") if o.leg is RuntimeOrderLeg.EXIT]) == 1
    store.close()

    unknown, unknown_store, unknown_broker, _ = _filled_live_runtime(tmp_path / "unknown")
    unknown.on_market_tick(tick("111", timestamp=at(9, 18)))
    exit_order = next(
        order
        for order in unknown_store.list_orders("session-1")
        if order.leg is RuntimeOrderLeg.EXIT
    )
    unknown_broker.order_snapshots[exit_order.broker_order_id] = order_snapshot(
        broker_order_id=exit_order.broker_order_id,
        unique_order_id=exit_order.unique_order_id,
        tag=exit_order.broker_order_tag,
        action=BrokerTransactionAction.SELL,
        state=BrokerOrderState.UNKNOWN,
    )
    unknown.reconcile(at(9, 19))
    assert unknown.phase is RuntimePhase.HALTED
    unknown.force_square_off(at(15, 20))
    assert len(
        [
            order
            for order in unknown_store.list_orders("session-1")
            if order.leg is RuntimeOrderLeg.EXIT
        ]
    ) == 1
    unknown_store.close()


def test_exit_overfill_halts_as_accounting_corruption(tmp_path: Path) -> None:
    runtime, store, broker, _ = _filled_live_runtime(tmp_path)
    runtime.on_market_tick(tick("111", timestamp=at(9, 18)))
    exit_order = next(o for o in store.list_orders("session-1") if o.leg is RuntimeOrderLeg.EXIT)
    broker.order_snapshots[exit_order.broker_order_id] = order_snapshot(
        broker_order_id=exit_order.broker_order_id,
        unique_order_id=exit_order.unique_order_id,
        tag=exit_order.broker_order_tag,
        action=BrokerTransactionAction.SELL,
        state=BrokerOrderState.FILLED,
        filled=11,
        remaining=0,
        requested=11,
    )
    broker.fills += (
        broker_fill(
            "OVER",
            11,
            "111",
            order_id=exit_order.broker_order_id,
            action=BrokerTransactionAction.SELL,
        ),
    )
    runtime.reconcile(at(9, 19))
    assert runtime.phase is RuntimePhase.HALTED
    assert "exceed" in store.get_session("session-1").halt_reason
    assert store.load_trades("session-1") == ()
    store.close()


def test_partial_entry_is_cancelled_and_final_fill_quantity_precedes_exit(
    tmp_path: Path,
) -> None:
    broker = FakeBroker()
    runtime, store, _ = service(
        tmp_path, mode=RuntimeMode.LIVE, live_enabled=True, broker=broker
    )
    broker.state_store = store
    runtime.start()
    runtime.process_plans((plan(),), decision_at=at(9, 16))
    broker.order_snapshots["ORDER-1"] = order_snapshot(
        state=BrokerOrderState.PARTIALLY_FILLED, filled=4, remaining=6
    )
    broker.fills = (broker_fill("ENTRY-1", 4, "100"),)
    broker.positions = (
        BrokerPosition(
            instrument=master().resolve("AAA"),
            product_type="INTRADAY",
            net_quantity=4,
            buy_quantity=4,
            sell_quantity=0,
            buy_average_price=Decimal("100"),
            sell_average_price=Decimal("0"),
            net_average_price=Decimal("100"),
        ),
    )
    runtime.reconcile(at(9, 17))
    broker.order_snapshots["ORDER-1"] = order_snapshot(
        state=BrokerOrderState.PARTIALLY_FILLED, filled=6, remaining=4
    )
    broker.fills += (broker_fill("ENTRY-2", 2, "102"),)
    broker.positions = (
        broker.positions[0].model_copy(
            update={"net_quantity": 6, "buy_quantity": 6}
        ),
    )
    runtime.on_market_tick(tick("111", timestamp=at(9, 18)))
    requests = [request for name, request in broker.calls if name == "place_order"]
    assert [request.quantity for request in requests] == [10, 6]
    assert [name for name, _ in broker.calls].count("cancel_order") == 1
    assert runtime.positions[0].entry_fill.quantity == 6
    store.close()


@pytest.mark.parametrize(
    "current,expected",
    [
        (at(9), RuntimePhase.READY),
        (at(9, 15), RuntimePhase.TRADING),
        (at(15, 10), RuntimePhase.ENTRY_CLOSED),
        (at(15, 20), RuntimePhase.SQUARE_OFF),
        (at(15, 31), RuntimePhase.ENTRY_CLOSED),
    ],
)
def test_starting_late_derives_safe_phase(
    tmp_path: Path, current: datetime, expected: RuntimePhase
) -> None:
    runtime, store, _ = service(tmp_path, current=current)
    assert runtime.start() is expected
    store.close()


def test_entry_cutoff_is_hard_gate_and_halt_never_auto_unhalts(tmp_path: Path) -> None:
    runtime, store, clock = service(tmp_path, current=at(9, 16))
    runtime.start()
    clock.current = at(15, 10)
    with pytest.raises(RuntimeError, match="cutoff"):
        runtime.process_plans((plan(),))
    assert runtime.phase is RuntimePhase.ENTRY_CLOSED
    runtime.halt("test halt", at(15, 11))
    runtime.market_open(at(15, 12))
    assert runtime.phase is RuntimePhase.HALTED
    assert store.get_session("session-1").halt_reason == "test halt"
    store.close()


def test_entry_cutoff_durably_cancels_paper_actual_but_not_shadow(tmp_path: Path) -> None:
    runtime, store, clock = service(
        tmp_path, capital=Decimal("50000"), margin=Decimal("40000")
    )
    runtime.start()
    actual = plan(
        symbol="AAA",
        quality=0.9,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("90"),
    )
    shadow = plan(
        symbol="BBB",
        quality=0.1,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("90"),
    )
    runtime.process_plans((actual, shadow), decision_at=at(9, 16))
    clock.current = at(15, 10)
    runtime.close_entries()
    statuses = {
        fingerprint: status
        for fingerprint, _, _, status in store.load_allocations("session-1")
    }
    assert statuses[candidate_fingerprint(actual.candidate)] == "CANCELLED"
    assert statuses[candidate_fingerprint(shadow.candidate)] == "PENDING"
    assert runtime.portfolio_state.active_reservations == ()
    runtime.force_square_off(at(15, 20))
    assert all(
        status == "CANCELLED"
        for _, _, _, status in store.load_allocations("session-1")
    )
    store.close()

    reopened = RuntimeStateStore(config(
        tmp_path, capital=Decimal("50000")
    ).state_db_path)
    recovered = RuntimeService(
        runtime_session_id="session-1",
        trading_date=TRADING_DATE,
        config=config(tmp_path, capital=Decimal("50000")),
        clock=FakeClock(at(15, 21)),
        trading_calendar=ExplicitTradingDayCalendar([TRADING_DATE]),
        state_store=reopened,
        margin_provider=FixedMarginProvider(Decimal("40000")),
    )
    assert recovered.start() is RuntimePhase.SQUARE_OFF
    recovered.on_market_tick(tick("90", symbol="AAA", timestamp=at(15, 21)))
    recovered.on_market_tick(tick("90", symbol="BBB", timestamp=at(15, 21)))
    assert recovered.positions == ()
    reopened.close()


def test_live_entry_cutoff_cancels_unfilled_actual_and_releases_reservation(
    tmp_path: Path,
) -> None:
    broker = FakeBroker()
    runtime, store, clock = service(
        tmp_path, mode=RuntimeMode.LIVE, live_enabled=True, broker=broker
    )
    broker.state_store = store
    runtime.start()
    runtime.process_plans((plan(),), decision_at=at(9, 16))
    broker.order_snapshots["ORDER-1"] = order_snapshot(state=BrokerOrderState.OPEN)
    clock.current = at(15, 10)
    runtime.close_entries()
    assert [name for name, _ in broker.calls].count("cancel_order") == 1
    assert runtime.positions == ()
    assert runtime.portfolio_state.active_reservations == ()
    assert store.load_allocations("session-1")[0][3] == "CANCELLED"
    store.close()


def test_halted_live_cutoff_cancels_open_entry_once_and_preserves_halt(
    tmp_path: Path,
) -> None:
    broker = FakeBroker()
    runtime, store, _ = service(
        tmp_path, mode=RuntimeMode.LIVE, live_enabled=True, broker=broker
    )
    broker.state_store = store
    runtime.start()
    runtime.process_plans((plan(),), decision_at=at(9, 16))
    broker.order_snapshots["ORDER-1"] = order_snapshot(state=BrokerOrderState.OPEN)
    runtime.halt("pre-cutoff safety halt", at(15, 9))

    runtime.close_entries(at(15, 10))
    assert runtime.phase is RuntimePhase.HALTED
    assert store.list_orders("session-1")[0].lifecycle is RuntimeOrderLifecycle.CANCELLED
    assert [name for name, _ in broker.calls].count("cancel_order") == 1
    assert [name for name, _ in broker.calls].count("place_order") == 1
    assert runtime.portfolio_state.active_reservations == ()

    runtime.close_entries(at(15, 11))
    assert runtime.phase is RuntimePhase.HALTED
    assert [name for name, _ in broker.calls].count("cancel_order") == 1
    assert [name for name, _ in broker.calls].count("place_order") == 1
    store.close()


def test_halted_cutoff_stabilizes_partial_entry_and_retains_actual_exposure(
    tmp_path: Path,
) -> None:
    broker = FakeBroker()
    runtime, store, _ = service(
        tmp_path, mode=RuntimeMode.LIVE, live_enabled=True, broker=broker
    )
    broker.state_store = store
    runtime.start()
    runtime.process_plans((plan(),), decision_at=at(9, 16))
    broker.order_snapshots["ORDER-1"] = order_snapshot(
        state=BrokerOrderState.PARTIALLY_FILLED, filled=4, remaining=6
    )
    broker.fills = (broker_fill("ENTRY-PARTIAL", 4, "100"),)
    broker.positions = (
        BrokerPosition(
            instrument=master().resolve("AAA"),
            product_type="INTRADAY",
            net_quantity=4,
            buy_quantity=4,
            sell_quantity=0,
            buy_average_price=Decimal("100"),
            sell_average_price=Decimal("0"),
            net_average_price=Decimal("100"),
        ),
    )
    runtime.halt("pre-cutoff safety halt", at(15, 9))
    runtime.close_entries(at(15, 10))

    assert runtime.phase is RuntimePhase.HALTED
    assert runtime.positions[0].entry_fill.quantity == 4
    assert runtime.positions[0].reservation is not None
    assert runtime.portfolio_state.reserved_margin == Decimal("20000")
    assert store.load_allocations("session-1")[0][3] == "OPEN"
    assert store.list_orders("session-1")[0].lifecycle is RuntimeOrderLifecycle.CANCELLED
    assert [name for name, _ in broker.calls].count("cancel_order") == 1
    runtime.close_entries(at(15, 11))
    assert [name for name, _ in broker.calls].count("cancel_order") == 1
    store.close()


def test_after_cutoff_recovery_does_not_restore_cancelled_actual_pending_entry(
    tmp_path: Path,
) -> None:
    runtime, store, clock = service(tmp_path)
    runtime.start()
    selected = plan(
        order_type=OrderType.LIMIT,
        limit_price=Decimal("90"),
    )
    runtime.process_plans((selected,), decision_at=at(9, 16))
    clock.current = at(15, 10)
    runtime.close_entries()
    assert store.load_allocations("session-1")[0][3] == "CANCELLED"
    store.close()

    reopened = RuntimeStateStore(config(tmp_path).state_db_path)
    recovered = RuntimeService(
        runtime_session_id="session-1",
        trading_date=TRADING_DATE,
        config=config(tmp_path),
        clock=FakeClock(at(15, 11)),
        trading_calendar=ExplicitTradingDayCalendar([TRADING_DATE]),
        state_store=reopened,
        margin_provider=FixedMarginProvider(),
    )
    assert recovered.start() is RuntimePhase.ENTRY_CLOSED
    recovered.on_market_tick(tick("90", timestamp=at(15, 11)))
    assert recovered.positions == ()
    reopened.close()


def test_allocation_batch_is_atomic_before_first_live_order_side_effect(
    tmp_path: Path,
) -> None:
    broker = FakeBroker()
    runtime, store, _ = service(
        tmp_path,
        mode=RuntimeMode.LIVE,
        capital=Decimal("50000"),
        margin=Decimal("40000"),
        live_enabled=True,
        broker=broker,
    )
    broker.state_store = store
    runtime.start()
    selected = (plan(symbol="AAA", quality=0.9), plan(symbol="BBB", quality=0.1))

    def assert_batch_durable(_request) -> None:
        allocations = store.load_allocations("session-1")
        assert len(allocations) == 2
        assert {decision.outcome for _, _, decision, _ in allocations} == {
            AllocationOutcome.ALLOCATED,
            AllocationOutcome.CAPACITY_REJECTED,
        }
        assert len(store.load_reservations("session-1")) == 1

    broker.before_place = assert_batch_durable
    broker.fail_place = True
    with pytest.raises(Exception, match="ambiguous"):
        runtime.process_plans(selected, decision_at=at(9, 16))
    assert len(store.load_allocations("session-1")) == 2
    assert runtime.phase is RuntimePhase.HALTED
    store.close()


def test_allocation_batch_persistence_failure_prevents_broker_side_effect(
    tmp_path: Path,
) -> None:
    class FailingBatchStore(RuntimeStateStore):
        def save_allocation_batch(self, *args, **kwargs):
            raise OSError("injected persistence failure")

    selected_config = config(tmp_path, mode=RuntimeMode.LIVE, live_enabled=True)
    store = FailingBatchStore(selected_config.state_db_path)
    broker = FakeBroker(store)
    runtime = RuntimeService(
        runtime_session_id="session-1",
        trading_date=TRADING_DATE,
        config=selected_config,
        clock=FakeClock(at(9, 16)),
        trading_calendar=ExplicitTradingDayCalendar([TRADING_DATE]),
        state_store=store,
        margin_provider=FixedMarginProvider(),
        broker=broker,
        instrument_master=master(),
    )
    runtime.start()
    with pytest.raises(OSError, match="injected"):
        runtime.process_plans((plan(),), decision_at=at(9, 16))
    assert broker.place_count == 0
    assert runtime.phase is RuntimePhase.HALTED
    store.close()


def test_nontrading_date_and_market_close_open_position_fail_closed(tmp_path: Path) -> None:
    selected_config = config(tmp_path)
    store = RuntimeStateStore(selected_config.state_db_path)
    runtime = RuntimeService(
        runtime_session_id="closed-day",
        trading_date=TRADING_DATE,
        config=selected_config,
        clock=FakeClock(at(9)),
        trading_calendar=ExplicitTradingDayCalendar([]),
        state_store=store,
        margin_provider=FixedMarginProvider(),
    )
    assert runtime.start() is RuntimePhase.HALTED
    store.close()

    active, active_store, _ = service(tmp_path / "active")
    active.start()
    active.process_plans((plan(protective_exit=None),), decision_at=at(9, 16))
    active.on_market_tick(tick("100", timestamp=at(9, 16)))
    assert not active.market_close_check(at(15, 30))
    assert active.phase is RuntimePhase.HALTED
    assert active.shutdown(at(15, 35)) is RuntimePhase.HALTED
    active_store.close()


@pytest.mark.parametrize(
    "state",
    [
        BrokerOrderState.OPEN,
        BrokerOrderState.PARTIALLY_FILLED,
        BrokerOrderState.UNKNOWN,
    ],
)
def test_shutdown_refuses_nonterminal_live_entry_states(
    tmp_path: Path, state: BrokerOrderState
) -> None:
    broker = FakeBroker()
    runtime, store, _ = service(
        tmp_path, mode=RuntimeMode.LIVE, live_enabled=True, broker=broker
    )
    broker.state_store = store
    runtime.start()
    runtime.process_plans((plan(),), decision_at=at(9, 16))
    filled = 4 if state is BrokerOrderState.PARTIALLY_FILLED else 0
    broker.order_snapshots["ORDER-1"] = order_snapshot(
        state=state, filled=filled, remaining=10 - filled
    )
    if filled:
        broker.fills = (broker_fill("ENTRY-PARTIAL", 4, "100"),)
        broker.positions = (
            BrokerPosition(
                instrument=master().resolve("AAA"),
                product_type="INTRADAY",
                net_quantity=4,
                buy_quantity=4,
                sell_quantity=0,
                buy_average_price=Decimal("100"),
                sell_average_price=Decimal("0"),
                net_average_price=Decimal("100"),
            ),
        )
    assert runtime.shutdown(at(15, 35)) is RuntimePhase.HALTED


def test_shutdown_refuses_submission_ambiguous_live_entry(tmp_path: Path) -> None:
    broker = FakeBroker()
    runtime, store, _ = service(
        tmp_path, mode=RuntimeMode.LIVE, live_enabled=True, broker=broker
    )
    broker.state_store = store
    runtime.start()
    broker.fail_place = True
    with pytest.raises(Exception, match="ambiguous"):
        runtime.process_plans((plan(),), decision_at=at(9, 16))
    assert runtime.shutdown(at(15, 35)) is RuntimePhase.HALTED


def test_shutdown_refuses_pending_actual_paper_entry_and_orphan_reservation(
    tmp_path: Path,
) -> None:
    pending, _, _ = service(tmp_path / "pending")
    pending.start()
    pending.process_plans((plan(),), decision_at=at(9, 16))
    assert pending.shutdown(at(15, 35)) is RuntimePhase.HALTED

    orphaned, orphaned_store, _ = service(tmp_path / "orphaned")
    orphaned.start()
    selected = plan()
    orphaned.process_plans((selected,), decision_at=at(9, 16))
    orphaned._paper_gateway.cancel_pending_actual_entries()
    orphaned_store.update_allocation_status(
        "session-1", candidate_fingerprint(selected.candidate), "CANCELLED"
    )
    assert orphaned.shutdown(at(15, 35)) is RuntimePhase.HALTED


def test_shutdown_allows_clean_terminal_actual_state_and_ignores_shadow_only(
    tmp_path: Path,
) -> None:
    broker = FakeBroker()
    terminal, terminal_store, _ = service(
        tmp_path / "terminal",
        mode=RuntimeMode.LIVE,
        live_enabled=True,
        broker=broker,
    )
    broker.state_store = terminal_store
    terminal.start()
    terminal.process_plans((plan(),), decision_at=at(9, 16))
    broker.order_snapshots["ORDER-1"] = order_snapshot(
        state=BrokerOrderState.CANCELLED
    )
    assert terminal.shutdown(at(15, 35)) is RuntimePhase.STOPPED

    shadow, _, _ = service(
        tmp_path / "shadow",
        capital=Decimal("10000"),
        margin=Decimal("20000"),
    )
    shadow.start()
    result = shadow.process_plans((plan(),), decision_at=at(9, 16))
    assert result.decisions[0].outcome is AllocationOutcome.CAPACITY_REJECTED
    assert shadow.shutdown(at(15, 35)) is RuntimePhase.STOPPED


def test_stale_market_data_and_stream_error_halt_entries(tmp_path: Path) -> None:
    runtime, store, _ = service(tmp_path)
    runtime.start()
    runtime.process_plans((plan(protective_exit=None),), decision_at=at(9, 16))
    runtime.on_market_tick(tick("100", timestamp=at(9, 16)))
    assert not runtime.check_market_data_health(at(9, 17))
    assert runtime.phase is RuntimePhase.HALTED
    store.close()

    other, other_store, _ = service(tmp_path / "other")
    other.start()
    other.on_stream_error(RuntimeError("socket"), at(9, 17))
    assert other.phase is RuntimePhase.HALTED
    other_store.close()


class FakeStream:
    def __init__(self) -> None:
        self.calls = []
        self.connected = Event()
        self.closed = Event()

    def configure_initial_subscription(self, instruments):
        self.calls.append(("configure", instruments))

    def connect(self):
        self.calls.append(("connect", None))
        self.connected.set()
        self.closed.wait()

    def unsubscribe(self, instruments):
        self.calls.append(("unsubscribe", instruments))

    def close(self):
        self.calls.append(("close", None))
        self.closed.set()


def test_stream_lifecycle_is_explicit_single_and_cleanly_closed(tmp_path: Path) -> None:
    selected_config = config(tmp_path)
    store = RuntimeStateStore(selected_config.state_db_path)
    stream = FakeStream()
    runtime = RuntimeService(
        runtime_session_id="stream-session",
        trading_date=TRADING_DATE,
        config=selected_config,
        clock=FakeClock(at(9, 16)),
        trading_calendar=ExplicitTradingDayCalendar([TRADING_DATE]),
        state_store=store,
        margin_provider=FixedMarginProvider(),
        stream=stream,
    )
    runtime.start()
    instruments = (master().resolve("AAA"),)
    assert stream.calls == []
    runtime.connect_stream(instruments)
    assert stream.connected.wait(timeout=1)
    with pytest.raises(RuntimeError, match="already connected"):
        runtime.connect_stream(instruments)
    assert runtime.shutdown(at(15, 35)) is RuntimePhase.STOPPED
    assert runtime._stream_thread is not None
    assert runtime._stream_thread.daemon
    assert not runtime._stream_thread.is_alive()
    assert stream.calls == [
        ("configure", instruments),
        ("connect", None),
        ("unsubscribe", instruments),
        ("close", None),
    ]


class StuckStream(FakeStream):
    def __init__(self) -> None:
        super().__init__()
        self.release = Event()

    def connect(self):
        self.calls.append(("connect", None))
        self.connected.set()
        self.release.wait()

    def close(self):
        self.calls.append(("close", None))


def test_stuck_stream_shutdown_is_bounded_audited_and_halted(tmp_path: Path) -> None:
    selected_config = config(tmp_path).model_copy(
        update={"stream_shutdown_timeout_seconds": 0.01}
    )
    store = RuntimeStateStore(selected_config.state_db_path)
    stream = StuckStream()
    runtime = RuntimeService(
        runtime_session_id="stuck-stream-session",
        trading_date=TRADING_DATE,
        config=selected_config,
        clock=FakeClock(at(9, 16)),
        trading_calendar=ExplicitTradingDayCalendar([TRADING_DATE]),
        state_store=store,
        margin_provider=FixedMarginProvider(),
        stream=stream,
    )
    runtime.start()
    runtime.connect_stream((master().resolve("AAA"),))
    assert stream.connected.wait(timeout=1)
    assert runtime.shutdown(at(15, 35)) is RuntimePhase.HALTED
    stream.release.set()
    assert runtime._stream_thread is not None
    runtime._stream_thread.join(timeout=1)
    reopened = RuntimeStateStore(selected_config.state_db_path)
    events = reopened.list_events("stuck-stream-session")
    session = reopened.get_session("stuck-stream-session")
    reopened.close()
    assert session is not None and session.phase is RuntimePhase.HALTED
    assert any(
        event.event_type == "STREAM_THREAD_SHUTDOWN_TIMEOUT" for event in events
    )


def test_live_unknown_external_position_and_invalid_protective_geometry_halt(
    tmp_path: Path,
) -> None:
    broker = FakeBroker()
    broker.positions = (
        BrokerPosition(
            instrument=master().resolve("AAA"),
            product_type="INTRADAY",
            net_quantity=1,
            buy_quantity=1,
            sell_quantity=0,
            buy_average_price=Decimal("100"),
            sell_average_price=Decimal("0"),
            net_average_price=Decimal("100"),
        ),
    )
    runtime, store, _ = service(
        tmp_path / "external",
        mode=RuntimeMode.LIVE,
        live_enabled=False,
        broker=broker,
    )
    assert runtime.start() is RuntimePhase.HALTED
    assert "unexpected external" in store.get_session("session-1").halt_reason
    store.close()

    clean_broker = FakeBroker()
    invalid, invalid_store, _ = service(
        tmp_path / "geometry",
        mode=RuntimeMode.LIVE,
        live_enabled=True,
        broker=clean_broker,
    )
    clean_broker.state_store = invalid_store
    invalid.start()
    invalid_plan = plan(
        protective_exit=ProtectiveExitSpec(stop_price=Decimal("105"))
    )
    invalid.process_plans((invalid_plan,), decision_at=at(9, 16))
    clean_broker.order_snapshots["ORDER-1"] = order_snapshot(
        state=BrokerOrderState.FILLED, filled=10, remaining=0
    )
    clean_broker.fills = (broker_fill("F1", 10, "100"),)
    with pytest.raises(ValueError, match="stop_price"):
        invalid.reconcile(at(9, 17))
    assert invalid.phase is RuntimePhase.HALTED
    invalid_store.close()


def test_live_unknown_active_broker_order_halts_without_adoption(tmp_path: Path) -> None:
    broker = FakeBroker()
    broker.order_snapshots["MANUAL"] = order_snapshot(
        broker_order_id="MANUAL",
        unique_order_id="MANUAL-UNIQUE",
        tag="MANUAL-TAG",
        state=BrokerOrderState.OPEN,
    )
    runtime, store, _ = service(
        tmp_path,
        mode=RuntimeMode.LIVE,
        live_enabled=False,
        broker=broker,
    )
    assert runtime.start() is RuntimePhase.HALTED
    assert "unexpected active broker order" in store.get_session("session-1").halt_reason
    assert store.list_orders("session-1") == ()
    store.close()


def test_live_restart_reconciles_persisted_order_and_respects_cutoff(tmp_path: Path) -> None:
    broker = FakeBroker()
    runtime, store, _ = service(
        tmp_path,
        mode=RuntimeMode.LIVE,
        live_enabled=True,
        broker=broker,
    )
    broker.state_store = store
    runtime.start()
    selected = plan()
    runtime.process_plans((selected,), decision_at=at(9, 16))
    broker.order_snapshots["ORDER-1"] = order_snapshot(
        state=BrokerOrderState.PARTIALLY_FILLED, filled=4, remaining=6
    )
    broker.fills = (broker_fill("F1", 4, "100"),)
    broker.positions = (
        BrokerPosition(
            instrument=master().resolve("AAA"),
            product_type="INTRADAY",
            net_quantity=4,
            buy_quantity=4,
            sell_quantity=0,
            buy_average_price=Decimal("100"),
            sell_average_price=Decimal("0"),
            net_average_price=Decimal("100"),
        ),
    )
    store.close()

    reopened = RuntimeStateStore(config(
        tmp_path, mode=RuntimeMode.LIVE, live_enabled=True
    ).state_db_path)
    recovered = RuntimeService(
        runtime_session_id="session-1",
        trading_date=TRADING_DATE,
        config=config(tmp_path, mode=RuntimeMode.LIVE, live_enabled=True),
        clock=FakeClock(at(15, 10)),
        trading_calendar=ExplicitTradingDayCalendar([TRADING_DATE]),
        state_store=reopened,
        margin_provider=FixedMarginProvider(),
        broker=broker,
        instrument_master=master(),
    )
    broker.state_store = reopened
    assert recovered.start() is RuntimePhase.ENTRY_CLOSED
    assert recovered.positions[0].entry_fill.quantity == 4
    assert recovered.portfolio_state.active_reservations
    with pytest.raises(RuntimeError, match="TRADING"):
        recovered.process_plans((plan(symbol="BBB", minute=17),), decision_at=at(15, 10))
    assert [name for name, _ in broker.calls].count("place_order") == 1
    reopened.close()


def test_after_close_restart_cancels_known_open_entry_before_clean_shutdown(
    tmp_path: Path,
) -> None:
    broker = FakeBroker()
    runtime, store, _ = service(
        tmp_path, mode=RuntimeMode.LIVE, live_enabled=True, broker=broker
    )
    broker.state_store = store
    runtime.start()
    runtime.process_plans((plan(),), decision_at=at(9, 16))
    broker.order_snapshots["ORDER-1"] = order_snapshot(state=BrokerOrderState.OPEN)
    store.close()

    reopened = RuntimeStateStore(config(
        tmp_path, mode=RuntimeMode.LIVE, live_enabled=True
    ).state_db_path)
    recovered = RuntimeService(
        runtime_session_id="session-1",
        trading_date=TRADING_DATE,
        config=config(tmp_path, mode=RuntimeMode.LIVE, live_enabled=True),
        clock=FakeClock(at(15, 31)),
        trading_calendar=ExplicitTradingDayCalendar([TRADING_DATE]),
        state_store=reopened,
        margin_provider=FixedMarginProvider(),
        broker=broker,
        instrument_master=master(),
    )
    broker.state_store = reopened
    assert recovered.start() is RuntimePhase.ENTRY_CLOSED
    assert [name for name, _ in broker.calls].count("cancel_order") == 1
    assert reopened.list_orders("session-1")[0].lifecycle is RuntimeOrderLifecycle.CANCELLED
    assert recovered.positions == ()
    assert recovered.portfolio_state.active_reservations == ()
    assert recovered.shutdown(at(15, 35)) is RuntimePhase.STOPPED


def test_after_close_restart_with_partial_entry_remains_halted_with_exposure(
    tmp_path: Path,
) -> None:
    broker = FakeBroker()
    runtime, store, _ = service(
        tmp_path, mode=RuntimeMode.LIVE, live_enabled=True, broker=broker
    )
    broker.state_store = store
    runtime.start()
    runtime.process_plans((plan(),), decision_at=at(9, 16))
    broker.order_snapshots["ORDER-1"] = order_snapshot(
        state=BrokerOrderState.PARTIALLY_FILLED, filled=4, remaining=6
    )
    broker.fills = (broker_fill("ENTRY-PARTIAL", 4, "100"),)
    broker.positions = (
        BrokerPosition(
            instrument=master().resolve("AAA"),
            product_type="INTRADAY",
            net_quantity=4,
            buy_quantity=4,
            sell_quantity=0,
            buy_average_price=Decimal("100"),
            sell_average_price=Decimal("0"),
            net_average_price=Decimal("100"),
        ),
    )
    store.close()

    reopened = RuntimeStateStore(config(
        tmp_path, mode=RuntimeMode.LIVE, live_enabled=True
    ).state_db_path)
    recovered = RuntimeService(
        runtime_session_id="session-1",
        trading_date=TRADING_DATE,
        config=config(tmp_path, mode=RuntimeMode.LIVE, live_enabled=True),
        clock=FakeClock(at(15, 31)),
        trading_calendar=ExplicitTradingDayCalendar([TRADING_DATE]),
        state_store=reopened,
        margin_provider=FixedMarginProvider(),
        broker=broker,
        instrument_master=master(),
    )
    broker.state_store = reopened
    assert recovered.start() is RuntimePhase.HALTED
    assert recovered.positions[0].entry_fill.quantity == 4
    assert recovered.positions[0].reservation is not None
    assert [name for name, _ in broker.calls].count("cancel_order") == 1
    assert recovered.shutdown(at(15, 35)) is RuntimePhase.HALTED


def test_after_close_restart_with_ambiguous_cancellation_remains_halted(
    tmp_path: Path,
) -> None:
    broker = FakeBroker()
    runtime, store, _ = service(
        tmp_path, mode=RuntimeMode.LIVE, live_enabled=True, broker=broker
    )
    broker.state_store = store
    runtime.start()
    runtime.process_plans((plan(),), decision_at=at(9, 16))
    broker.order_snapshots["ORDER-1"] = order_snapshot(state=BrokerOrderState.OPEN)
    store.close()
    broker.fail_cancel = True

    reopened = RuntimeStateStore(config(
        tmp_path, mode=RuntimeMode.LIVE, live_enabled=True
    ).state_db_path)
    recovered = RuntimeService(
        runtime_session_id="session-1",
        trading_date=TRADING_DATE,
        config=config(tmp_path, mode=RuntimeMode.LIVE, live_enabled=True),
        clock=FakeClock(at(15, 31)),
        trading_calendar=ExplicitTradingDayCalendar([TRADING_DATE]),
        state_store=reopened,
        margin_provider=FixedMarginProvider(),
        broker=broker,
        instrument_master=master(),
    )
    broker.state_store = reopened
    assert recovered.start() is RuntimePhase.HALTED
    assert [name for name, _ in broker.calls].count("cancel_order") == 1
    recovered.close_entries(at(15, 32))
    assert [name for name, _ in broker.calls].count("cancel_order") == 1
    assert recovered.shutdown(at(15, 35)) is RuntimePhase.HALTED


class FakeCandleClient:
    def __init__(self, frame: pl.DataFrame) -> None:
        self.frame = frame
        self.calls = []

    def get_five_minute_candles(self, instrument, start, end):
        self.calls.append((instrument, start, end))
        return self.frame.clone()


def test_completed_candles_respect_bar_availability_without_mutation() -> None:
    frame = pl.DataFrame(
        {
            "timestamp": [at(9, 15), at(9, 20)],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.0, 101.0],
            "volume": [1.0, 2.0],
            "symbol": ["AAA", "AAA"],
        },
        schema={
            "timestamp": pl.Datetime("us", "Asia/Kolkata"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "symbol": pl.String,
        },
    )
    source = frame.clone()
    client = FakeCandleClient(frame)
    assert get_completed_five_minute_candles(
        client, master().resolve("AAA"), at(9, 15), at(9, 25), at(9, 19, 59)
    ).is_empty()
    completed = get_completed_five_minute_candles(
        client, master().resolve("AAA"), at(9, 15), at(9, 25), at(9, 20)
    )
    assert completed["timestamp"].to_list() == [at(9, 15)]
    assert frame.equals(source)


def test_five_minute_cycle_fetches_completed_data_and_health_checks_first(
    tmp_path: Path,
) -> None:
    runtime, store, clock = service(tmp_path, current=at(9, 25))
    runtime.start()
    runtime.on_market_tick(tick("100", timestamp=at(9, 25)))
    frame = pl.DataFrame(
        {
            "timestamp": [at(9, 15), at(9, 20)],
            "open": [100.0, 100.0],
            "high": [101.0, 101.0],
            "low": [99.0, 99.0],
            "close": [100.0, 100.0],
            "volume": [100.0, 100.0],
            "symbol": ["AAA", "AAA"],
        }
    )
    calls = []

    class Plans:
        def plans_for_cycle(self, completed_candles, decision_at):
            calls.append((completed_candles, decision_at, runtime.phase))
            return ()

    cycle = FiveMinuteStrategyCycle(
        service=runtime,
        clock=clock,
        candle_client=FakeCandleClient(frame),
        instruments=(master().resolve("AAA"),),
        history_start={"AAA": at(9, 15)},
        plan_provider=Plans(),
    )
    assert cycle() == ()
    assert calls[0][0]["AAA"]["timestamp"].to_list() == [at(9, 15), at(9, 20)]
    assert runtime.phase is RuntimePhase.TRADING
    store.close()


class FakeScheduler:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.jobs = {}
        self.started = False
        self.shutdown_waits = []

    def add_job(self, callback, **kwargs):
        self.jobs[kwargs["id"]] = (callback, kwargs)

    def start(self):
        self.started = True

    def shutdown(self, wait=True):
        self.shutdown_waits.append(wait)
        self.started = False


class SchedulerService:
    def __init__(self) -> None:
        self.calls = []

    def market_open(self, **kwargs):
        self.calls.append(("open", kwargs))

    def close_entries(self, **kwargs):
        self.calls.append(("cutoff", kwargs))

    def force_square_off(self, **kwargs):
        self.calls.append(("square", kwargs))

    def market_close_check(self, **kwargs):
        self.calls.append(("market-close", kwargs))

    def shutdown(self, **kwargs):
        self.calls.append(("shutdown", kwargs))
        return RuntimePhase.STOPPED


def test_scheduler_uses_explicit_date_jobs_deterministically() -> None:
    selected_service = SchedulerService()
    calendar = ExplicitTradingDayCalendar([TRADING_DATE])
    runtime_scheduler = RuntimeScheduler(
        selected_service,
        calendar,
        scheduler_factory=FakeScheduler,
    )
    ids = runtime_scheduler.configure_date(
        TRADING_DATE, RuntimeSessionTimes(), "session-1"
    )
    assert len(ids) == 5
    assert runtime_scheduler.scheduler.kwargs["timezone"] == IST
    assert all(job[1]["max_instances"] == 1 for job in runtime_scheduler.scheduler.jobs.values())
    assert all(job[1]["coalesce"] is True for job in runtime_scheduler.scheduler.jobs.values())
    assert runtime_scheduler.configure_date(
        TRADING_DATE, RuntimeSessionTimes(), "session-1"
    ) == ids
    assert len(runtime_scheduler.scheduler.jobs) == 5
    for callback, kwargs in runtime_scheduler.scheduler.jobs.values():
        callback()
        assert "kwargs" not in kwargs
    assert [name for name, _ in selected_service.calls] == [
        "open",
        "cutoff",
        "square",
        "market-close",
        "shutdown",
    ]
    assert RuntimeScheduler(
        selected_service,
        ExplicitTradingDayCalendar([]),
        scheduler_factory=FakeScheduler,
    ).configure_date(TRADING_DATE, RuntimeSessionTimes(), "x") == ()


def test_runtime_composition_refuses_live_without_connectivity_preflight(
    tmp_path: Path,
) -> None:
    runtime, store, clock = service(tmp_path)
    scheduler = RuntimeScheduler(
        runtime,
        ExplicitTradingDayCalendar([TRADING_DATE]),
        scheduler_factory=FakeScheduler,
    )

    class Plans:
        def plans_for_cycle(self, completed_candles, decision_at):
            return ()

    with pytest.raises(ValueError, match="connectivity preflight"):
        compose_runtime_application(
            config=config(tmp_path / "live", mode=RuntimeMode.LIVE),
            service=runtime,
            clock=clock,
            candle_client=FakeCandleClient(pl.DataFrame()),
            instruments=(master().resolve("AAA"),),
            history_start={"AAA": at(9, 15)},
            plan_provider=Plans(),
            scheduler=scheduler,
        )
    store.close()


def test_runtime_composition_wires_smartapi_connectivity_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from algo_trader.runtime import composition

    runtime, store, clock = service(tmp_path)
    scheduler = RuntimeScheduler(
        runtime,
        ExplicitTradingDayCalendar([TRADING_DATE]),
        scheduler_factory=FakeScheduler,
    )
    calls = []
    monkeypatch.setattr(
        composition,
        "run_smartapi_connectivity_check",
        lambda **kwargs: calls.append(kwargs) or "ok",
    )

    class Plans:
        def plans_for_cycle(self, completed_candles, decision_at):
            return ()

    app = compose_runtime_application(
        config=config(tmp_path / "live", mode=RuntimeMode.LIVE),
        service=runtime,
        clock=clock,
        candle_client=FakeCandleClient(pl.DataFrame()),
        instruments=(master().resolve("AAA"),),
        history_start={"AAA": at(9, 15)},
        plan_provider=Plans(),
        scheduler=scheduler,
        instrument_master=master(),
    )
    assert app.live_connectivity_preflight is not None
    assert app.live_connectivity_preflight() == "ok"
    assert calls[0]["checked_at"] == clock.now()
    assert calls[0]["quote_symbol"] == "AAA"
    store.close()


def test_scheduler_keeps_scheduled_time_separate_from_actual_runtime_time(
    tmp_path: Path,
) -> None:
    runtime, store, clock = service(tmp_path, current=at(15, 19))
    runtime.start()
    scheduler = RuntimeScheduler(
        runtime,
        ExplicitTradingDayCalendar([TRADING_DATE]),
        scheduler_factory=FakeScheduler,
    )
    scheduler.configure_date(TRADING_DATE, RuntimeSessionTimes(), "session-1")
    square_job = scheduler.scheduler.jobs[
        f"runtime:session-1:{TRADING_DATE.isoformat()}:square-off"
    ]
    assert square_job[1]["trigger"].run_date == at(15, 20)
    clock.current = at(15, 22)
    square_job[0]()
    event = next(
        event
        for event in store.list_events("session-1")
        if event.event_type == "SQUARE_OFF_STARTED"
    )
    assert event.occurred_at == at(15, 22)
    clock.current = at(15, 37)
    shutdown_job = scheduler.scheduler.jobs[
        f"runtime:session-1:{TRADING_DATE.isoformat()}:shutdown"
    ]
    shutdown_job[0]()
    reopened = RuntimeStateStore(config(tmp_path).state_db_path)
    stopped = next(
        event
        for event in reopened.list_events("session-1")
        if event.event_type == "SESSION_STOPPED"
    )
    reopened.close()
    assert stopped.occurred_at == at(15, 37)
    assert scheduler.scheduler.shutdown_waits == [False]


def test_scheduler_owns_idempotent_manual_and_scheduled_teardown() -> None:
    selected_service = SchedulerService()
    owner = RuntimeScheduler(
        selected_service,
        ExplicitTradingDayCalendar([TRADING_DATE]),
        scheduler_factory=FakeScheduler,
    )
    owner.configure_date(TRADING_DATE, RuntimeSessionTimes(), "session-1")
    shutdown_job = owner.scheduler.jobs[
        f"runtime:session-1:{TRADING_DATE.isoformat()}:shutdown"
    ][0]
    assert shutdown_job() is RuntimePhase.STOPPED
    assert owner.shutdown() is RuntimePhase.STOPPED
    assert selected_service.calls == [("shutdown", {})]
    assert owner.scheduler.shutdown_waits == [False]

    manual_service = SchedulerService()
    manual = RuntimeScheduler(
        manual_service,
        ExplicitTradingDayCalendar([TRADING_DATE]),
        scheduler_factory=FakeScheduler,
    )
    assert manual.shutdown() is RuntimePhase.STOPPED
    assert manual.shutdown() is RuntimePhase.STOPPED
    assert manual_service.calls == [("shutdown", {})]
    assert manual.scheduler.shutdown_waits == [True]


class BlockingSchedulerService(SchedulerService):
    def __init__(self) -> None:
        super().__init__()
        self.started = Event()
        self.release = Event()

    def shutdown(self, **kwargs):
        self.calls.append(("shutdown", kwargs))
        self.started.set()
        if not self.release.wait(timeout=2):
            raise RuntimeError("test service release was not signaled")
        return RuntimePhase.STOPPED


class CallbackAwareScheduler(FakeScheduler):
    def __init__(self, callback_returned: Event, **kwargs) -> None:
        super().__init__(**kwargs)
        self.callback_returned = callback_returned

    def shutdown(self, wait=True):
        self.shutdown_waits.append(wait)
        if wait and not self.callback_returned.wait(timeout=2):
            raise RuntimeError("scheduled callback did not return before scheduler wait")
        self.started = False


class ObservableCondition(Condition):
    def __init__(self) -> None:
        super().__init__()
        self.wait_started = Event()

    def wait(self, timeout=None):
        self.wait_started.set()
        return super().wait(timeout)


def test_external_shutdown_owner_does_not_deadlock_scheduled_callback() -> None:
    service = BlockingSchedulerService()
    callback_returned = Event()
    schedulers = []

    def scheduler_factory(**kwargs):
        scheduler = CallbackAwareScheduler(callback_returned, **kwargs)
        schedulers.append(scheduler)
        return scheduler

    owner = RuntimeScheduler(
        service,
        ExplicitTradingDayCalendar([TRADING_DATE]),
        scheduler_factory=scheduler_factory,
    )
    external_results = []
    external_errors = []
    scheduled_results = []

    def run_external() -> None:
        try:
            external_results.append(owner.shutdown(wait=True))
        except BaseException as error:
            external_errors.append(error)

    def run_scheduled() -> None:
        try:
            scheduled_results.append(owner._scheduled_shutdown())
        finally:
            callback_returned.set()

    external_thread = Thread(target=run_external)
    external_thread.start()
    assert service.started.wait(timeout=1)
    scheduled_thread = Thread(target=run_scheduled)
    scheduled_thread.start()
    scheduled_thread.join(timeout=1)
    scheduled_returned_promptly = not scheduled_thread.is_alive()
    service.release.set()
    external_thread.join(timeout=2)
    scheduled_thread.join(timeout=2)

    assert scheduled_returned_promptly
    assert not external_thread.is_alive()
    assert not scheduled_thread.is_alive()
    assert external_errors == []
    assert external_results == [RuntimePhase.STOPPED]
    assert scheduled_results == [None]
    assert service.calls == [("shutdown", {})]
    assert schedulers[0].shutdown_waits == [True]


def test_scheduled_shutdown_owner_allows_external_caller_to_wait_safely() -> None:
    service = BlockingSchedulerService()
    owner = RuntimeScheduler(
        service,
        ExplicitTradingDayCalendar([TRADING_DATE]),
        scheduler_factory=FakeScheduler,
    )
    condition = ObservableCondition()
    owner._shutdown_condition = condition
    scheduled_results = []
    external_results = []
    errors = []

    def run_scheduled() -> None:
        try:
            scheduled_results.append(owner._scheduled_shutdown())
        except BaseException as error:
            errors.append(error)

    def run_external() -> None:
        try:
            external_results.append(owner.shutdown())
        except BaseException as error:
            errors.append(error)

    scheduled_thread = Thread(target=run_scheduled)
    scheduled_thread.start()
    assert service.started.wait(timeout=1)
    external_thread = Thread(target=run_external)
    external_thread.start()
    assert condition.wait_started.wait(timeout=1)
    service.release.set()
    scheduled_thread.join(timeout=2)
    external_thread.join(timeout=2)

    assert not scheduled_thread.is_alive()
    assert not external_thread.is_alive()
    assert errors == []
    assert scheduled_results == [RuntimePhase.STOPPED]
    assert external_results == [RuntimePhase.STOPPED]
    assert service.calls == [("shutdown", {})]
    assert owner.scheduler.shutdown_waits == [False]


def test_scheduler_stops_after_service_shutdown_error_and_reraises() -> None:
    class FailingService(SchedulerService):
        def shutdown(self, **kwargs):
            super().shutdown(**kwargs)
            raise RuntimeError("service failed")

    selected_service = FailingService()
    owner = RuntimeScheduler(
        selected_service,
        ExplicitTradingDayCalendar([TRADING_DATE]),
        scheduler_factory=FakeScheduler,
    )
    with pytest.raises(RuntimeError, match="service failed"):
        owner.shutdown()
    assert owner.scheduler.shutdown_waits == [True]
    with pytest.raises(RuntimeError, match="service failed"):
        owner.shutdown()
    assert len(selected_service.calls) == 1


def test_scheduler_preserves_service_error_when_scheduler_cleanup_also_fails() -> None:
    class FailingService(SchedulerService):
        def shutdown(self, **kwargs):
            super().shutdown(**kwargs)
            raise RuntimeError("primary service failure")

    class FailingScheduler(FakeScheduler):
        def shutdown(self, wait=True):
            self.shutdown_waits.append(wait)
            raise RuntimeError("secondary scheduler failure")

    selected_service = FailingService()
    owner = RuntimeScheduler(
        selected_service,
        ExplicitTradingDayCalendar([TRADING_DATE]),
        scheduler_factory=FailingScheduler,
    )
    with pytest.raises(RuntimeError, match="primary service failure"):
        owner.shutdown()
    with pytest.raises(RuntimeError, match="primary service failure"):
        owner.shutdown()
    assert selected_service.calls == [("shutdown", {})]
    assert owner.scheduler.shutdown_waits == [True]


class ConnectivityBroker:
    def __init__(self) -> None:
        self.calls = []

    def authenticate(self, credentials, checked_at):
        self.calls.append("authenticate")
        return AngelOneSession(
            client_code=credentials.client_code,
            jwt_token="jwt-secret",
            refresh_token="refresh-secret",
            feed_token="feed-secret",
            authenticated_at=checked_at,
            sdk_version="1.5.5",
        )

    def get_funds(self):
        self.calls.append("funds")
        return BrokerFunds(net=Decimal("1"), available_cash=Decimal("1"))

    def list_positions(self):
        self.calls.append("positions")
        return ()

    def list_orders(self):
        self.calls.append("orders")
        return ()

    def get_ltp(self, symbol, checked_at):
        self.calls.append("quote")
        return BrokerQuote(
            instrument=master().resolve(symbol),
            observed_at=checked_at,
            ltp=Decimal("100"),
        )

    def logout(self, session):
        self.calls.append("logout")


def test_connectivity_check_is_read_only_secret_safe_and_interlock_independent(
    tmp_path: Path,
) -> None:
    selected_broker = ConnectivityBroker()
    selected_config = config(tmp_path, mode=RuntimeMode.LIVE, live_enabled=False)
    report = run_smartapi_connectivity_check(
        config=selected_config,
        instrument_master=master(),
        checked_at=at(9),
        quote_symbol="AAA",
        broker_factory=lambda _: selected_broker,
        credentials_loader=lambda _: AngelOneCredentials(
            api_key="api-secret",
            client_code="CLIENT",
            pin="1234",
            totp_secret="totp-secret",
        ),
    )
    assert selected_broker.calls == [
        "authenticate",
        "funds",
        "positions",
        "orders",
        "quote",
        "logout",
    ]
    serialized = report.model_dump_json()
    assert report.authenticated and report.quote_read_ok
    for secret in ("api-secret", "1234", "totp-secret", "jwt-secret"):
        assert secret not in serialized
    assert not hasattr(selected_broker, "place_order")
    assert not hasattr(selected_broker, "cancel_order")


def test_clean_paper_restart_recovers_reservation_and_position(tmp_path: Path) -> None:
    runtime, store, clock = service(tmp_path)
    runtime.start()
    selected = plan(protective_exit=None)
    runtime.process_plans((selected,), decision_at=at(9, 16))
    runtime.on_market_tick(tick("100", timestamp=at(9, 16)))
    assert len(runtime.positions) == 1
    store.close()

    reopened_store = RuntimeStateStore(config(tmp_path).state_db_path)
    recovered = RuntimeService(
        runtime_session_id="session-1",
        trading_date=TRADING_DATE,
        config=config(tmp_path),
        clock=clock,
        trading_calendar=ExplicitTradingDayCalendar([TRADING_DATE]),
        state_store=reopened_store,
        margin_provider=FixedMarginProvider(),
    )
    assert recovered.start() is RuntimePhase.TRADING
    assert len(recovered.positions) == 1
    assert len(recovered.portfolio_state.active_reservations) == 1
    identity = runtime_client_order_id(
        "session-1", selected.candidate.identity, RuntimeOrderLeg.ENTRY
    )
    assert identity == runtime_client_order_id(
        "session-1", recovered.positions[0].candidate.identity, RuntimeOrderLeg.ENTRY
    )
    reopened_store.close()


@pytest.mark.parametrize("mode", [RuntimeMode.PAPER, RuntimeMode.LIVE])
def test_different_id_cannot_bypass_unfinished_same_day_session(
    tmp_path: Path, mode: RuntimeMode
) -> None:
    broker = FakeBroker() if mode is RuntimeMode.LIVE else None
    first, store, _ = service(
        tmp_path,
        mode=mode,
        live_enabled=mode is RuntimeMode.LIVE,
        broker=broker,
        runtime_session_id="existing-session",
    )
    first.start()
    different = RuntimeService(
        runtime_session_id="different-session",
        trading_date=TRADING_DATE,
        config=config(
            tmp_path,
            mode=mode,
            live_enabled=mode is RuntimeMode.LIVE,
        ),
        clock=FakeClock(at(9, 17)),
        trading_calendar=ExplicitTradingDayCalendar([TRADING_DATE]),
        state_store=store,
        margin_provider=FixedMarginProvider(),
        broker=broker,
        instrument_master=master() if mode is RuntimeMode.LIVE else None,
    )
    with pytest.raises(RuntimeError, match="existing-session"):
        different.start()
    with pytest.raises(LookupError):
        store.get_session("different-session")
    store.close()


def test_multiple_unfinished_same_day_sessions_fail_as_ambiguous(tmp_path: Path) -> None:
    selected_config = config(tmp_path)
    store = RuntimeStateStore(selected_config.state_db_path)
    for session_id in ("one", "two"):
        store.create_session(
            RuntimeSessionRecord(
                runtime_session_id=session_id,
                trading_date=TRADING_DATE,
                mode=RuntimeMode.PAPER,
                starting_capital=Decimal("100000"),
                current_capital=Decimal("100000"),
                started_at=at(9),
                live_order_submission_enabled=False,
                configuration_fingerprint="fixture",
            )
        )
    runtime = RuntimeService(
        runtime_session_id="three",
        trading_date=TRADING_DATE,
        config=selected_config,
        clock=FakeClock(at(9, 16)),
        trading_calendar=ExplicitTradingDayCalendar([TRADING_DATE]),
        state_store=store,
        margin_provider=FixedMarginProvider(),
    )
    with pytest.raises(RuntimeError, match="multiple unfinished"):
        runtime.start()
    store.close()


def test_ended_same_day_session_does_not_block_new_session(tmp_path: Path) -> None:
    first, _, _ = service(
        tmp_path, runtime_session_id="ended-session"
    )
    first.start()
    assert first.shutdown(at(15, 35)) is RuntimePhase.STOPPED

    store = RuntimeStateStore(config(tmp_path).state_db_path)
    second = RuntimeService(
        runtime_session_id="new-session",
        trading_date=TRADING_DATE,
        config=config(tmp_path),
        clock=FakeClock(at(9, 16)),
        trading_calendar=ExplicitTradingDayCalendar([TRADING_DATE]),
        state_store=store,
        margin_provider=FixedMarginProvider(),
    )
    assert second.start() is RuntimePhase.TRADING
    store.close()


def test_exit_attempt_sequence_survives_restart_and_never_reuses_identity(
    tmp_path: Path,
) -> None:
    runtime, store, broker, _ = _filled_live_runtime(tmp_path)
    runtime.on_market_tick(tick("111", timestamp=at(9, 18)))
    first_exit = next(o for o in store.list_orders("session-1") if o.leg is RuntimeOrderLeg.EXIT)
    broker.order_snapshots[first_exit.broker_order_id] = order_snapshot(
        broker_order_id=first_exit.broker_order_id,
        unique_order_id=first_exit.unique_order_id,
        tag=first_exit.broker_order_tag,
        action=BrokerTransactionAction.SELL,
        state=BrokerOrderState.CANCELLED,
        filled=4,
        remaining=6,
    )
    broker.fills += (
        broker_fill(
            "EXIT-1",
            4,
            "111",
            order_id=first_exit.broker_order_id,
            action=BrokerTransactionAction.SELL,
        ),
    )
    broker.positions = (
        broker.positions[0].model_copy(
            update={"net_quantity": 6, "sell_quantity": 4}
        ),
    )
    runtime.reconcile(at(9, 19))
    store.close()

    reopened = RuntimeStateStore(config(
        tmp_path, mode=RuntimeMode.LIVE, live_enabled=True
    ).state_db_path)
    recovered = RuntimeService(
        runtime_session_id="session-1",
        trading_date=TRADING_DATE,
        config=config(tmp_path, mode=RuntimeMode.LIVE, live_enabled=True),
        clock=FakeClock(at(9, 20)),
        trading_calendar=ExplicitTradingDayCalendar([TRADING_DATE]),
        state_store=reopened,
        margin_provider=FixedMarginProvider(),
        broker=broker,
        instrument_master=master(),
    )
    broker.state_store = reopened
    assert recovered.start() is RuntimePhase.TRADING
    recovered.on_market_tick(tick("112", timestamp=at(9, 20)))
    exits = sorted(
        (o for o in reopened.list_orders("session-1") if o.leg is RuntimeOrderLeg.EXIT),
        key=lambda order: order.attempt,
    )
    assert [order.attempt for order in exits] == [1, 2]
    assert exits[0].client_order_id == first_exit.client_order_id
    assert exits[1].client_order_id != first_exit.client_order_id
    assert exits[1].quantity == 6
    reopened.close()
