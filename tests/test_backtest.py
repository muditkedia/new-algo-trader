from datetime import UTC, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from algo_trader import (
    ExitReason,
    MLScore,
    OrderIntent,
    OrderType,
    ProtectiveExitSpec,
    Side,
    Signal,
    SignalStatus,
)
from algo_trader.backtest import (
    BacktestConfig,
    BacktestIntegrityError,
    BacktestRequestOutcome,
    BacktestRequestResult,
    BacktestTradeRecord,
    BacktestTradeRequest,
    HistoricalBacktester,
)
from algo_trader.costs import BrokeragePlan
from algo_trader.data import MarketDataConfig, ParquetMarketDataStore
from algo_trader.execution import FixedBasisPointsSlippage, HistoricalExecutionSimulator
from algo_trader.portfolio import (
    AllocationCandidate,
    MarginRequirementQuote,
    PortfolioState,
)

MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")


def at(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=MARKET_TIMEZONE)


def write_symbol(path: Path, symbol: str, rows: list[tuple]) -> None:
    timestamps, opens, highs, lows, closes = zip(*rows, strict=True)
    table = pa.table(
        {
            "date": pa.array(
                [timestamp.astimezone(UTC) for timestamp in timestamps],
                type=pa.timestamp("us", tz="UTC"),
            ),
            "open": pa.array(opens, type=pa.float64()),
            "high": pa.array(highs, type=pa.float64()),
            "low": pa.array(lows, type=pa.float64()),
            "close": pa.array(closes, type=pa.float64()),
            "volume": pa.array([1_000.0] * len(rows), type=pa.float64()),
        }
    )
    pq.write_table(table, path / f"{symbol}.parquet")


def make_store(tmp_path: Path, symbols: dict[str, list[tuple]]) -> ParquetMarketDataStore:
    for symbol, rows in symbols.items():
        write_symbol(tmp_path, symbol, rows)
    return ParquetMarketDataStore(MarketDataConfig(dataset_path=tmp_path))


def make_candidate(
    timestamp: datetime,
    *,
    strategy_id: str = "strategy",
    symbol: str = "TEST",
    side: Side = Side.LONG,
    quantity: int = 100,
    order_type: OrderType = OrderType.MARKET,
    limit_price: Decimal | None = None,
    quality_score: float = 0.5,
    requested_notional: int = 50_000,
    model_version: str = "model-1",
) -> AllocationCandidate:
    signal = Signal(
        strategy_id=strategy_id,
        strategy_version="1",
        symbol=symbol,
        timestamp=timestamp - timedelta(minutes=5),
        side=side,
    )
    order = OrderIntent(
        signal=signal,
        timestamp=timestamp,
        quantity=quantity,
        requested_notional=requested_notional,
        order_type=order_type,
        limit_price=limit_price,
    )
    score = MLScore(
        model_version=model_version,
        quality_score=quality_score,
        calibrated_probability=0.5,
        predicted_net_return=0.01,
        recommended_notional=requested_notional,
    )
    return AllocationCandidate(order_intent=order, ml_score=score)


def make_request(timestamp: datetime, **candidate_updates) -> BacktestTradeRequest:
    return BacktestTradeRequest(candidate=make_candidate(timestamp, **candidate_updates))


def make_config(start: datetime, end: datetime, **updates) -> BacktestConfig:
    values = {
        "run_id": "run-1",
        "git_commit": "deadbeef",
        "window_start": start,
        "window_end": end,
        "brokerage_plan": BrokeragePlan.PLUS,
    }
    values.update(updates)
    return BacktestConfig(**values)


class StubMarginProvider:
    def __init__(self, requirements: dict[str, Decimal]) -> None:
        self.requirements = requirements
        self.calls: list[tuple[str, Decimal, Decimal]] = []

    def quote(
        self,
        candidate: AllocationCandidate,
        state: PortfolioState,
    ) -> MarginRequirementQuote:
        strategy_id = candidate.order_intent.signal.strategy_id
        self.calls.append((strategy_id, state.capital_limit, state.available_margin))
        return MarginRequirementQuote(
            provider_id="test-margin",
            required_margin=self.requirements[strategy_id],
        )


class CountingMarketDataStore(ParquetMarketDataStore):
    def __init__(self, config: MarketDataConfig) -> None:
        super().__init__(config)
        self.loads: list[tuple[str, datetime, datetime]] = []

    def load_candles(self, symbols, start: datetime, end: datetime):
        assert isinstance(symbols, str)
        self.loads.append((symbols, start, end))
        return super().load_candles(symbols, start, end)


class TrackingExecutionSimulator(HistoricalExecutionSimulator):
    def __init__(self) -> None:
        super().__init__()
        self.entry_candle_starts: list[datetime] = []
        self.protective_candle_starts: list[datetime] = []
        self.market_exit_calls: list[tuple[datetime, ExitReason]] = []
        self.market_exit_candle_starts: list[list[datetime]] = []

    def fill_entry_order(self, order, candles):
        self.entry_candle_starts.extend(candles["timestamp"].to_list())
        return super().fill_entry_order(order, candles)

    def fill_protective_exit(self, **kwargs):
        self.protective_candle_starts.extend(
            kwargs["candles"]["timestamp"].to_list()
        )
        return super().fill_protective_exit(**kwargs)

    def fill_market_exit(self, **kwargs):
        self.market_exit_calls.append(
            (kwargs["requested_at"], kwargs["exit_reason"])
        )
        self.market_exit_candle_starts.append(
            kwargs["candles"]["timestamp"].to_list()
        )
        return super().fill_market_exit(**kwargs)


def run_backtest(
    store: ParquetMarketDataStore,
    requests: list[BacktestTradeRequest],
    provider: StubMarginProvider,
    config: BacktestConfig,
    *,
    simulator: HistoricalExecutionSimulator | None = None,
):
    return HistoricalBacktester(store, provider, simulator).run(config, requests)


@pytest.mark.parametrize(
    ("side", "exit_price", "profitable"),
    [
        (Side.LONG, 110.0, True),
        (Side.LONG, 90.0, False),
        (Side.SHORT, 90.0, True),
        (Side.SHORT, 110.0, False),
    ],
)
def test_completed_long_and_short_profit_and_loss(
    tmp_path: Path,
    side: Side,
    exit_price: float,
    profitable: bool,
) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 15, 25), exit_price, exit_price + 1, exit_price - 1, exit_price),
            ]
        },
    )
    request = make_request(at(*day, 9, 20), side=side)

    result = run_backtest(
        store,
        [request],
        StubMarginProvider({"strategy": Decimal("50000")}),
        make_config(at(*day, 9, 15), at(*day, 15, 30)),
    )

    record = result.actual_trade_records[0]
    assert record.trade.exit_reason is ExitReason.TIME_EXIT
    assert (record.trade.gross_pnl > 0) is profitable
    assert record.trade.net_pnl == (
        record.trade.gross_pnl - record.round_trip_cost_breakdown.total
    )
    assert result.ending_capital == Decimal("100000") + record.trade.net_pnl
    assert record.trade.signal.status is SignalStatus.EXECUTED
    assert request.candidate.order_intent.signal.status is SignalStatus.GENERATED


@pytest.mark.parametrize(
    ("protective", "high", "low", "expected_reason"),
    [
        (
            ProtectiveExitSpec(stop_price=Decimal("95")),
            101.0,
            94.0,
            ExitReason.STOP_LOSS,
        ),
        (
            ProtectiveExitSpec(target_price=Decimal("105")),
            106.0,
            99.0,
            ExitReason.TARGET_REACHED,
        ),
    ],
)
def test_protective_exit_reasons_are_preserved(
    tmp_path: Path,
    protective: ProtectiveExitSpec,
    high: float,
    low: float,
    expected_reason: ExitReason,
) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, high, low, 100.0),
                (at(*day, 15, 25), 100.0, 101.0, 99.0, 100.0),
            ]
        },
    )
    request = BacktestTradeRequest(
        candidate=make_candidate(at(*day, 9, 20)),
        protective_exit=protective,
    )

    result = run_backtest(
        store,
        [request],
        StubMarginProvider({"strategy": Decimal("1")}),
        make_config(at(*day, 9, 15), at(*day, 15, 30)),
    )

    assert result.actual_trade_records[0].trade.exit_reason is expected_reason
    assert result.actual_trade_records[0].trade.exit_fill.timestamp == at(*day, 9, 25)


def test_strategy_exit_precedes_forced_time_exit(tmp_path: Path) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 10, 0), 105.0, 106.0, 104.0, 105.0),
                (at(*day, 15, 25), 110.0, 111.0, 109.0, 110.0),
            ]
        },
    )
    request = BacktestTradeRequest(
        candidate=make_candidate(at(*day, 9, 20)),
        strategy_exit_at=at(*day, 10, 0),
    )

    result = run_backtest(
        store,
        [request],
        StubMarginProvider({"strategy": Decimal("1")}),
        make_config(at(*day, 9, 15), at(*day, 15, 30)),
    )

    trade = result.actual_trade_records[0].trade
    assert trade.exit_reason is ExitReason.STRATEGY_EXIT
    assert trade.exit_fill.timestamp == at(*day, 10, 0)


def test_protective_exit_wins_same_fill_timestamp_tie(tmp_path: Path) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 9, 55), 100.0, 106.0, 99.0, 104.0),
                (at(*day, 10, 0), 104.0, 105.0, 103.0, 104.0),
                (at(*day, 15, 25), 104.0, 105.0, 103.0, 104.0),
            ]
        },
    )
    request = BacktestTradeRequest(
        candidate=make_candidate(at(*day, 9, 20)),
        protective_exit=ProtectiveExitSpec(target_price=Decimal("105")),
        strategy_exit_at=at(*day, 10, 0),
    )

    result = run_backtest(
        store,
        [request],
        StubMarginProvider({"strategy": Decimal("1")}),
        make_config(at(*day, 9, 15), at(*day, 15, 30)),
    )

    assert result.actual_trade_records[0].trade.exit_reason is ExitReason.TARGET_REACHED


def test_strategy_exit_wins_same_timestamp_as_time_exit(tmp_path: Path) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 15, 25), 101.0, 102.0, 100.0, 101.0),
            ]
        },
    )
    request = BacktestTradeRequest(
        candidate=make_candidate(at(*day, 9, 20)),
        strategy_exit_at=at(*day, 15, 25),
    )

    result = run_backtest(
        store,
        [request],
        StubMarginProvider({"strategy": Decimal("1")}),
        make_config(at(*day, 9, 15), at(*day, 15, 30)),
    )

    assert result.actual_trade_records[0].trade.exit_reason is ExitReason.STRATEGY_EXIT


def test_slippage_and_costs_are_each_applied_once(tmp_path: Path) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 15, 25), 110.0, 111.0, 109.0, 110.0),
            ]
        },
    )
    simulator = HistoricalExecutionSimulator(
        slippage_model=FixedBasisPointsSlippage(Decimal("10"))
    )

    result = run_backtest(
        store,
        [make_request(at(*day, 9, 20))],
        StubMarginProvider({"strategy": Decimal("1")}),
        make_config(at(*day, 9, 15), at(*day, 15, 30)),
        simulator=simulator,
    )

    record = result.actual_trade_records[0]
    assert record.trade.entry_fill.price == Decimal("100.100")
    assert record.trade.exit_fill.price == Decimal("109.890")
    assert record.round_trip_cost_breakdown.entry.turnover == Decimal("10010.000")
    assert record.trade.net_pnl == record.trade.gross_pnl - record.trade.total_costs


@pytest.mark.parametrize(
    ("order_time", "order_type", "limit_price"),
    [
        (at(2025, 1, 2, 15, 25), OrderType.MARKET, None),
        (at(2025, 1, 2, 15, 20), OrderType.LIMIT, Decimal("90")),
    ],
)
def test_entry_at_cutoff_or_unfilled_limit_has_terminal_no_fill(
    tmp_path: Path,
    order_time: datetime,
    order_type: OrderType,
    limit_price: Decimal | None,
) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 15, 20), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 15, 25), 100.0, 101.0, 99.0, 100.0),
            ]
        },
    )
    request = make_request(
        order_time,
        order_type=order_type,
        limit_price=limit_price,
    )

    result = run_backtest(
        store,
        [request],
        StubMarginProvider({"strategy": Decimal("100000")}),
        make_config(at(*day, 9, 15), at(*day, 15, 30)),
    )

    assert result.request_results[0].outcome is (
        BacktestRequestOutcome.ALLOCATED_ENTRY_NOT_FILLED
    )
    assert result.actual_trade_records == ()
    assert result.ending_portfolio_state is not None
    assert result.ending_portfolio_state.reserved_margin == 0
    assert request.candidate.order_intent.signal.status is SignalStatus.GENERATED


def test_entry_never_rolls_to_next_trading_day(tmp_path: Path) -> None:
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(2025, 1, 2, 15, 20), 100.0, 101.0, 99.0, 100.0),
                (at(2025, 1, 2, 15, 25), 100.0, 101.0, 99.0, 100.0),
                (at(2025, 1, 3, 9, 15), 80.0, 91.0, 79.0, 85.0),
            ]
        },
    )
    request = make_request(
        at(2025, 1, 2, 15, 20),
        order_type=OrderType.LIMIT,
        limit_price=Decimal("90"),
    )

    result = run_backtest(
        store,
        [request],
        StubMarginProvider({"strategy": Decimal("1")}),
        make_config(at(2025, 1, 2, 9, 15), at(2025, 1, 4, 0, 0)),
    )

    assert result.request_results[0].outcome is (
        BacktestRequestOutcome.ALLOCATED_ENTRY_NOT_FILLED
    )


def test_unfilled_reservation_releases_at_cutoff_for_later_day(tmp_path: Path) -> None:
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(2025, 1, 2, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(2025, 1, 2, 15, 25), 100.0, 101.0, 99.0, 100.0),
                (at(2025, 1, 3, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(2025, 1, 3, 15, 25), 101.0, 102.0, 100.0, 101.0),
            ]
        },
    )
    first = make_request(
        at(2025, 1, 2, 9, 20),
        strategy_id="first",
        order_type=OrderType.LIMIT,
        limit_price=Decimal("90"),
    )
    second = make_request(at(2025, 1, 3, 9, 20), strategy_id="second")
    provider = StubMarginProvider(
        {"first": Decimal("100000"), "second": Decimal("100000")}
    )

    result = run_backtest(
        store,
        [second, first],
        provider,
        make_config(at(2025, 1, 2, 9, 15), at(2025, 1, 4, 0, 0)),
    )

    assert [item.outcome for item in result.request_results] == [
        BacktestRequestOutcome.ALLOCATED_ENTRY_NOT_FILLED,
        BacktestRequestOutcome.COMPLETED_ACTUAL,
    ]
    assert provider.calls[1][2] == Decimal("100000")


@pytest.mark.parametrize("eventual_exit_price", [150.0, 50.0])
def test_active_reservation_blocks_intervening_signal_and_future_pnl_is_irrelevant(
    tmp_path: Path,
    eventual_exit_price: float,
) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 9, 45), 100.0, 101.0, 99.0, 100.0),
                (
                    at(*day, 10, 30),
                    eventual_exit_price,
                    eventual_exit_price + 1,
                    eventual_exit_price - 1,
                    eventual_exit_price,
                ),
                (
                    at(*day, 15, 25),
                    eventual_exit_price,
                    eventual_exit_price + 1,
                    eventual_exit_price - 1,
                    eventual_exit_price,
                ),
            ]
        },
    )
    first = BacktestTradeRequest(
        candidate=make_candidate(at(*day, 9, 20), strategy_id="first"),
        strategy_exit_at=at(*day, 10, 30),
    )
    competing = BacktestTradeRequest(
        candidate=make_candidate(at(*day, 9, 45), strategy_id="competing"),
        strategy_exit_at=at(*day, 10, 30),
    )
    provider = StubMarginProvider(
        {"first": Decimal("100000"), "competing": Decimal("100000")}
    )

    result = run_backtest(
        store,
        [competing, first],
        provider,
        make_config(at(*day, 9, 15), at(*day, 15, 30)),
    )

    by_strategy = {
        item.request.candidate.order_intent.signal.strategy_id: item
        for item in result.request_results
    }
    assert by_strategy["first"].outcome is BacktestRequestOutcome.COMPLETED_ACTUAL
    assert by_strategy["competing"].outcome is BacktestRequestOutcome.COMPLETED_SHADOW
    assert provider.calls[1] == ("competing", Decimal("100000"), Decimal("0"))


def test_exact_time_release_cannot_fund_same_time_allocation(tmp_path: Path) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 9, 45), 101.0, 102.0, 100.0, 101.0),
                (at(*day, 15, 25), 102.0, 103.0, 101.0, 102.0),
            ]
        },
    )
    first = BacktestTradeRequest(
        candidate=make_candidate(at(*day, 9, 20), strategy_id="first"),
        strategy_exit_at=at(*day, 9, 45),
    )
    same_time = BacktestTradeRequest(
        candidate=make_candidate(at(*day, 9, 45), strategy_id="second"),
    )
    provider = StubMarginProvider(
        {"first": Decimal("100000"), "second": Decimal("100000")}
    )

    result = run_backtest(
        store,
        [same_time, first],
        provider,
        make_config(at(*day, 9, 15), at(*day, 15, 30)),
    )

    by_strategy = {
        item.request.candidate.order_intent.signal.strategy_id: item.outcome
        for item in result.request_results
    }
    assert by_strategy == {
        "first": BacktestRequestOutcome.COMPLETED_ACTUAL,
        "second": BacktestRequestOutcome.COMPLETED_SHADOW,
    }
    assert provider.calls[1][2] == Decimal("0")


@pytest.mark.parametrize("first_exit_price", [110.0, 90.0])
def test_release_and_realized_pnl_reaches_genuinely_later_allocation(
    tmp_path: Path,
    first_exit_price: float,
) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (
                    at(*day, 9, 45),
                    first_exit_price,
                    first_exit_price + 1,
                    first_exit_price - 1,
                    first_exit_price,
                ),
                (at(*day, 9, 50), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 15, 25), 101.0, 102.0, 100.0, 101.0),
            ]
        },
    )
    first = BacktestTradeRequest(
        candidate=make_candidate(at(*day, 9, 20), strategy_id="first"),
        strategy_exit_at=at(*day, 9, 45),
    )
    later = make_request(at(*day, 9, 50), strategy_id="later")
    provider = StubMarginProvider(
        {"first": Decimal("100000"), "later": Decimal("1")}
    )

    result = run_backtest(
        store,
        [later, first],
        provider,
        make_config(at(*day, 9, 15), at(*day, 15, 30)),
    )

    assert all(
        item.outcome is BacktestRequestOutcome.COMPLETED_ACTUAL
        for item in result.request_results
    )
    first_trade = next(
        record.trade
        for record in result.actual_trade_records
        if record.trade.signal.strategy_id == "first"
    )
    assert provider.calls[1][1] == Decimal("100000") + first_trade.net_pnl


def test_daily_candles_are_cached_and_sources_remain_unchanged(tmp_path: Path) -> None:
    day = (2025, 1, 2)
    write_symbol(
        tmp_path,
        "TEST",
        [
            (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
            (at(*day, 9, 45), 100.0, 102.0, 98.0, 101.0),
            (at(*day, 15, 25), 101.0, 102.0, 100.0, 101.0),
        ],
    )
    store = CountingMarketDataStore(MarketDataConfig(dataset_path=tmp_path))
    before = store.load_candles(
        "TEST",
        at(*day, 0, 0),
        at(2025, 1, 3, 0, 0),
    )
    store.loads.clear()
    first = make_request(at(*day, 9, 20), strategy_id="first")
    second = make_request(at(*day, 9, 45), strategy_id="second")
    requests_before = [first.model_dump(), second.model_dump()]

    run_backtest(
        store,
        [second, first],
        StubMarginProvider({"first": Decimal("60000"), "second": Decimal("60000")}),
        make_config(at(*day, 9, 15), at(*day, 15, 30)),
    )
    after = store.load_candles(
        "TEST",
        at(*day, 0, 0),
        at(2025, 1, 3, 0, 0),
    )

    assert len(store.loads) == 2  # one run load plus this explicit post-run read
    assert before.equals(after)
    assert [first.model_dump(), second.model_dump()] == requests_before


@pytest.mark.parametrize("shadow_exit", [120.0, 80.0])
def test_shadow_profit_or_loss_has_zero_capital_impact(
    tmp_path: Path,
    shadow_exit: float,
) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 9, 45), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 10, 30), shadow_exit, shadow_exit + 1, shadow_exit - 1, shadow_exit),
                (at(*day, 15, 25), 100.0, 101.0, 99.0, 100.0),
            ]
        },
    )
    actual = make_request(at(*day, 9, 20), strategy_id="actual")
    shadow = BacktestTradeRequest(
        candidate=make_candidate(at(*day, 9, 45), strategy_id="shadow"),
        strategy_exit_at=at(*day, 10, 30),
    )

    result = run_backtest(
        store,
        [shadow, actual],
        StubMarginProvider(
            {"actual": Decimal("100000"), "shadow": Decimal("100000")}
        ),
        make_config(at(*day, 9, 15), at(*day, 15, 30)),
    )

    actual_trade = result.actual_trade_records[0].trade
    shadow_trade = result.shadow_trade_records[0].trade
    assert shadow_trade.is_shadow
    assert shadow_trade.signal.status is SignalStatus.CAPACITY_REJECTED
    assert shadow_trade.mfe_return >= 0
    assert shadow_trade.mae_return <= 0
    assert result.ending_capital == Decimal("100000") + actual_trade.net_pnl


def test_capacity_rejected_limit_no_fill_is_auditable(tmp_path: Path) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 15, 25), 100.0, 101.0, 99.0, 100.0),
            ]
        },
    )
    actual = make_request(at(*day, 9, 20), strategy_id="actual", quality_score=1.0)
    shadow = make_request(
        at(*day, 9, 20),
        strategy_id="shadow",
        quality_score=0.0,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("90"),
    )

    result = run_backtest(
        store,
        [shadow, actual],
        StubMarginProvider(
            {"actual": Decimal("100000"), "shadow": Decimal("100000")}
        ),
        make_config(at(*day, 9, 15), at(*day, 15, 30)),
    )

    shadow_result = next(
        item
        for item in result.request_results
        if item.request.candidate.order_intent.signal.strategy_id == "shadow"
    )
    assert shadow_result.outcome is BacktestRequestOutcome.SHADOW_ENTRY_NOT_FILLED
    assert shadow_result.trade_record is None


def test_mfe_mae_use_only_bar_starts_inside_holding_interval(tmp_path: Path) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 15), 100.0, 1000.0, 1.0, 100.0),
                (at(*day, 9, 20), 100.0, 110.0, 90.0, 100.0),
                (at(*day, 9, 25), 105.0, 1000.0, 1.0, 105.0),
                (at(*day, 15, 25), 105.0, 106.0, 104.0, 105.0),
            ]
        },
    )
    request = BacktestTradeRequest(
        candidate=make_candidate(at(*day, 9, 20)),
        strategy_exit_at=at(*day, 9, 25),
    )

    result = run_backtest(
        store,
        [request],
        StubMarginProvider({"strategy": Decimal("1")}),
        make_config(at(*day, 9, 15), at(*day, 15, 30)),
    )

    trade = result.actual_trade_records[0].trade
    assert trade.mfe_return == Decimal("0.1")
    assert trade.mae_return == Decimal("-0.1")


def test_missing_same_day_exit_aborts_run_without_partial_result(tmp_path: Path) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(2025, 1, 3, 9, 15), 101.0, 102.0, 100.0, 101.0),
            ]
        },
    )

    with pytest.raises(BacktestIntegrityError, match="mandatory same-day exit"):
        run_backtest(
            store,
            [make_request(at(*day, 9, 20))],
            StubMarginProvider({"strategy": Decimal("1")}),
            make_config(at(*day, 9, 15), at(*day, 15, 30)),
        )


def test_fixed_policy_is_identical_for_2016_and_2026_trades(tmp_path: Path) -> None:
    store = make_store(
        tmp_path,
        {
            "OLD": [
                (at(2016, 1, 2, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(2016, 1, 2, 15, 25), 101.0, 102.0, 100.0, 101.0),
            ],
            "NEW": [
                (at(2026, 1, 2, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(2026, 1, 2, 15, 25), 101.0, 102.0, 100.0, 101.0),
            ],
        },
    )
    old = make_request(at(2016, 1, 2, 9, 20), strategy_id="old", symbol="OLD")
    new = make_request(at(2026, 1, 2, 9, 20), strategy_id="new", symbol="NEW")

    result = run_backtest(
        store,
        [new, old],
        StubMarginProvider({"old": Decimal("1"), "new": Decimal("1")}),
        make_config(at(2016, 1, 1, 0, 0), at(2027, 1, 1, 0, 0)),
    )

    assert {record.cost_policy_id for record in result.actual_trade_records} == {
        result.cost_policy_id
    }
    assert (
        result.actual_trade_records[0].round_trip_cost_breakdown
        == result.actual_trade_records[1].round_trip_cost_breakdown
    )


def test_reversed_input_produces_identical_deterministic_result(tmp_path: Path) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 15, 25), 101.0, 102.0, 100.0, 101.0),
            ]
        },
    )
    alpha = make_request(at(*day, 9, 20), strategy_id="alpha")
    beta = make_request(at(*day, 9, 20), strategy_id="beta")
    config = make_config(at(*day, 9, 15), at(*day, 15, 30))

    forward = run_backtest(
        store,
        [alpha, beta],
        StubMarginProvider({"alpha": Decimal("60000"), "beta": Decimal("60000")}),
        config,
    )
    reversed_result = run_backtest(
        store,
        [beta, alpha],
        StubMarginProvider({"alpha": Decimal("60000"), "beta": Decimal("60000")}),
        config,
    )

    assert forward == reversed_result


def test_capital_exhaustion_stops_future_allocations(tmp_path: Path) -> None:
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(2025, 1, 2, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(2025, 1, 2, 15, 25), 1.0, 2.0, 0.5, 1.0),
                (at(2025, 1, 3, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(2025, 1, 3, 15, 25), 101.0, 102.0, 100.0, 101.0),
            ]
        },
    )
    loss = make_request(
        at(2025, 1, 2, 9, 20),
        strategy_id="loss",
        quantity=20,
    )
    future = make_request(at(2025, 1, 3, 9, 20), strategy_id="future")

    result = run_backtest(
        store,
        [future, loss],
        StubMarginProvider({"loss": Decimal("100"), "future": Decimal("1")}),
        make_config(
            at(2025, 1, 2, 9, 15),
            at(2025, 1, 4, 0, 0),
            initial_capital=Decimal("1000"),
        ),
    )

    assert result.capital_exhausted
    assert result.ending_capital <= 0
    assert result.ending_portfolio_state is None
    assert any(
        item.outcome is BacktestRequestOutcome.CAPITAL_EXHAUSTED
        for item in result.request_results
    )


def test_request_and_config_validation() -> None:
    timestamp = at(2025, 1, 2, 9, 20)
    candidate = make_candidate(timestamp)

    with pytest.raises(ValidationError, match="timezone-aware"):
        BacktestTradeRequest(
            candidate=candidate,
            strategy_exit_at=datetime(2025, 1, 2, 10, 0),
        )
    with pytest.raises(ValidationError, match="trading date"):
        BacktestTradeRequest(
            candidate=candidate,
            strategy_exit_at=at(2025, 1, 3, 10, 0),
        )
    with pytest.raises(ValidationError, match="earlier than"):
        make_config(timestamp, timestamp)


@pytest.mark.parametrize(
    ("first_fill_at", "strategy_exit_at"),
    [
        (at(2025, 1, 2, 10, 0), at(2025, 1, 2, 9, 45)),
        (at(2025, 1, 2, 9, 45), at(2025, 1, 2, 9, 45)),
    ],
    ids=["fill-after-strategy-exit", "fill-exactly-at-strategy-exit"],
)
def test_pending_limit_cannot_enter_at_or_after_strategy_exit(
    tmp_path: Path,
    first_fill_at: datetime,
    strategy_exit_at: datetime,
) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (first_fill_at, 90.0, 91.0, 89.0, 90.0),
                (at(*day, 15, 25), 100.0, 101.0, 99.0, 100.0),
            ]
        },
    )
    request = BacktestTradeRequest(
        candidate=make_candidate(
            at(*day, 9, 20),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("90"),
        ),
        strategy_exit_at=strategy_exit_at,
    )

    result = run_backtest(
        store,
        [request],
        StubMarginProvider({"strategy": Decimal("100000")}),
        make_config(at(*day, 9, 15), at(*day, 15, 30)),
    )

    terminal = result.request_results[0]
    assert terminal.outcome is BacktestRequestOutcome.ALLOCATED_ENTRY_NOT_FILLED
    assert terminal.terminal_at == strategy_exit_at
    assert terminal.trade_record is None
    assert result.actual_trade_records == ()


def test_cancelled_pending_reservation_blocks_exact_time_and_funds_later_batch(
    tmp_path: Path,
) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 9, 45), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 9, 50), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 15, 25), 101.0, 102.0, 100.0, 101.0),
            ]
        },
    )
    cancelled = BacktestTradeRequest(
        candidate=make_candidate(
            at(*day, 9, 20),
            strategy_id="cancelled",
            order_type=OrderType.LIMIT,
            limit_price=Decimal("90"),
        ),
        strategy_exit_at=at(*day, 9, 45),
    )
    exact_time = make_request(at(*day, 9, 45), strategy_id="exact")
    later = make_request(at(*day, 9, 50), strategy_id="later")
    provider = StubMarginProvider(
        {
            "cancelled": Decimal("100000"),
            "exact": Decimal("100000"),
            "later": Decimal("100000"),
        }
    )

    result = run_backtest(
        store,
        [later, exact_time, cancelled],
        provider,
        make_config(at(*day, 9, 15), at(*day, 15, 30)),
    )

    by_strategy = {
        item.request.candidate.order_intent.signal.strategy_id: item.outcome
        for item in result.request_results
    }
    assert by_strategy == {
        "cancelled": BacktestRequestOutcome.ALLOCATED_ENTRY_NOT_FILLED,
        "exact": BacktestRequestOutcome.COMPLETED_SHADOW,
        "later": BacktestRequestOutcome.COMPLETED_ACTUAL,
    }
    assert provider.calls[1][2] == Decimal("0")
    assert provider.calls[2][2] == Decimal("100000")


def test_capacity_rejected_pending_limit_obeys_strategy_exit_deadline(
    tmp_path: Path,
) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 10, 0), 90.0, 91.0, 89.0, 90.0),
                (at(*day, 15, 25), 100.0, 101.0, 99.0, 100.0),
            ]
        },
    )
    allocated = make_request(
        at(*day, 9, 20),
        strategy_id="allocated",
        quality_score=1.0,
    )
    shadow = BacktestTradeRequest(
        candidate=make_candidate(
            at(*day, 9, 20),
            strategy_id="shadow",
            quality_score=0.0,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("90"),
        ),
        strategy_exit_at=at(*day, 9, 45),
    )

    result = run_backtest(
        store,
        [shadow, allocated],
        StubMarginProvider(
            {"allocated": Decimal("100000"), "shadow": Decimal("100000")}
        ),
        make_config(at(*day, 9, 15), at(*day, 15, 30)),
    )

    shadow_result = next(
        item
        for item in result.request_results
        if item.request.candidate.order_intent.signal.strategy_id == "shadow"
    )
    assert shadow_result.outcome is BacktestRequestOutcome.SHADOW_ENTRY_NOT_FILLED
    assert shadow_result.terminal_at == at(*day, 9, 45)
    assert shadow_result.trade_record is None
    assert result.shadow_trade_records == ()


def test_forced_cutoff_is_entry_deadline_when_strategy_exit_is_later(
    tmp_path: Path,
) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 15, 25), 90.0, 91.0, 89.0, 90.0),
            ]
        },
    )
    request = BacktestTradeRequest(
        candidate=make_candidate(
            at(*day, 9, 20),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("90"),
        ),
        strategy_exit_at=at(*day, 15, 30),
    )

    result = run_backtest(
        store,
        [request],
        StubMarginProvider({"strategy": Decimal("1")}),
        make_config(at(*day, 9, 15), at(*day, 15, 35)),
    )

    terminal = result.request_results[0]
    assert terminal.outcome is BacktestRequestOutcome.ALLOCATED_ENTRY_NOT_FILLED
    assert terminal.terminal_at == at(*day, 15, 25)


@pytest.mark.parametrize(
    ("loss_strategy", "win_strategy"),
    [("a-loss", "z-win"), ("z-loss", "a-win")],
    ids=["loss-identity-first", "win-identity-first"],
)
def test_same_timestamp_realizations_are_atomic_and_identity_order_independent(
    tmp_path: Path,
    loss_strategy: str,
    win_strategy: str,
) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 10, 0), 200.0, 201.0, 199.0, 200.0),
                (at(*day, 10, 5), 200.0, 201.0, 199.0, 200.0),
                (at(*day, 15, 25), 201.0, 202.0, 200.0, 201.0),
            ]
        },
    )
    loss = BacktestTradeRequest(
        candidate=make_candidate(
            at(*day, 9, 20),
            strategy_id=loss_strategy,
            side=Side.SHORT,
            quantity=2,
        ),
        strategy_exit_at=at(*day, 10, 0),
    )
    win = BacktestTradeRequest(
        candidate=make_candidate(
            at(*day, 9, 20),
            strategy_id=win_strategy,
            side=Side.LONG,
            quantity=2,
        ),
        strategy_exit_at=at(*day, 10, 0),
    )
    later = make_request(at(*day, 10, 5), strategy_id="later", quantity=1)
    provider = StubMarginProvider(
        {
            loss_strategy: Decimal("50"),
            win_strategy: Decimal("50"),
            "later": Decimal("1"),
        }
    )

    result = run_backtest(
        store,
        [later, win, loss],
        provider,
        make_config(
            at(*day, 9, 15),
            at(*day, 15, 30),
            initial_capital=Decimal("100"),
        ),
    )

    realized = [
        record.trade
        for record in result.actual_trade_records
        if record.trade.signal.strategy_id in {loss_strategy, win_strategy}
    ]
    expected_post_group = Decimal("100") + sum(
        (trade.net_pnl for trade in realized),
        start=Decimal("0"),
    )
    assert min(trade.net_pnl for trade in realized) + Decimal("100") <= 0
    assert expected_post_group > 0
    assert not result.capital_exhausted
    assert provider.calls[2][1] == expected_post_group
    assert next(
        item.outcome
        for item in result.request_results
        if item.request.candidate.order_intent.signal.strategy_id == "later"
    ) is BacktestRequestOutcome.COMPLETED_ACTUAL


def test_reversing_realization_identity_order_preserves_economic_result(
    tmp_path: Path,
) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 10, 0), 200.0, 201.0, 199.0, 200.0),
                (at(*day, 10, 5), 200.0, 201.0, 199.0, 200.0),
                (at(*day, 15, 25), 201.0, 202.0, 200.0, 201.0),
            ]
        },
    )
    summaries = []
    for loss_strategy, win_strategy in [("a", "z"), ("z", "a")]:
        loss = BacktestTradeRequest(
            candidate=make_candidate(
                at(*day, 9, 20),
                strategy_id=loss_strategy,
                side=Side.SHORT,
                quantity=2,
            ),
            strategy_exit_at=at(*day, 10, 0),
        )
        win = BacktestTradeRequest(
            candidate=make_candidate(
                at(*day, 9, 20),
                strategy_id=win_strategy,
                side=Side.LONG,
                quantity=2,
            ),
            strategy_exit_at=at(*day, 10, 0),
        )
        later = make_request(at(*day, 10, 5), strategy_id="later", quantity=1)
        provider = StubMarginProvider(
            {
                loss_strategy: Decimal("50"),
                win_strategy: Decimal("50"),
                "later": Decimal("1"),
            }
        )
        result = run_backtest(
            store,
            [later, win, loss],
            provider,
            make_config(
                at(*day, 9, 15),
                at(*day, 15, 30),
                initial_capital=Decimal("100"),
            ),
        )
        later_outcome = next(
            item.outcome
            for item in result.request_results
            if item.request.candidate.order_intent.signal.strategy_id == "later"
        )
        summaries.append(
            (
                result.capital_exhausted,
                result.ending_capital,
                later_outcome,
                provider.calls[2][1],
            )
        )

    assert summaries[0] == summaries[1]


def test_same_timestamp_group_aggregate_can_exhaust_capital(tmp_path: Path) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 10, 0), 1.0, 2.0, 0.5, 1.0),
                (at(*day, 10, 5), 1.0, 2.0, 0.5, 1.0),
                (at(*day, 15, 25), 1.0, 2.0, 0.5, 1.0),
            ]
        },
    )
    first = BacktestTradeRequest(
        candidate=make_candidate(
            at(*day, 9, 20), strategy_id="first", quantity=1
        ),
        strategy_exit_at=at(*day, 10, 0),
    )
    second = BacktestTradeRequest(
        candidate=make_candidate(
            at(*day, 9, 20), strategy_id="second", quantity=1
        ),
        strategy_exit_at=at(*day, 10, 0),
    )
    later = make_request(at(*day, 10, 5), strategy_id="later", quantity=1)

    result = run_backtest(
        store,
        [later, second, first],
        StubMarginProvider(
            {"first": Decimal("50"), "second": Decimal("50"), "later": Decimal("1")}
        ),
        make_config(
            at(*day, 9, 15),
            at(*day, 15, 30),
            initial_capital=Decimal("100"),
        ),
    )

    assert result.capital_exhausted
    assert result.ending_capital <= 0
    assert next(
        item.outcome
        for item in result.request_results
        if item.request.candidate.order_intent.signal.strategy_id == "later"
    ) is BacktestRequestOutcome.CAPITAL_EXHAUSTED


def test_later_open_trade_pnl_is_retained_without_resuming_after_exhaustion(
    tmp_path: Path,
) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 10, 0), 1.0, 2.0, 0.5, 1.0),
                (at(*day, 10, 30), 1.0, 2.0, 0.5, 1.0),
                (at(*day, 10, 35), 1.0, 2.0, 0.5, 1.0),
                (at(*day, 15, 25), 1.0, 2.0, 0.5, 1.0),
            ]
        },
    )
    losses = [
        BacktestTradeRequest(
            candidate=make_candidate(
                at(*day, 9, 20),
                strategy_id=strategy_id,
                side=Side.LONG,
                quantity=1,
            ),
            strategy_exit_at=at(*day, 10, 0),
        )
        for strategy_id in ("loss-a", "loss-b")
    ]
    recovery = BacktestTradeRequest(
        candidate=make_candidate(
            at(*day, 9, 20),
            strategy_id="recovery",
            side=Side.SHORT,
            quantity=3,
        ),
        strategy_exit_at=at(*day, 10, 30),
    )
    later = make_request(at(*day, 10, 35), strategy_id="later", quantity=1)

    result = run_backtest(
        store,
        [later, recovery, *losses],
        StubMarginProvider(
            {
                "loss-a": Decimal("30"),
                "loss-b": Decimal("30"),
                "recovery": Decimal("40"),
                "later": Decimal("1"),
            }
        ),
        make_config(
            at(*day, 9, 15),
            at(*day, 15, 30),
            initial_capital=Decimal("100"),
        ),
    )

    assert result.capital_exhausted
    assert result.ending_capital > 0
    assert next(
        item.outcome
        for item in result.request_results
        if item.request.candidate.order_intent.signal.strategy_id == "later"
    ) is BacktestRequestOutcome.CAPITAL_EXHAUSTED
    assert any(
        record.trade.signal.strategy_id == "recovery"
        for record in result.actual_trade_records
    )


def test_same_time_allocation_cannot_use_atomic_realization_group(tmp_path: Path) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 10, 0), 110.0, 111.0, 109.0, 110.0),
                (at(*day, 15, 25), 111.0, 112.0, 110.0, 111.0),
            ]
        },
    )
    first = BacktestTradeRequest(
        candidate=make_candidate(at(*day, 9, 20), strategy_id="first"),
        strategy_exit_at=at(*day, 10, 0),
    )
    second = BacktestTradeRequest(
        candidate=make_candidate(at(*day, 9, 20), strategy_id="second"),
        strategy_exit_at=at(*day, 10, 0),
    )
    same_time = make_request(at(*day, 10, 0), strategy_id="same-time")

    result = run_backtest(
        store,
        [same_time, second, first],
        StubMarginProvider(
            {
                "first": Decimal("50000"),
                "second": Decimal("50000"),
                "same-time": Decimal("1"),
            }
        ),
        make_config(at(*day, 9, 15), at(*day, 15, 30)),
    )

    assert next(
        item.outcome
        for item in result.request_results
        if item.request.candidate.order_intent.signal.strategy_id == "same-time"
    ) is BacktestRequestOutcome.COMPLETED_SHADOW


@pytest.fixture
def completed_audit_objects(tmp_path: Path):
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 15, 25), 101.0, 102.0, 100.0, 101.0),
            ]
        },
    )
    result = run_backtest(
        store,
        [make_request(at(*day, 9, 20))],
        StubMarginProvider({"strategy": Decimal("1")}),
        make_config(at(*day, 9, 15), at(*day, 15, 30)),
    )
    return result.actual_trade_records[0], result.request_results[0]


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("total_costs", "total_costs"),
        ("cost_policy_id", "cost policy ID"),
        ("ml_score", "MLScore"),
        ("target_notional", "target_notional"),
    ],
)
def test_trade_audit_record_rejects_tampered_cost_or_candidate_provenance(
    completed_audit_objects,
    field: str,
    message: str,
) -> None:
    record, _ = completed_audit_objects
    values = {
        "trade": record.trade,
        "round_trip_cost_breakdown": record.round_trip_cost_breakdown,
        "cost_policy_id": record.cost_policy_id,
        "allocation_identity": record.allocation_identity,
        "allocation_decision": record.allocation_decision,
    }
    if field == "total_costs":
        values["trade"] = record.trade.model_copy(
            update={
                "total_costs": record.trade.total_costs + Decimal("1"),
                "net_pnl": record.trade.net_pnl - Decimal("1"),
            }
        )
    elif field == "cost_policy_id":
        values["cost_policy_id"] = "wrong-policy"
    elif field == "ml_score":
        values["trade"] = record.trade.model_copy(
            update={
                "ml_score": record.trade.ml_score.model_copy(
                    update={"model_version": "tampered"}
                )
            }
        )
    else:
        values["trade"] = record.trade.model_copy(update={"target_notional": 55_000})

    with pytest.raises(ValidationError, match=message):
        BacktestTradeRecord(**values)


def test_request_result_rejects_mismatched_request_and_trade_identity(
    completed_audit_objects,
) -> None:
    record, terminal = completed_audit_objects
    other_request = BacktestTradeRequest(
        candidate=make_candidate(
            terminal.request.candidate.order_intent.timestamp,
            strategy_id="other",
        )
    )

    with pytest.raises(ValidationError, match="request candidate"):
        BacktestRequestResult(
            request=other_request,
            outcome=terminal.outcome,
            terminal_at=terminal.terminal_at,
            allocation_decision=terminal.allocation_decision,
            trade_record=record,
        )


def test_request_result_rejects_tampered_trade_record_identity(
    completed_audit_objects,
) -> None:
    record, terminal = completed_audit_objects
    other_candidate = make_candidate(
        terminal.request.candidate.order_intent.timestamp,
        strategy_id="other",
    )
    original_decision = record.allocation_decision
    original_reservation = original_decision.reservation
    assert original_reservation is not None
    other_reservation = type(original_reservation)(
        candidate=other_candidate,
        margin_quote=original_reservation.margin_quote,
    )
    other_decision = type(original_decision)(
        candidate=other_candidate,
        outcome=original_decision.outcome,
        margin_quote=original_decision.margin_quote,
        signal=other_candidate.order_intent.signal,
        reservation=other_reservation,
    )
    other_record = BacktestTradeRecord(
        trade=record.trade,
        round_trip_cost_breakdown=record.round_trip_cost_breakdown,
        cost_policy_id=record.cost_policy_id,
        allocation_identity=other_candidate.identity,
        allocation_decision=other_decision,
    )

    with pytest.raises(ValidationError, match="trade record identity"):
        BacktestRequestResult(
            request=terminal.request,
            outcome=terminal.outcome,
            terminal_at=terminal.terminal_at,
            allocation_decision=terminal.allocation_decision,
            trade_record=other_record,
        )


def test_missing_cutoff_candle_does_not_use_later_time_exit(tmp_path: Path) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 15, 30), 101.0, 102.0, 100.0, 101.0),
            ]
        },
    )

    with pytest.raises(BacktestIntegrityError, match="exactly at cutoff"):
        run_backtest(
            store,
            [make_request(at(*day, 9, 20))],
            StubMarginProvider({"strategy": Decimal("1")}),
            make_config(at(*day, 9, 15), at(*day, 15, 35)),
        )


def test_entered_shadow_also_requires_exact_cutoff_candle(tmp_path: Path) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 10, 0), 101.0, 102.0, 100.0, 101.0),
                (at(*day, 15, 30), 102.0, 103.0, 101.0, 102.0),
            ]
        },
    )
    allocated = BacktestTradeRequest(
        candidate=make_candidate(
            at(*day, 9, 20),
            strategy_id="allocated",
            quality_score=1.0,
        ),
        strategy_exit_at=at(*day, 10, 0),
    )
    shadow = make_request(
        at(*day, 9, 20),
        strategy_id="shadow",
        quality_score=0.0,
    )

    with pytest.raises(BacktestIntegrityError, match="exactly at cutoff"):
        run_backtest(
            store,
            [shadow, allocated],
            StubMarginProvider(
                {"allocated": Decimal("100000"), "shadow": Decimal("100000")}
            ),
            make_config(at(*day, 9, 15), at(*day, 15, 35)),
        )


def test_strategy_exit_after_cutoff_is_not_admissible(tmp_path: Path) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 15, 30), 101.0, 102.0, 100.0, 101.0),
            ]
        },
    )
    request = BacktestTradeRequest(
        candidate=make_candidate(at(*day, 9, 20)),
        strategy_exit_at=at(*day, 15, 20),
    )

    with pytest.raises(BacktestIntegrityError, match="exactly at cutoff"):
        run_backtest(
            store,
            [request],
            StubMarginProvider({"strategy": Decimal("1")}),
            make_config(at(*day, 9, 15), at(*day, 15, 35)),
        )


def test_protective_exit_after_cutoff_is_ignored_for_valid_time_exit(
    tmp_path: Path,
) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 15, 25), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 15, 30), 106.0, 107.0, 105.0, 106.0),
            ]
        },
    )
    request = BacktestTradeRequest(
        candidate=make_candidate(at(*day, 9, 20)),
        protective_exit=ProtectiveExitSpec(target_price=Decimal("105")),
    )

    result = run_backtest(
        store,
        [request],
        StubMarginProvider({"strategy": Decimal("1")}),
        make_config(at(*day, 9, 15), at(*day, 15, 35)),
    )

    trade = result.actual_trade_records[0].trade
    assert trade.exit_reason is ExitReason.TIME_EXIT
    assert trade.exit_fill.timestamp == at(*day, 15, 25)


def test_protective_exit_exactly_at_cutoff_beats_time_exit(tmp_path: Path) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 15, 20), 100.0, 106.0, 99.0, 105.0),
                (at(*day, 15, 25), 104.0, 105.0, 103.0, 104.0),
            ]
        },
    )
    request = BacktestTradeRequest(
        candidate=make_candidate(at(*day, 9, 20)),
        protective_exit=ProtectiveExitSpec(target_price=Decimal("105")),
    )

    result = run_backtest(
        store,
        [request],
        StubMarginProvider({"strategy": Decimal("1")}),
        make_config(at(*day, 9, 15), at(*day, 15, 30)),
    )

    trade = result.actual_trade_records[0].trade
    assert trade.exit_reason is ExitReason.TARGET_REACHED
    assert trade.exit_fill.timestamp == at(*day, 15, 25)


def test_strategy_exit_exactly_at_cutoff_beats_time_exit(tmp_path: Path) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 15, 25), 101.0, 102.0, 100.0, 101.0),
            ]
        },
    )
    request = BacktestTradeRequest(
        candidate=make_candidate(at(*day, 9, 20)),
        strategy_exit_at=at(*day, 15, 25),
    )

    result = run_backtest(
        store,
        [request],
        StubMarginProvider({"strategy": Decimal("1")}),
        make_config(at(*day, 9, 15), at(*day, 15, 30)),
    )

    trade = result.actual_trade_records[0].trade
    assert trade.exit_reason is ExitReason.STRATEGY_EXIT
    assert trade.exit_fill.timestamp == at(*day, 15, 25)


def test_earlier_protective_exit_does_not_require_cutoff_candle(tmp_path: Path) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 10, 0), 100.0, 106.0, 99.0, 105.0),
                (at(*day, 15, 30), 110.0, 111.0, 109.0, 110.0),
            ]
        },
    )
    request = BacktestTradeRequest(
        candidate=make_candidate(at(*day, 9, 20)),
        protective_exit=ProtectiveExitSpec(target_price=Decimal("105")),
    )

    result = run_backtest(
        store,
        [request],
        StubMarginProvider({"strategy": Decimal("1")}),
        make_config(at(*day, 9, 15), at(*day, 15, 35)),
    )

    trade = result.actual_trade_records[0].trade
    assert trade.exit_reason is ExitReason.TARGET_REACHED
    assert trade.exit_fill.timestamp == at(*day, 10, 5)


def test_earlier_strategy_exit_does_not_require_cutoff_candle(tmp_path: Path) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 10, 0), 101.0, 102.0, 100.0, 101.0),
                (at(*day, 15, 30), 102.0, 103.0, 101.0, 102.0),
            ]
        },
    )
    request = BacktestTradeRequest(
        candidate=make_candidate(at(*day, 9, 20)),
        strategy_exit_at=at(*day, 10, 0),
    )

    result = run_backtest(
        store,
        [request],
        StubMarginProvider({"strategy": Decimal("1")}),
        make_config(at(*day, 9, 15), at(*day, 15, 35)),
    )

    trade = result.actual_trade_records[0].trade
    assert trade.exit_reason is ExitReason.STRATEGY_EXIT
    assert trade.exit_fill.timestamp == at(*day, 10, 0)


def test_protective_exit_at_cutoff_is_sufficient_without_cutoff_candle(
    tmp_path: Path,
) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 15, 20), 100.0, 106.0, 99.0, 105.0),
                (at(*day, 15, 30), 110.0, 111.0, 109.0, 110.0),
            ]
        },
    )
    request = BacktestTradeRequest(
        candidate=make_candidate(at(*day, 9, 20)),
        protective_exit=ProtectiveExitSpec(target_price=Decimal("105")),
    )

    result = run_backtest(
        store,
        [request],
        StubMarginProvider({"strategy": Decimal("1")}),
        make_config(at(*day, 9, 15), at(*day, 15, 35)),
    )

    trade = result.actual_trade_records[0].trade
    assert trade.exit_reason is ExitReason.TARGET_REACHED
    assert trade.exit_fill.timestamp == at(*day, 15, 25)


def test_optional_exit_after_cutoff_cannot_rescue_missing_time_exit(
    tmp_path: Path,
) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 15, 30), 106.0, 107.0, 105.0, 106.0),
            ]
        },
    )
    request = BacktestTradeRequest(
        candidate=make_candidate(at(*day, 9, 20)),
        protective_exit=ProtectiveExitSpec(target_price=Decimal("105")),
    )

    with pytest.raises(BacktestIntegrityError, match="exactly at cutoff"):
        run_backtest(
            store,
            [request],
            StubMarginProvider({"strategy": Decimal("1")}),
            make_config(at(*day, 9, 15), at(*day, 15, 35)),
        )


def test_shadow_earlier_exit_also_does_not_require_cutoff_candle(
    tmp_path: Path,
) -> None:
    day = (2025, 1, 2)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 10, 0), 101.0, 102.0, 100.0, 101.0),
                (at(*day, 15, 30), 102.0, 103.0, 101.0, 102.0),
            ]
        },
    )
    allocated = BacktestTradeRequest(
        candidate=make_candidate(
            at(*day, 9, 20),
            strategy_id="allocated",
            quality_score=1.0,
        ),
        strategy_exit_at=at(*day, 10, 0),
    )
    shadow = BacktestTradeRequest(
        candidate=make_candidate(
            at(*day, 9, 20),
            strategy_id="shadow",
            quality_score=0.0,
        ),
        strategy_exit_at=at(*day, 10, 0),
    )

    result = run_backtest(
        store,
        [shadow, allocated],
        StubMarginProvider(
            {"allocated": Decimal("100000"), "shadow": Decimal("100000")}
        ),
        make_config(at(*day, 9, 15), at(*day, 15, 35)),
    )

    assert result.actual_trade_records[0].trade.exit_reason is ExitReason.STRATEGY_EXIT
    assert result.shadow_trade_records[0].trade.exit_reason is ExitReason.STRATEGY_EXIT
    assert result.shadow_trade_records[0].trade.exit_fill.timestamp == at(*day, 10, 0)


@pytest.mark.parametrize(
    ("forced_exit_time", "order_timestamp"),
    [
        (time(15, 25), at(2025, 1, 2, 15, 30)),
        (time(14, 0), at(2025, 1, 2, 14, 5)),
    ],
    ids=["default-cutoff", "custom-cutoff"],
)
def test_order_after_forced_cutoff_is_rejected_before_provider_or_data_access(
    tmp_path: Path,
    forced_exit_time: time,
    order_timestamp: datetime,
) -> None:
    day = (2025, 1, 2)
    write_symbol(
        tmp_path,
        "TEST",
        [(at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0)],
    )
    store = CountingMarketDataStore(MarketDataConfig(dataset_path=tmp_path))
    provider = StubMarginProvider({"strategy": Decimal("1")})

    with pytest.raises(ValueError, match="later than.*forced-exit cutoff"):
        run_backtest(
            store,
            [make_request(order_timestamp)],
            provider,
            make_config(
                at(*day, 9, 15),
                at(*day, 16, 0),
                forced_exit_time=forced_exit_time,
            ),
        )

    assert provider.calls == []
    assert store.loads == []


def test_order_exactly_at_cutoff_remains_auditable_no_fill(tmp_path: Path) -> None:
    day = (2025, 1, 2)
    cutoff = at(*day, 15, 25)
    store = make_store(
        tmp_path,
        {"TEST": [(cutoff, 100.0, 101.0, 99.0, 100.0)]},
    )

    result = run_backtest(
        store,
        [make_request(cutoff)],
        StubMarginProvider({"strategy": Decimal("100000")}),
        make_config(at(*day, 9, 15), at(*day, 15, 30)),
    )

    terminal = result.request_results[0]
    assert terminal.outcome is BacktestRequestOutcome.ALLOCATED_ENTRY_NOT_FILLED
    assert terminal.terminal_at == cutoff
    assert terminal.trade_record is None
    assert result.ending_portfolio_state is not None
    assert result.ending_portfolio_state.reserved_margin == 0


def test_request_result_terminal_cannot_precede_order_timestamp() -> None:
    order_timestamp = at(2025, 1, 2, 10, 0)
    request = make_request(order_timestamp)

    with pytest.raises(ValidationError, match="terminal_at cannot precede"):
        BacktestRequestResult(
            request=request,
            outcome=BacktestRequestOutcome.CAPITAL_EXHAUSTED,
            terminal_at=at(2025, 1, 2, 9, 55),
        )

    valid = BacktestRequestResult(
        request=request,
        outcome=BacktestRequestOutcome.CAPITAL_EXHAUSTED,
        terminal_at=order_timestamp,
    )
    assert valid.terminal_at == order_timestamp


def test_entry_simulator_receives_only_bar_starts_before_entry_deadline(
    tmp_path: Path,
) -> None:
    day = (2025, 1, 2)
    deadline = at(*day, 9, 45)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 9, 40), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 10, 0), 90.0, 91.0, 89.0, 90.0),
                (at(*day, 15, 25), 100.0, 101.0, 99.0, 100.0),
            ]
        },
    )
    simulator = TrackingExecutionSimulator()
    request = BacktestTradeRequest(
        candidate=make_candidate(
            at(*day, 9, 20),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("90"),
        ),
        strategy_exit_at=deadline,
    )

    result = run_backtest(
        store,
        [request],
        StubMarginProvider({"strategy": Decimal("1")}),
        make_config(at(*day, 9, 15), at(*day, 15, 30)),
        simulator=simulator,
    )

    assert result.request_results[0].outcome is (
        BacktestRequestOutcome.ALLOCATED_ENTRY_NOT_FILLED
    )
    assert simulator.entry_candle_starts == [
        at(*day, 9, 20),
        at(*day, 9, 40),
    ]
    assert all(timestamp < deadline for timestamp in simulator.entry_candle_starts)


def test_intrabar_entry_timestamped_at_deadline_remains_no_fill(tmp_path: Path) -> None:
    day = (2025, 1, 2)
    deadline = at(*day, 9, 45)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (at(*day, 9, 40), 100.0, 101.0, 89.0, 95.0),
                (at(*day, 15, 25), 100.0, 101.0, 99.0, 100.0),
            ]
        },
    )
    request = BacktestTradeRequest(
        candidate=make_candidate(
            at(*day, 9, 20),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("90"),
        ),
        strategy_exit_at=deadline,
    )

    result = run_backtest(
        store,
        [request],
        StubMarginProvider({"strategy": Decimal("1")}),
        make_config(at(*day, 9, 15), at(*day, 15, 30)),
    )

    terminal = result.request_results[0]
    assert terminal.outcome is BacktestRequestOutcome.ALLOCATED_ENTRY_NOT_FILLED
    assert terminal.terminal_at == deadline


def test_strategy_exit_after_cutoff_is_skipped_and_time_exit_is_fallback(
    tmp_path: Path,
) -> None:
    day = (2025, 1, 2)
    cutoff = at(*day, 15, 25)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (cutoff, 101.0, 102.0, 100.0, 101.0),
                (at(*day, 15, 30), 102.0, 103.0, 101.0, 102.0),
            ]
        },
    )
    simulator = TrackingExecutionSimulator()
    request = BacktestTradeRequest(
        candidate=make_candidate(at(*day, 9, 20)),
        strategy_exit_at=at(*day, 15, 30),
    )

    result = run_backtest(
        store,
        [request],
        StubMarginProvider({"strategy": Decimal("1")}),
        make_config(at(*day, 9, 15), at(*day, 15, 35)),
        simulator=simulator,
    )

    trade = result.actual_trade_records[0].trade
    assert trade.exit_reason is ExitReason.TIME_EXIT
    assert simulator.market_exit_calls == [(cutoff, ExitReason.TIME_EXIT)]
    assert all(
        bar_start <= cutoff
        for frame_starts in simulator.market_exit_candle_starts
        for bar_start in frame_starts
    )


def test_earlier_strategy_exit_returns_without_requesting_time_exit(
    tmp_path: Path,
) -> None:
    day = (2025, 1, 2)
    strategy_exit_at = at(*day, 10, 0)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 101.0, 99.0, 100.0),
                (strategy_exit_at, 101.0, 102.0, 100.0, 101.0),
                (at(*day, 15, 25), 102.0, 103.0, 101.0, 102.0),
            ]
        },
    )
    simulator = TrackingExecutionSimulator()
    request = BacktestTradeRequest(
        candidate=make_candidate(at(*day, 9, 20)),
        strategy_exit_at=strategy_exit_at,
    )

    result = run_backtest(
        store,
        [request],
        StubMarginProvider({"strategy": Decimal("1")}),
        make_config(at(*day, 9, 15), at(*day, 15, 30)),
        simulator=simulator,
    )

    assert result.actual_trade_records[0].trade.exit_reason is ExitReason.STRATEGY_EXIT
    assert simulator.market_exit_calls == [
        (strategy_exit_at, ExitReason.STRATEGY_EXIT)
    ]


def test_earlier_protective_exit_returns_without_requesting_time_exit(
    tmp_path: Path,
) -> None:
    day = (2025, 1, 2)
    cutoff = at(*day, 15, 25)
    store = make_store(
        tmp_path,
        {
            "TEST": [
                (at(*day, 9, 20), 100.0, 106.0, 99.0, 105.0),
                (cutoff, 105.0, 106.0, 104.0, 105.0),
                (at(*day, 15, 30), 110.0, 111.0, 109.0, 110.0),
            ]
        },
    )
    simulator = TrackingExecutionSimulator()
    request = BacktestTradeRequest(
        candidate=make_candidate(at(*day, 9, 20)),
        protective_exit=ProtectiveExitSpec(target_price=Decimal("105")),
    )

    result = run_backtest(
        store,
        [request],
        StubMarginProvider({"strategy": Decimal("1")}),
        make_config(at(*day, 9, 15), at(*day, 15, 35)),
        simulator=simulator,
    )

    assert result.actual_trade_records[0].trade.exit_reason is ExitReason.TARGET_REACHED
    assert simulator.market_exit_calls == []
    assert all(timestamp <= cutoff for timestamp in simulator.protective_candle_starts)
