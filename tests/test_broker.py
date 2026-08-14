import copy
import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from threading import Event, Thread
from zoneinfo import ZoneInfo

import polars as pl
import pyotp
import pytest
from pydantic import SecretStr, ValidationError

from algo_trader.broker import (
    ANGEL_DURATION,
    ANGEL_PRODUCT_TYPE,
    ANGEL_VARIETY,
    BROKER_ARCHITECTURE_VERSION,
    FIVE_MINUTE_INTERVAL,
    HISTORICAL_MARGIN_CALCULATION_METHOD,
    LIVE_MARGIN_PROVIDER_ID,
    NSE_CASH_EXCHANGE_TYPE,
    AngelOneBroker,
    AngelOneCandleClient,
    AngelOneCredentials,
    AngelOneLiveMarginProvider,
    AngelOneMarketDataStream,
    AngelOneSession,
    BrokerAmbiguousStateError,
    BrokerApiError,
    BrokerAuthenticationError,
    BrokerCancellationAcknowledgement,
    BrokerDataError,
    BrokerInstrumentError,
    BrokerOrderAcknowledgement,
    BrokerOrderRequest,
    BrokerOrderState,
    BrokerSystemicError,
    BrokerTradeFill,
    BrokerTransactionAction,
    HistoricalMarginRequirementProvider,
    HistoricalMarginSnapshot,
    HistoricalMarginSnapshotEntry,
    angel_order_payload,
    broker_order_tag,
    create_historical_margin_snapshot,
    create_margin_snapshot_entry,
    entry_action,
    exit_action,
    load_historical_margin_snapshot,
    parse_instrument_master,
    save_historical_margin_snapshot,
)
from algo_trader.data import CANONICAL_CANDLE_COLUMNS
from algo_trader.domain import MLScore, OrderIntent, OrderType, Side, Signal
from algo_trader.portfolio import (
    AllocationCandidate,
    MarginRequirementProvider,
    PortfolioState,
)

MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")
TOTP_SECRET = "JBSWY3DPEHPK3PXP"


def at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 8, 14, hour, minute, tzinfo=MARKET_TIMEZONE)


def master_rows() -> list[dict[str, str]]:
    return [
        {
            "token": "2885",
            "symbol": "RELIANCE-EQ",
            "name": "RELIANCE",
            "expiry": "",
            "strike": "-1.000000",
            "lotsize": "1",
            "instrumenttype": "",
            "exch_seg": "NSE",
            "tick_size": "5.000000",
        },
        {
            "token": "9991",
            "symbol": "RELIANCE-BE",
            "name": "RELIANCE",
            "expiry": "",
            "strike": "-1.000000",
            "lotsize": "1",
            "instrumenttype": "",
            "exch_seg": "NSE",
            "tick_size": "5.000000",
        },
        {
            "token": "9992",
            "symbol": "RELIANCE-BL",
            "name": "RELIANCE",
            "expiry": "",
            "strike": "-1.000000",
            "lotsize": "1",
            "instrumenttype": "",
            "exch_seg": "NSE",
            "tick_size": "5.000000",
        },
        {
            "token": "9993",
            "symbol": "RELIANCE-EQ",
            "name": "RELIANCE",
            "expiry": "",
            "strike": "-1.000000",
            "lotsize": "1",
            "instrumenttype": "EQ",
            "exch_seg": "BSE",
            "tick_size": "5.000000",
        },
    ]


def instrument_master():
    return parse_instrument_master(json.dumps(master_rows()))


class FakeSDK:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.session_response: object = {
            "status": True,
            "message": "SUCCESS",
            "data": {
                "clientcode": "CLIENT1",
                "jwtToken": "jwt-secret",
                "refreshToken": "refresh-secret",
                "feedToken": "feed-secret",
            },
        }
        self.refresh_response: object = {
            "status": True,
            "data": {"jwtToken": "jwt-new", "feedToken": "feed-new"},
        }
        self.logout_response: object = {"status": True, "message": "SUCCESS", "data": {}}
        self.place_response: object = {
            "status": True,
            "message": "SUCCESS",
            "errorcode": "",
            "data": {"orderid": "ORDER1", "uniqueorderid": "UNIQUE1"},
        }
        self.cancel_response: object = {"status": True, "message": "SUCCESS", "data": {}}
        self.order_response: object = {"status": True, "data": []}
        self.individual_response: object = {"status": True, "data": {}}
        self.trade_response: object = {"status": True, "data": []}
        self.position_response: object = {"status": True, "data": []}
        self.funds_response: object = {
            "status": True,
            "data": {"net": "100000.25", "availablecash": "80000.50"},
        }
        self.ltp_response: object = {"status": True, "data": {"ltp": "2500.25"}}
        self.candle_response: object = {"status": True, "data": []}
        self.margin_response: object = {
            "status": True,
            "data": {"totalMarginRequired": "12345.67"},
        }

    def generateSession(self, client_code, pin, totp):
        self.calls.append(("generateSession", (client_code, pin, totp)))
        return self.session_response

    def generateToken(self, refresh_token):
        self.calls.append(("generateToken", refresh_token))
        return self.refresh_response

    def terminateSession(self, client_code):
        self.calls.append(("terminateSession", client_code))
        return self.logout_response

    def placeOrderFullResponse(self, payload):
        self.calls.append(("placeOrderFullResponse", copy.deepcopy(payload)))
        return self.place_response

    def cancelOrder(self, order_id, variety):
        self.calls.append(("cancelOrder", (order_id, variety)))
        return self.cancel_response

    def orderBook(self):
        self.calls.append(("orderBook", None))
        return self.order_response

    def individual_order_details(self, query):
        self.calls.append(("individual_order_details", query))
        return self.individual_response

    def tradeBook(self):
        self.calls.append(("tradeBook", None))
        return self.trade_response

    def position(self):
        self.calls.append(("position", None))
        return self.position_response

    def rmsLimit(self):
        self.calls.append(("rmsLimit", None))
        return self.funds_response

    def ltpData(self, exchange, trading_symbol, token):
        self.calls.append(("ltpData", (exchange, trading_symbol, token)))
        return self.ltp_response

    def getCandleData(self, payload):
        self.calls.append(("getCandleData", copy.deepcopy(payload)))
        return self.candle_response

    def getMarginApi(self, payload):
        self.calls.append(("getMarginApi", copy.deepcopy(payload)))
        return self.margin_response


def credentials() -> AngelOneCredentials:
    return AngelOneCredentials(
        api_key="api-secret",
        client_code="CLIENT1",
        pin="1234",
        totp_secret=TOTP_SECRET,
    )


def session() -> AngelOneSession:
    return AngelOneSession(
        client_code="CLIENT1",
        jwt_token="jwt-secret",
        refresh_token="refresh-secret",
        feed_token="feed-secret",
        authenticated_at=at(9),
        sdk_version="1.5.5",
    )


def order_request(
    *,
    order_type: OrderType = OrderType.MARKET,
    action: BrokerTransactionAction = BrokerTransactionAction.BUY,
    consent: bool = False,
) -> BrokerOrderRequest:
    return BrokerOrderRequest(
        client_order_id="research-order-123",
        instrument=instrument_master().resolve("RELIANCE"),
        transaction_action=action,
        order_type=order_type,
        quantity=20,
        limit_price=Decimal("2500.05") if order_type is OrderType.LIMIT else None,
        submitted_at=at(10),
        scrip_consent=consent,
    )


def candidate(
    *,
    side: Side = Side.LONG,
    notional: int = 50_000,
    order_type: OrderType = OrderType.MARKET,
) -> AllocationCandidate:
    signal = Signal(
        strategy_id="strategy",
        strategy_version="1",
        symbol="RELIANCE",
        timestamp=at(9, 15),
        side=side,
    )
    intent = OrderIntent(
        signal=signal,
        timestamp=at(9, 20),
        quantity=20,
        requested_notional=notional,
        order_type=order_type,
        limit_price=Decimal("2500") if order_type is OrderType.LIMIT else None,
    )
    score = MLScore(
        model_version="bootstrap-v1",
        quality_score=0.5,
        calibrated_probability=0.5,
        predicted_net_return=0,
        recommended_notional=notional,
    )
    return AllocationCandidate(order_intent=intent, ml_score=score)


def order_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "orderid": "ORDER1",
        "uniqueorderid": "UNIQUE1",
        "ordertag": broker_order_tag("research-order-123"),
        "symboltoken": "2885",
        "transactiontype": "BUY",
        "ordertype": "MARKET",
        "quantity": "20",
        "filledshares": "0",
        "unfilledshares": "20",
        "averageprice": "0",
        "status": "open",
        "orderstatus": "open",
        "text": "",
        "updatetime": "14-Aug-2026 10:00:00",
        "exchtime": "14-Aug-2026 10:00:01",
    }
    row.update(updates)
    return row


def test_credentials_and_sessions_are_immutable_and_secret_safe() -> None:
    selected = credentials()
    active = session()
    assert isinstance(selected.api_key, SecretStr)
    assert isinstance(selected.pin, SecretStr)
    assert isinstance(selected.totp_secret, SecretStr)
    assert isinstance(active.jwt_token, SecretStr)
    assert isinstance(active.refresh_token, SecretStr)
    assert isinstance(active.feed_token, SecretStr)
    combined = repr(selected) + repr(active) + selected.model_dump_json() + active.model_dump_json()
    for secret in ("api-secret", "1234", TOTP_SECRET, "jwt-secret", "refresh-secret"):
        assert secret not in combined
    with pytest.raises(ValidationError):
        selected.client_code = "OTHER"


def test_authentication_uses_caller_time_totp_refreshes_immutably_and_logs_out() -> None:
    sdk = FakeSDK()
    factory_calls = []

    def factory(**kwargs):
        factory_calls.append(kwargs)
        return sdk

    broker = AngelOneBroker(instrument_master(), sdk_factory=factory)
    occurred_at = at(9, 30)
    authenticated = broker.authenticate(credentials(), occurred_at)
    expected_totp = pyotp.TOTP(TOTP_SECRET).at(int(occurred_at.timestamp()))
    assert factory_calls == [{"api_key": "api-secret"}]
    assert sdk.calls[0] == ("generateSession", ("CLIENT1", "1234", expected_totp))
    assert authenticated.authenticated_at == occurred_at
    assert authenticated.jwt_token.get_secret_value() == "jwt-secret"
    assert broker.authenticated_client() is sdk

    refreshed = broker.refresh_session(authenticated, at(10))
    assert refreshed is not authenticated
    assert authenticated.jwt_token.get_secret_value() == "jwt-secret"
    assert refreshed.jwt_token.get_secret_value() == "jwt-new"
    assert refreshed.refresh_token.get_secret_value() == "refresh-secret"
    broker.logout(refreshed)
    assert sdk.calls[-1] == ("terminateSession", "CLIENT1")


@pytest.mark.parametrize(
    "response",
    [
        {"status": False, "message": "bad login", "data": None},
        {"status": True, "data": {"jwtToken": "x"}},
    ],
)
def test_authentication_rejects_failure_or_missing_tokens(response: object) -> None:
    sdk = FakeSDK()
    sdk.session_response = response
    broker = AngelOneBroker(instrument_master(), sdk_factory=lambda **kwargs: sdk)
    with pytest.raises(BrokerAuthenticationError):
        broker.authenticate(credentials(), at(9))
    with pytest.raises(ValueError, match="timezone-aware"):
        broker.authenticate(credentials(), datetime(2026, 8, 14, 9))


def test_instrument_parser_selects_only_exact_nse_eq_deterministically() -> None:
    first = parse_instrument_master(json.dumps(master_rows()))
    second = parse_instrument_master(json.dumps(list(reversed(master_rows()))))
    instrument = first.resolve("RELIANCE")
    assert instrument == second.resolve("RELIANCE")
    assert instrument.trading_symbol == "RELIANCE-EQ"
    assert instrument.symbol_token == "2885"
    assert instrument.lot_size == 1
    assert instrument.tick_size == Decimal("0.05")
    assert len(first.instruments) == 1
    with pytest.raises(BrokerInstrumentError, match="no exact"):
        first.resolve("UNKNOWN")
    aliased = first.with_aliases({"RELIANCE-OLD": "RELIANCE"})
    assert aliased.resolve("RELIANCE-OLD") == instrument
    assert aliased.aliases == {"RELIANCE-OLD": "RELIANCE"}
    with pytest.raises(BrokerInstrumentError, match="targets"):
        first.with_aliases({"OLD": "MISSING"})


def test_instrument_parser_rejects_ambiguous_and_malformed_exact_records() -> None:
    duplicate = [master_rows()[0], master_rows()[0] | {"token": "2886"}]
    with pytest.raises(BrokerInstrumentError, match="ambiguous"):
        parse_instrument_master(duplicate).resolve("RELIANCE")
    malformed = [master_rows()[0] | {"lotsize": "not-a-number"}]
    with pytest.raises(BrokerDataError, match="lotsize"):
        parse_instrument_master(malformed)


@pytest.mark.parametrize(
    ("side", "entry", "exit_value"),
    [
        (Side.LONG, BrokerTransactionAction.BUY, BrokerTransactionAction.SELL),
        (Side.SHORT, BrokerTransactionAction.SELL, BrokerTransactionAction.BUY),
    ],
)
def test_entry_and_exit_actions_are_explicit(
    side: Side,
    entry: BrokerTransactionAction,
    exit_value: BrokerTransactionAction,
) -> None:
    assert entry_action(side) is entry
    assert exit_action(side) is exit_value


def test_order_request_and_payload_enforce_fixed_contract_and_consent() -> None:
    request = order_request()
    payload = angel_order_payload(request)
    assert payload == {
        "variety": ANGEL_VARIETY,
        "tradingsymbol": "RELIANCE-EQ",
        "symboltoken": "2885",
        "transactiontype": "BUY",
        "exchange": "NSE",
        "ordertype": "MARKET",
        "producttype": ANGEL_PRODUCT_TYPE,
        "duration": ANGEL_DURATION,
        "quantity": "20",
        "price": "0",
        "ordertag": broker_order_tag(request.client_order_id),
    }
    assert len(payload["ordertag"]) == 19
    assert broker_order_tag(request.client_order_id) == broker_order_tag(request.client_order_id)
    assert angel_order_payload(order_request(consent=True))["scripconsent"] == "yes"
    assert angel_order_payload(order_request(order_type=OrderType.LIMIT))["price"] == "2500.05"

    values = request.model_dump()
    with pytest.raises(ValidationError, match="cannot have"):
        BrokerOrderRequest(**(values | {"limit_price": Decimal("1")}))
    with pytest.raises(ValidationError, match="require"):
        BrokerOrderRequest(**(values | {"order_type": OrderType.LIMIT}))
    with pytest.raises(ValidationError):
        BrokerOrderRequest(**(values | {"quantity": 0}))


def test_place_order_returns_acknowledgement_only_once_without_mutating_request() -> None:
    sdk = FakeSDK()
    broker = AngelOneBroker(instrument_master(), sdk_client=sdk)
    request = order_request()
    before = request.model_dump()
    acknowledgement = broker.place_order(request, at(10, 1))
    assert isinstance(acknowledgement, BrokerOrderAcknowledgement)
    assert acknowledgement.broker_order_id == "ORDER1"
    assert acknowledgement.unique_order_id == "UNIQUE1"
    assert not isinstance(acknowledgement, BrokerTradeFill)
    assert [name for name, _ in sdk.calls].count("placeOrderFullResponse") == 1
    assert request.model_dump() == before

    sdk.place_response = {"status": False, "message": "static IP rejected", "data": None}
    with pytest.raises(BrokerApiError, match="static IP rejected"):
        broker.place_order(request, at(10, 2))
    assert [name for name, _ in sdk.calls].count("placeOrderFullResponse") == 2


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"status": "open", "orderstatus": "open"}, BrokerOrderState.OPEN),
        ({"status": "open pending", "orderstatus": "open pending"}, BrokerOrderState.PENDING),
        (
            {
                "status": "complete",
                "orderstatus": "complete",
                "filledshares": "20",
                "unfilledshares": "0",
            },
            BrokerOrderState.FILLED,
        ),
        (
            {"filledshares": "5", "unfilledshares": "15"},
            BrokerOrderState.PARTIALLY_FILLED,
        ),
        ({"status": "cancelled", "orderstatus": "cancelled"}, BrokerOrderState.CANCELLED),
        ({"status": "rejected", "orderstatus": "rejected"}, BrokerOrderState.REJECTED),
        ({"status": "future status", "orderstatus": "future status"}, BrokerOrderState.UNKNOWN),
    ],
)
def test_order_states_preserve_partial_and_unknown_evidence(
    updates: dict[str, object],
    expected: BrokerOrderState,
) -> None:
    sdk = FakeSDK()
    sdk.order_response = {"status": True, "data": [order_row(**updates)]}
    snapshot = AngelOneBroker(instrument_master(), sdk_client=sdk).list_orders()[0]
    assert snapshot.state is expected
    assert snapshot.raw_status == updates.get("status", "open")
    if expected is BrokerOrderState.PARTIALLY_FILLED:
        assert (snapshot.filled_quantity, snapshot.remaining_quantity) == (5, 15)


def test_exact_order_reconciliation_ambiguity_and_malformed_data() -> None:
    sdk = FakeSDK()
    sdk.order_response = {
        "status": True,
        "data": [order_row(), order_row(orderid="ORDER2", uniqueorderid="UNIQUE2")],
    }
    broker = AngelOneBroker(instrument_master(), sdk_client=sdk)
    assert broker.get_order(broker_order_id="ORDER2").broker_order_id == "ORDER2"
    with pytest.raises(BrokerAmbiguousStateError):
        broker.get_order(broker_order_tag=broker_order_tag("research-order-123"))

    sdk.individual_response = {"status": True, "data": order_row()}
    assert broker.get_order(unique_order_id="UNIQUE1").unique_order_id == "UNIQUE1"
    assert sdk.calls[-1] == ("individual_order_details", "?uniqueorderid=UNIQUE1")
    sdk.individual_response = {
        "status": True,
        "data": [order_row(), order_row(orderid="ORDER2", uniqueorderid="UNIQUE2")],
    }
    assert broker.get_order(unique_order_id="UNIQUE2").broker_order_id == "ORDER2"
    with pytest.raises(LookupError, match="no broker order"):
        broker.get_order(unique_order_id="MISSING")

    sdk.order_response = {"status": True, "data": [order_row(quantity="bad")]}
    with pytest.raises(BrokerDataError, match="numeric"):
        broker.list_orders()
    sdk.order_response = {"status": True, "data": [order_row(averageprice="NaN")]}
    with pytest.raises(BrokerDataError, match="finite"):
        broker.list_orders()


def test_cancel_acknowledgement_is_not_final_no_fill_proof() -> None:
    sdk = FakeSDK()
    acknowledgement = AngelOneBroker(instrument_master(), sdk_client=sdk).cancel_order(
        "ORDER1", at(10, 5)
    )
    assert isinstance(acknowledgement, BrokerCancellationAcknowledgement)
    assert not hasattr(acknowledgement, "state")
    assert sdk.calls == [("cancelOrder", ("ORDER1", "NORMAL"))]
    with pytest.raises(ValueError, match="non-empty"):
        AngelOneBroker(instrument_master(), sdk_client=sdk).cancel_order("", at(10, 5))
    assert sdk.calls == [("cancelOrder", ("ORDER1", "NORMAL"))]


def test_trade_fills_are_decimal_separate_and_deterministically_sorted() -> None:
    sdk = FakeSDK()
    later = {
        "orderid": "ORDER1",
        "fillid": "FILL2",
        "symboltoken": "2885",
        "transactiontype": "BUY",
        "filltime": "14-Aug-2026 10:00:02",
        "fillprice": "2500.15",
        "fillsize": "12",
    }
    earlier = later | {"fillid": "FILL1", "filltime": "14-Aug-2026 10:00:01", "fillsize": "8"}
    sdk.trade_response = {"status": True, "data": [later, earlier]}
    fills = AngelOneBroker(instrument_master(), sdk_client=sdk).list_trade_fills()
    assert [fill.fill_id for fill in fills] == ["FILL1", "FILL2"]
    assert [fill.quantity for fill in fills] == [8, 12]
    assert fills[0].fill_price == Decimal("2500.15")
    assert fills[0].fill_timestamp.tzinfo == MARKET_TIMEZONE


def test_positions_and_funds_preserve_external_broker_truth() -> None:
    sdk = FakeSDK()
    position = {
        "symboltoken": "2885",
        "producttype": "INTRADAY",
        "netqty": "20",
        "buyqty": "20",
        "sellqty": "0",
        "buyavgprice": "2500.25",
        "sellavgprice": "0",
        "netprice": "2500.25",
    }
    sdk.position_response = {
        "status": True,
        "data": [position, position | {"netqty": "-5", "buyqty": "0", "sellqty": "5"}],
    }
    broker = AngelOneBroker(instrument_master(), sdk_client=sdk)
    positions = broker.list_positions()
    assert {item.net_quantity for item in positions} == {-5, 20}
    state = PortfolioState(capital_limit=Decimal("100000"))
    before = state.model_dump()
    funds = broker.get_funds()
    assert funds.net == Decimal("100000.25")
    assert funds.available_cash == Decimal("80000.50")
    assert state.model_dump() == before

    sdk.position_response = {"status": True, "data": [position | {"netqty": "bad"}]}
    with pytest.raises(BrokerDataError):
        broker.list_positions()
    sdk.funds_response = {"status": True, "data": {"net": "100"}}
    with pytest.raises(BrokerDataError, match="availablecash"):
        broker.get_funds()


def test_ltp_is_decimal_and_malformed_values_fail() -> None:
    sdk = FakeSDK()
    broker = AngelOneBroker(instrument_master(), sdk_client=sdk)
    quote = broker.get_ltp("RELIANCE", at(10))
    assert quote.ltp == Decimal("2500.25")
    assert sdk.calls[-1] == ("ltpData", ("NSE", "RELIANCE-EQ", "2885"))
    for bad in (None, "NaN", "Infinity", "0"):
        sdk.ltp_response = {"status": True, "data": {"ltp": bad}}
        with pytest.raises(BrokerDataError):
            broker.get_ltp("RELIANCE", at(10))


def test_five_minute_candles_are_canonical_half_open_and_source_preserved() -> None:
    sdk = FakeSDK()
    sdk.candle_response = {
        "status": True,
        "data": [
            ["2026-08-14T09:15:00+05:30", 100, 102, 99, 101, 1000],
            ["2026-08-14T09:20:00+05:30", 101, 103, 100, 102, 1100],
            ["2026-08-14T09:25:00+05:30", 102, 104, 101, 103, 1200],
        ],
    }
    source = copy.deepcopy(sdk.candle_response)
    frame = AngelOneCandleClient(sdk).get_five_minute_candles(
        instrument_master().resolve("RELIANCE"), at(9, 15), at(9, 25)
    )
    assert frame.columns == list(CANONICAL_CANDLE_COLUMNS)
    assert frame.height == 2
    assert frame["symbol"].to_list() == ["RELIANCE", "RELIANCE"]
    assert frame["timestamp"].dtype == pl.Datetime("us", "Asia/Kolkata")
    assert sdk.calls[-1][1]["interval"] == FIVE_MINUTE_INTERVAL
    assert sdk.candle_response == source


@pytest.mark.parametrize(
    "rows",
    [
        [
            ["2026-08-14T09:15:00+05:30", 100, 101, 99, 100, 100],
            ["2026-08-14T09:15:00+05:30", 100, 101, 99, 100, 100],
        ],
        [
            ["2026-08-14T09:20:00+05:30", 100, 101, 99, 100, 100],
            ["2026-08-14T09:15:00+05:30", 100, 101, 99, 100, 100],
        ],
        [["2026-08-14T09:15:00+05:30", 100, None, 99, 100, 100]],
    ],
)
def test_candles_reject_duplicates_unsorted_or_missing_values(rows: list[list[object]]) -> None:
    sdk = FakeSDK()
    sdk.candle_response = {"status": True, "data": rows}
    with pytest.raises(BrokerDataError):
        AngelOneCandleClient(sdk).get_five_minute_candles(
            instrument_master().resolve("RELIANCE"), at(9, 15), at(9, 30)
        )


class FakeWebSocket:
    constructions = 0

    def __init__(self, *args, **kwargs) -> None:
        type(self).constructions += 1
        self.args = args
        self.kwargs = kwargs
        self.calls: list[tuple[str, object]] = []
        self.close_count = 0
        self.connected = Event()
        self.closed = Event()

    def connect(self):
        self.calls.append(("connect", None))
        self.connected.set()
        self.on_open(self)
        self.closed.wait()

    def subscribe(self, correlation_id, mode, token_list):
        self.calls.append(("subscribe", (correlation_id, mode, copy.deepcopy(token_list))))

    def unsubscribe(self, correlation_id, mode, token_list):
        self.calls.append(("unsubscribe", (correlation_id, mode, copy.deepcopy(token_list))))

    def close_connection(self):
        self.close_count += 1
        self.closed.set()


def test_websocket_is_explicit_deterministic_scaled_and_safe() -> None:
    FakeWebSocket.constructions = 0
    ticks = []
    errors = []
    stream = AngelOneMarketDataStream(
        credentials(),
        session(),
        ticks.append,
        errors.append,
        websocket_factory=FakeWebSocket,
    )
    assert FakeWebSocket.constructions == 0
    instrument = instrument_master().resolve("RELIANCE")
    stream.configure_initial_subscription((instrument,))
    connection = Thread(target=stream.connect)
    connection.start()
    assert FakeWebSocket.constructions == 1
    socket = stream._socket
    assert socket.connected.wait(timeout=1)
    assert socket.calls[-1][1][2] == [
        {"exchangeType": NSE_CASH_EXCHANGE_TYPE, "tokens": ["2885"]}
    ]
    timestamp_ms = int(at(10).timestamp() * 1000)
    socket.on_data(
        None,
        {
            "exchange_type": 1,
            "token": "2885",
            "exchange_timestamp": timestamp_ms,
            "last_traded_price": 250025,
        },
    )
    assert ticks[0].instrument == instrument
    assert ticks[0].last_traded_price == Decimal("2500.25")
    assert ticks[0].exchange_timestamp == at(10)

    socket.on_data(
        None,
        {
            "exchange_type": 1,
            "token": "unknown",
            "exchange_timestamp": timestamp_ms,
            "last_traded_price": 1,
        },
    )
    assert len(errors) == 1
    assert len(ticks) == 1
    with pytest.raises(BrokerDataError, match="already subscribed"):
        stream.subscribe((instrument,))
    stream.unsubscribe((instrument,))
    stream.close()
    connection.join(timeout=1)
    assert not connection.is_alive()
    stream.close()
    assert socket.close_count == 1
    assert socket.kwargs["max_retry_attempt"] == 0


@pytest.mark.parametrize(
    ("side", "order_type", "expected_action", "expected_price"),
    [
        (Side.LONG, OrderType.MARKET, "BUY", "0"),
        (Side.SHORT, OrderType.LIMIT, "SELL", "2500"),
    ],
)
def test_live_margin_uses_broker_calculation_without_clamping(
    side: Side,
    order_type: OrderType,
    expected_action: str,
    expected_price: str,
) -> None:
    sdk = FakeSDK()
    provider = AngelOneLiveMarginProvider(sdk, instrument_master())
    assert isinstance(provider, MarginRequirementProvider)
    state = PortfolioState(capital_limit=Decimal("1000"))
    selected = candidate(side=side, order_type=order_type)
    before = (selected.model_dump(), state.model_dump())
    quote = provider.quote(selected, state)
    assert quote.provider_id == LIVE_MARGIN_PROVIDER_ID
    assert quote.required_margin == Decimal("12345.67")
    payload = sdk.calls[-1][1]
    assert payload == {
        "positions": [
            {
                "exchange": "NSE",
                "productType": "INTRADAY",
                "token": "2885",
                "tradeType": expected_action,
                "orderType": order_type.value,
                "qty": 20,
                "price": int(expected_price),
            }
        ]
    }
    assert before == (selected.model_dump(), state.model_dump())
    assert len(sdk.calls) == 1


def test_live_margin_preserves_fractional_limit_price_as_json_number() -> None:
    sdk = FakeSDK()
    selected = candidate(order_type=OrderType.LIMIT).model_copy(
        update={
            "order_intent": candidate(order_type=OrderType.LIMIT).order_intent.model_copy(
                update={"limit_price": Decimal("2500.25")}
            )
        }
    )

    AngelOneLiveMarginProvider(sdk, instrument_master()).quote(
        selected,
        PortfolioState(),
    )

    assert sdk.calls[-1][1]["positions"][0]["price"] == 2500.25


def test_live_margin_sdk_exception_is_sanitized_and_systemic() -> None:
    class FailingSDK(FakeSDK):
        def getMarginApi(self, payload):
            self.calls.append(("getMarginApi", copy.deepcopy(payload)))
            raise TypeError(
                "Authorization: Bearer jwt-secret; token=2885; payload={'positions': []}"
            )

    sdk = FailingSDK()
    with pytest.raises(BrokerSystemicError) as captured:
        AngelOneLiveMarginProvider(sdk, instrument_master()).quote(
            candidate(),
            PortfolioState(),
        )

    detail = str(captured.value)
    assert "TypeError" in detail
    assert "sensitive detail redacted" in detail
    assert "jwt-secret" not in detail
    assert "2885" not in detail
    assert "positions" not in detail
    assert captured.value.__cause__ is None
    assert len(sdk.calls) == 1


def test_live_margin_declared_failure_exposes_only_safe_status_fields() -> None:
    sdk = FakeSDK()
    sdk.margin_response = {
        "status": False,
        "errorcode": "AB4022",
        "message": "Null or Empty Margin Data",
        "data": {"authorization": "secret"},
    }

    with pytest.raises(BrokerApiError) as captured:
        AngelOneLiveMarginProvider(sdk, instrument_master()).quote(
            candidate(),
            PortfolioState(),
        )

    detail = str(captured.value)
    assert "status=False" in detail
    assert "errorcode=AB4022" in detail
    assert "message=Null or Empty Margin Data" in detail
    assert "authorization" not in detail
    assert "secret" not in detail


def test_margin_snapshot_capture_fails_fast_on_first_systemic_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_strategy1_development_backtest as runner

    class Instrument:
        lot_size = 1

    class Master:
        def resolve(self, symbol):
            return Instrument()

    class Quote:
        ltp = Decimal("100")

    class Broker:
        def __init__(self) -> None:
            self.ltp_calls = 0
            self.logout_calls = 0

        def authenticate(self, credentials, captured_at):
            return object()

        def authenticated_client(self):
            return object()

        def get_ltp(self, symbol, captured_at):
            self.ltp_calls += 1
            return Quote()

        def logout(self, session):
            self.logout_calls += 1

    class Provider:
        def __init__(self) -> None:
            self.calls = 0

        def quote(self, candidate, state):
            self.calls += 1
            raise BrokerSystemicError("ConnectionError: service unavailable")

    broker = Broker()
    provider = Provider()
    monkeypatch.setattr(
        runner,
        "required_margin_pairs",
        lambda requests: {("AAA", Side.LONG), ("BBB", Side.SHORT)},
    )
    monkeypatch.setattr(runner, "latest_margin_snapshot", lambda: None)
    monkeypatch.setattr(runner, "load_smartapi_credentials", object)
    monkeypatch.setattr(runner, "fetch_instrument_master", Master)
    monkeypatch.setattr(runner, "AngelOneBroker", lambda instrument_master: broker)
    monkeypatch.setattr(
        runner,
        "AngelOneLiveMarginProvider",
        lambda sdk, instrument_master: provider,
    )

    with pytest.raises(RuntimeError, match="first representative shared/systemic"):
        runner.capture_or_expand_margin_snapshot(requests=[], captured_at=at(12))

    assert provider.calls == 1
    assert broker.ltp_calls == 1
    assert broker.logout_calls == 1


@pytest.mark.parametrize("value", [None, "0", "-1", "NaN", "Infinity", "bad"])
def test_live_margin_rejects_invalid_required_margin(value: object) -> None:
    sdk = FakeSDK()
    sdk.margin_response = {"status": True, "data": {"totalMarginRequired": value}}
    with pytest.raises(BrokerDataError):
        AngelOneLiveMarginProvider(sdk, instrument_master()).quote(
            candidate(), PortfolioState()
        )


def snapshot() -> HistoricalMarginSnapshot:
    return create_historical_margin_snapshot(
        snapshot_id="angel-margin-2026-08-14",
        captured_at=at(12),
        source_as_of_date=date(2026, 8, 14),
        broker_sdk_version="1.5.5",
        entries=(
            create_margin_snapshot_entry(
                "RELIANCE", Side.SHORT, Decimal("100000"), Decimal("25000")
            ),
            create_margin_snapshot_entry(
                "RELIANCE", Side.LONG, Decimal("100000"), Decimal("20000")
            ),
        ),
    )


def test_historical_snapshot_is_exact_immutable_and_validated() -> None:
    selected = snapshot()
    assert selected.broker_architecture_version == BROKER_ARCHITECTURE_VERSION
    assert selected.calculation_method == HISTORICAL_MARGIN_CALCULATION_METHOD
    assert selected.entries[0].side is Side.LONG
    assert selected.entries[0].required_margin_fraction == Decimal("0.2")
    assert selected.entries[0].leverage_equivalent == Decimal("5")
    with pytest.raises(ValidationError):
        selected.snapshot_id = "changed"
    with pytest.raises(ValidationError, match="timezone-aware"):
        create_historical_margin_snapshot(
            snapshot_id="bad",
            captured_at=datetime(2026, 8, 14),
            source_as_of_date=date(2026, 8, 14),
            entries=selected.entries,
        )
    with pytest.raises(ValidationError, match="unique"):
        HistoricalMarginSnapshot(
            **(selected.model_dump(exclude={"entries"}) | {"entries": (selected.entries[0],) * 2})
        )
    with pytest.raises(ValidationError, match="must equal"):
        HistoricalMarginSnapshotEntry(
            symbol="RELIANCE",
            side=Side.LONG,
            reference_notional=Decimal("100000"),
            broker_required_margin=Decimal("20000"),
            required_margin_fraction=Decimal("0.3"),
        )
    with pytest.raises(BrokerDataError, match="positive"):
        create_margin_snapshot_entry(
            "RELIANCE", Side.LONG, Decimal("0"), Decimal("20000")
        )
    with pytest.raises(ValidationError):
        HistoricalMarginSnapshot(
            **(selected.model_dump() | {"broker_id": "OTHER"})
        )


def test_historical_provider_uses_requested_notional_and_side_without_leverage() -> None:
    provider = HistoricalMarginRequirementProvider(snapshot())
    assert isinstance(provider, MarginRequirementProvider)
    state = PortfolioState(capital_limit=Decimal("1000"))
    assert provider.quote(candidate(notional=50_000), state).required_margin == Decimal("10000.0")
    short_quote = provider.quote(candidate(side=Side.SHORT, notional=100_000), state)
    assert short_quote.required_margin == Decimal("25000.00")
    assert short_quote.provider_id.endswith("angel-margin-2026-08-14")
    missing = candidate().model_copy(
        update={
            "order_intent": candidate().order_intent.model_copy(
                update={
                    "signal": candidate().order_intent.signal.model_copy(update={"symbol": "OTHER"})
                }
            )
        }
    )
    with pytest.raises(LookupError):
        provider.quote(missing, state)


def test_snapshot_json_is_deterministic_verified_and_non_overwriting(tmp_path: Path) -> None:
    selected = snapshot()
    first = save_historical_margin_snapshot(selected, tmp_path / "snapshot.json")
    second = save_historical_margin_snapshot(selected, tmp_path / "other.json")
    assert first.read_bytes() == second.read_bytes()
    assert load_historical_margin_snapshot(first) == selected
    loaded_provider = HistoricalMarginRequirementProvider(load_historical_margin_snapshot(first))
    assert loaded_provider.quote(candidate(), PortfolioState()) == (
        HistoricalMarginRequirementProvider(selected).quote(candidate(), PortfolioState())
    )
    assert "secret" not in first.read_text(encoding="utf-8").lower()
    with pytest.raises(FileExistsError):
        save_historical_margin_snapshot(selected, first)

    envelope = json.loads(first.read_text(encoding="utf-8"))
    envelope["snapshot"]["entries"][0]["broker_required_margin"] = "99999"
    first.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(BrokerDataError, match="fingerprint"):
        load_historical_margin_snapshot(first)
