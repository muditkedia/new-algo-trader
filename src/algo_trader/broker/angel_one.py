"""Official SmartAPI 1.5.5 authentication, orders, trades, positions, and funds."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from datetime import datetime
from decimal import Decimal, InvalidOperation
from importlib.metadata import version
from urllib.parse import quote
from zoneinfo import ZoneInfo

import pyotp
from SmartApi.smartConnect import SmartConnect

from algo_trader.broker.exceptions import (
    BrokerAmbiguousStateError,
    BrokerApiError,
    BrokerAuthenticationError,
    BrokerDataError,
)
from algo_trader.broker.instruments import AngelOneInstrumentMaster
from algo_trader.broker.models import (
    AngelOneCredentials,
    AngelOneSession,
    BrokerCancellationAcknowledgement,
    BrokerFunds,
    BrokerOrderAcknowledgement,
    BrokerOrderRequest,
    BrokerOrderSnapshot,
    BrokerOrderState,
    BrokerPosition,
    BrokerQuote,
    BrokerTradeFill,
    BrokerTransactionAction,
)
from algo_trader.domain import OrderType, Side

SMARTAPI_SDK_VERSION = version("smartapi-python")
MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")
ANGEL_VARIETY = "NORMAL"
ANGEL_PRODUCT_TYPE = "INTRADAY"
ANGEL_DURATION = "DAY"


def entry_action(side: Side) -> BrokerTransactionAction:
    """Return the explicit transaction action required to open a side."""
    if side is Side.LONG:
        return BrokerTransactionAction.BUY
    if side is Side.SHORT:
        return BrokerTransactionAction.SELL
    raise TypeError("side must be Side.LONG or Side.SHORT")


def exit_action(side: Side) -> BrokerTransactionAction:
    """Return the explicit opposite transaction action required to close a side."""
    if side is Side.LONG:
        return BrokerTransactionAction.SELL
    if side is Side.SHORT:
        return BrokerTransactionAction.BUY
    raise TypeError("side must be Side.LONG or Side.SHORT")


def broker_order_tag(client_order_id: str) -> str:
    """Map caller identity to a deterministic 19-character Angel order tag."""
    if not isinstance(client_order_id, str) or not client_order_id.strip():
        raise ValueError("client_order_id must be a non-empty string")
    digest = hashlib.sha256(client_order_id.encode("utf-8")).hexdigest()
    return f"NAT{digest[:16]}"


def angel_order_payload(request: BrokerOrderRequest) -> dict[str, str]:
    """Translate one broker-neutral request to the fixed Angel NSE intraday contract."""
    if not isinstance(request, BrokerOrderRequest):
        raise TypeError("request must be a BrokerOrderRequest")
    payload = {
        "variety": ANGEL_VARIETY,
        "tradingsymbol": request.instrument.trading_symbol,
        "symboltoken": request.instrument.symbol_token,
        "transactiontype": request.transaction_action.value,
        "exchange": request.instrument.exchange.value,
        "ordertype": request.order_type.value,
        "producttype": ANGEL_PRODUCT_TYPE,
        "duration": ANGEL_DURATION,
        "quantity": str(request.quantity),
        "price": (
            "0"
            if request.order_type is OrderType.MARKET
            else _decimal_text(request.limit_price)
        ),
        "ordertag": broker_order_tag(request.client_order_id),
    }
    if request.scrip_consent:
        payload["scripconsent"] = "yes"
    return payload


class AngelOneBroker:
    """Small synchronous adapter over documented public SmartConnect operations."""

    def __init__(
        self,
        instrument_master: AngelOneInstrumentMaster,
        *,
        sdk_factory: Callable[..., object] = SmartConnect,
        sdk_client: object | None = None,
    ) -> None:
        if not isinstance(instrument_master, AngelOneInstrumentMaster):
            raise TypeError("instrument_master must be an AngelOneInstrumentMaster")
        self._instrument_master = instrument_master
        self._sdk_factory = sdk_factory
        self._sdk = sdk_client

    def authenticate(
        self,
        credentials: AngelOneCredentials,
        occurred_at: datetime,
    ) -> AngelOneSession:
        """Construct SmartConnect explicitly and calculate TOTP at caller time."""
        if not isinstance(credentials, AngelOneCredentials):
            raise TypeError("credentials must be AngelOneCredentials")
        _require_aware(occurred_at, "occurred_at")
        try:
            sdk = self._sdk_factory(api_key=credentials.api_key.get_secret_value())
            totp = pyotp.TOTP(credentials.totp_secret.get_secret_value()).at(
                int(occurred_at.timestamp())
            )
            response = sdk.generateSession(
                credentials.client_code,
                credentials.pin.get_secret_value(),
                totp,
            )
        except Exception as error:
            raise BrokerAuthenticationError("Angel One authentication operation failed") from error
        try:
            data = _successful_data(
                response, BrokerAuthenticationError, "Angel One authentication"
            )
            session = _session_from_data(credentials.client_code, data, occurred_at)
        except BrokerDataError as error:
            raise BrokerAuthenticationError(
                "Angel One authentication response lacks required session data"
            ) from error
        self._sdk = sdk
        return session

    def refresh_session(
        self,
        session: AngelOneSession,
        refreshed_at: datetime,
    ) -> AngelOneSession:
        """Explicitly refresh tokens and return a new immutable session."""
        _require_aware(refreshed_at, "refreshed_at")
        sdk = self._require_sdk()
        try:
            response = sdk.generateToken(session.refresh_token.get_secret_value())
        except Exception as error:
            raise BrokerAuthenticationError("Angel One token refresh operation failed") from error
        try:
            data = _successful_data(
                response, BrokerAuthenticationError, "Angel One token refresh"
            )
            combined = dict(data)
            combined.setdefault("refreshToken", session.refresh_token.get_secret_value())
            return _session_from_data(session.client_code, combined, refreshed_at)
        except BrokerDataError as error:
            raise BrokerAuthenticationError(
                "Angel One token refresh response lacks required session data"
            ) from error

    def logout(self, session: AngelOneSession) -> None:
        """Explicitly terminate the caller-selected client session once."""
        sdk = self._require_sdk()
        try:
            response = sdk.terminateSession(session.client_code)
        except Exception as error:
            raise BrokerApiError("Angel One logout operation failed") from error
        _successful_response(response, "Angel One logout")

    def place_order(
        self,
        request: BrokerOrderRequest,
        acknowledged_at: datetime,
    ) -> BrokerOrderAcknowledgement:
        """Submit exactly once and return acknowledgement, never a domain fill."""
        _require_aware(acknowledged_at, "acknowledged_at")
        sdk = self._require_sdk()
        payload = angel_order_payload(request)
        try:
            response = sdk.placeOrderFullResponse(payload)
        except Exception as error:
            raise BrokerApiError("Angel One order placement operation failed") from error
        data = _successful_data(response, BrokerApiError, "Angel One order placement")
        order_id = _required_text(data, "orderid", "order acknowledgement")
        return BrokerOrderAcknowledgement(
            client_order_id=request.client_order_id,
            broker_order_tag=payload["ordertag"],
            broker_order_id=order_id,
            unique_order_id=_optional_text(data.get("uniqueorderid")),
            acknowledged_at=acknowledged_at,
            raw_status=True,
            raw_message=_optional_text(response.get("message")),
            raw_error_code=_optional_text(response.get("errorcode")),
        )

    def cancel_order(
        self,
        broker_order_id: str,
        acknowledged_at: datetime,
    ) -> BrokerCancellationAcknowledgement:
        """Submit one NORMAL cancellation request without asserting final fill state."""
        _require_aware(acknowledged_at, "acknowledged_at")
        broker_order_id = _nonempty_argument(broker_order_id, "broker_order_id")
        sdk = self._require_sdk()
        try:
            response = sdk.cancelOrder(broker_order_id, ANGEL_VARIETY)
        except Exception as error:
            raise BrokerApiError("Angel One cancellation operation failed") from error
        _successful_response(response, "Angel One cancellation")
        return BrokerCancellationAcknowledgement(
            broker_order_id=broker_order_id,
            acknowledged_at=acknowledged_at,
            raw_status=True,
            raw_message=_optional_text(response.get("message")),
            raw_error_code=_optional_text(response.get("errorcode")),
        )

    def list_orders(self) -> tuple[BrokerOrderSnapshot, ...]:
        """Return normalized order-book rows in deterministic order."""
        sdk = self._require_sdk()
        try:
            response = sdk.orderBook()
        except Exception as error:
            raise BrokerApiError("Angel One order-book operation failed") from error
        rows = _successful_rows(response, "Angel One order book")
        snapshots = tuple(self._parse_order(row) for row in rows)
        return tuple(sorted(snapshots, key=lambda item: item.broker_order_id))

    def get_order(
        self,
        *,
        unique_order_id: str | None = None,
        broker_order_id: str | None = None,
        broker_order_tag: str | None = None,
    ) -> BrokerOrderSnapshot:
        """Resolve exact broker evidence and reject ambiguous order-book matches."""
        if unique_order_id is not None:
            unique_order_id = _nonempty_argument(unique_order_id, "unique_order_id")
        if broker_order_id is not None:
            broker_order_id = _nonempty_argument(broker_order_id, "broker_order_id")
        if broker_order_tag is not None:
            broker_order_tag = _nonempty_argument(broker_order_tag, "broker_order_tag")
        if unique_order_id is None and broker_order_id is None and broker_order_tag is None:
            raise ValueError("one exact order identity is required")
        if unique_order_id is not None:
            sdk = self._require_sdk()
            try:
                response = sdk.individual_order_details(
                    f"?uniqueorderid={quote(unique_order_id, safe='')}"
                )
            except Exception as error:
                raise BrokerApiError("Angel One individual-order operation failed") from error
            normalized = _successful_response(response, "Angel One individual order")
            data = normalized.get("data")
            if isinstance(data, Mapping):
                rows = (data,)
            elif isinstance(data, list):
                rows = tuple(_mapping(row, "individual order row") for row in data)
            else:
                raise BrokerDataError("Angel One individual-order data is invalid")
            candidates = tuple(self._parse_order(row) for row in rows)
        else:
            candidates = self.list_orders()
        matches = tuple(
            item
            for item in candidates
            if (unique_order_id is None or item.unique_order_id == unique_order_id)
            and (broker_order_id is None or item.broker_order_id == broker_order_id)
            and (broker_order_tag is None or item.broker_order_tag == broker_order_tag)
        )
        if not matches:
            raise LookupError("no broker order matches the exact supplied identity")
        if len(matches) != 1:
            raise BrokerAmbiguousStateError("multiple broker orders match the exact identity")
        return matches[0]

    def list_trade_fills(self) -> tuple[BrokerTradeFill, ...]:
        """Preserve every trade-book fill and sort deterministic broker evidence."""
        sdk = self._require_sdk()
        try:
            response = sdk.tradeBook()
        except Exception as error:
            raise BrokerApiError("Angel One trade-book operation failed") from error
        rows = _successful_rows(response, "Angel One trade book")
        fills = tuple(self._parse_trade_fill(row) for row in rows)
        return tuple(
            sorted(
                fills,
                key=lambda item: (
                    item.fill_timestamp,
                    item.broker_order_id,
                    item.fill_id,
                ),
            )
        )

    def list_positions(self) -> tuple[BrokerPosition, ...]:
        """Return every parseable current position without hiding malformed rows."""
        sdk = self._require_sdk()
        try:
            response = sdk.position()
        except Exception as error:
            raise BrokerApiError("Angel One position operation failed") from error
        rows = _successful_rows(response, "Angel One positions")
        positions = tuple(self._parse_position(row) for row in rows)
        return tuple(
            sorted(
                positions,
                key=lambda item: (
                    item.instrument.symbol,
                    item.instrument.symbol_token,
                    item.net_quantity,
                ),
            )
        )

    def get_funds(self) -> BrokerFunds:
        """Normalize external RMS values without changing internal portfolio capital."""
        sdk = self._require_sdk()
        try:
            response = sdk.rmsLimit()
        except Exception as error:
            raise BrokerApiError("Angel One RMS operation failed") from error
        data = _successful_data(response, BrokerApiError, "Angel One RMS")
        return BrokerFunds(
            net=_required_decimal(data, "net", "funds"),
            available_cash=_required_decimal(data, "availablecash", "funds"),
            available_intraday_payin=_optional_decimal(data.get("availableintradaypayin")),
            available_limit_margin=_optional_decimal(data.get("availablelimitmargin")),
            collateral=_optional_decimal(data.get("collateral")),
            m2m_unrealized=_optional_decimal(data.get("m2munrealized")),
            m2m_realized=_optional_decimal(data.get("m2mrealized")),
            utilised_debits=_optional_decimal(data.get("utiliseddebits")),
            utilised_span=_optional_decimal(data.get("utilisedspan")),
            utilised_exposure=_optional_decimal(data.get("utilisedexposure")),
        )

    def authenticated_client(self) -> object:
        """Return the authenticated SDK boundary for broker-owned provider adapters."""
        return self._require_sdk()

    def get_ltp(self, symbol: str, observed_at: datetime) -> BrokerQuote:
        """Request one exact NSE LTP and normalize monetary fields to Decimal."""
        _require_aware(observed_at, "observed_at")
        instrument = self._instrument_master.resolve(symbol)
        sdk = self._require_sdk()
        try:
            response = sdk.ltpData(
                instrument.exchange.value,
                instrument.trading_symbol,
                instrument.symbol_token,
            )
        except Exception as error:
            raise BrokerApiError("Angel One LTP operation failed") from error
        data = _successful_data(response, BrokerApiError, "Angel One LTP")
        return BrokerQuote(
            instrument=instrument,
            observed_at=observed_at,
            ltp=_required_positive_decimal(data, "ltp", "quote"),
            open=_optional_positive_decimal(data.get("open")),
            high=_optional_positive_decimal(data.get("high")),
            low=_optional_positive_decimal(data.get("low")),
            close=_optional_positive_decimal(data.get("close")),
        )

    def _parse_order(self, row: object) -> BrokerOrderSnapshot:
        data = _mapping(row, "order row")
        token = _required_text_any(data, ("symboltoken", "symbolToken"), "order")
        instrument = self._instrument_master.resolve_token(token)
        requested = _required_nonnegative_int_any(data, ("quantity", "orderquantity"), "order")
        if requested <= 0:
            raise BrokerDataError("order requested quantity must be positive")
        filled = _required_nonnegative_int_any(data, ("filledshares", "filledquantity"), "order")
        remaining_value = _first(data, ("unfilledshares", "remainingquantity"))
        remaining = requested - filled if remaining_value is None else _nonnegative_int(
            remaining_value, "order remaining quantity"
        )
        raw_status = _optional_text(data.get("status"))
        raw_order_status = _optional_text(data.get("orderstatus"))
        return BrokerOrderSnapshot(
            broker_order_id=_required_text_any(data, ("orderid", "order_id"), "order"),
            unique_order_id=_optional_text(_first(data, ("uniqueorderid", "unique_order_id"))),
            broker_order_tag=_optional_text(_first(data, ("ordertag", "orderTag"))),
            instrument=instrument,
            transaction_action=_action(_first(data, ("transactiontype", "transactionType"))),
            order_type=_order_type(_first(data, ("ordertype", "orderType"))),
            requested_quantity=requested,
            filled_quantity=filled,
            remaining_quantity=remaining,
            average_price=_required_nonnegative_decimal_any(
                data, ("averageprice", "averagePrice"), "order"
            ),
            state=normalize_order_state(raw_status, raw_order_status, requested, filled),
            raw_status=raw_status,
            raw_order_status=raw_order_status,
            raw_text=_optional_text(_first(data, ("text", "statusmessage"))),
            updated_at=_optional_broker_timestamp(
                _first(data, ("updatetime", "orderupdatetime"))
            ),
            exchange_timestamp=_optional_broker_timestamp(
                _first(data, ("exchtime", "exchorderupdatetime"))
            ),
        )

    def _parse_trade_fill(self, row: object) -> BrokerTradeFill:
        data = _mapping(row, "trade row")
        instrument = self._instrument_master.resolve_token(
            _required_text_any(data, ("symboltoken", "symbolToken"), "trade")
        )
        return BrokerTradeFill(
            broker_order_id=_required_text_any(data, ("orderid", "order_id"), "trade"),
            fill_id=_required_text_any(data, ("fillid", "tradeid"), "trade"),
            instrument=instrument,
            transaction_action=_action(_first(data, ("transactiontype", "transactionType"))),
            fill_timestamp=_required_broker_timestamp(
                _first(data, ("filltime", "tradetime", "exchtime")), "fill timestamp"
            ),
            fill_price=_required_positive_decimal_any(
                data, ("fillprice", "tradeprice"), "trade"
            ),
            quantity=_required_positive_int_any(
                data, ("fillsize", "fillshares", "quantity"), "trade"
            ),
        )

    def _parse_position(self, row: object) -> BrokerPosition:
        data = _mapping(row, "position row")
        instrument = self._instrument_master.resolve_token(
            _required_text_any(data, ("symboltoken", "symbolToken"), "position")
        )
        product_type = _required_text_any(data, ("producttype", "productType"), "position")
        if product_type != ANGEL_PRODUCT_TYPE:
            raise BrokerDataError("unexpected non-INTRADAY position returned by Broker v1")
        return BrokerPosition(
            instrument=instrument,
            product_type=product_type,
            net_quantity=_required_int_any(data, ("netqty", "netquantity"), "position"),
            buy_quantity=_required_nonnegative_int_any(
                data, ("buyqty", "buyquantity"), "position"
            ),
            sell_quantity=_required_nonnegative_int_any(
                data, ("sellqty", "sellquantity"), "position"
            ),
            buy_average_price=_required_nonnegative_decimal_any(
                data, ("buyavgprice", "buyaverageprice"), "position"
            ),
            sell_average_price=_required_nonnegative_decimal_any(
                data, ("sellavgprice", "sellaverageprice"), "position"
            ),
            net_average_price=_optional_decimal(
                _first(data, ("netprice", "netaverageprice"))
            ),
        )

    def _require_sdk(self) -> object:
        if self._sdk is None:
            raise BrokerAuthenticationError("an authenticated Angel One SDK client is required")
        return self._sdk


def normalize_order_state(
    raw_status: str | None,
    raw_order_status: str | None,
    requested_quantity: int,
    filled_quantity: int,
) -> BrokerOrderState:
    """Normalize known text conservatively while preserving unknown raw values elsewhere."""
    selected_values = {
        value.lower().strip() for value in (raw_order_status, raw_status) if value
    }
    selected = " ".join(sorted(selected_values))
    if "reject" in selected:
        return BrokerOrderState.REJECTED
    if "cancel" in selected:
        return BrokerOrderState.CANCELLED
    if 0 < filled_quantity < requested_quantity:
        return BrokerOrderState.PARTIALLY_FILLED
    if filled_quantity == requested_quantity:
        return BrokerOrderState.FILLED
    if selected_values & {"open", "open order"}:
        return BrokerOrderState.OPEN
    if "pending" in selected:
        return BrokerOrderState.PENDING
    if selected_values & {"submitted", "put order req received", "validation pending"}:
        return BrokerOrderState.SUBMITTED
    return BrokerOrderState.UNKNOWN


def _session_from_data(
    client_code: str,
    data: Mapping[str, object],
    when: datetime,
) -> AngelOneSession:
    return AngelOneSession(
        client_code=_optional_text(data.get("clientcode")) or client_code,
        jwt_token=_required_text(data, "jwtToken", "session"),
        refresh_token=_required_text(data, "refreshToken", "session"),
        feed_token=_required_text(data, "feedToken", "session"),
        authenticated_at=when,
        sdk_version=SMARTAPI_SDK_VERSION,
    )


def _successful_response(response: object, operation: str) -> Mapping[str, object]:
    data = _mapping(response, f"{operation} response")
    if data.get("status") is not True:
        message = _optional_text(data.get("message")) or "broker declared failure"
        error_code = _optional_text(data.get("errorcode"))
        suffix = f" ({error_code})" if error_code else ""
        raise BrokerApiError(f"{operation} failed: {message}{suffix}")
    return data


def _successful_data(
    response: object,
    error_type: type[BrokerApiError | BrokerAuthenticationError],
    operation: str,
) -> Mapping[str, object]:
    try:
        normalized = _successful_response(response, operation)
    except BrokerApiError as error:
        raise error_type(str(error)) from error
    return _mapping(normalized.get("data"), f"{operation} data")


def _successful_rows(response: object, operation: str) -> tuple[Mapping[str, object], ...]:
    normalized = _successful_response(response, operation)
    rows = normalized.get("data")
    if not isinstance(rows, list):
        raise BrokerDataError(f"{operation} data must be a list")
    return tuple(_mapping(row, f"{operation} row") for row in rows)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise BrokerDataError(f"{name} must be an object")
    return value


def _first(data: Mapping[str, object], names: tuple[str, ...]) -> object:
    for name in names:
        if name in data:
            return data[name]
    return None


def _required_text(data: Mapping[str, object], key: str, context: str) -> str:
    value = data.get(key)
    result = _optional_text(value)
    if result is None:
        raise BrokerDataError(f"{context} requires non-empty {key}")
    return result


def _required_text_any(data: Mapping[str, object], keys: tuple[str, ...], context: str) -> str:
    result = _optional_text(_first(data, keys))
    if result is None:
        raise BrokerDataError(f"{context} requires non-empty {keys[0]}")
    return result


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _nonempty_argument(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _decimal(value: object, context: str) -> Decimal:
    if value is None or isinstance(value, bool):
        raise BrokerDataError(f"{context} must be numeric")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as error:
        raise BrokerDataError(f"{context} must be numeric") from error
    if not result.is_finite():
        raise BrokerDataError(f"{context} must be finite")
    return result


def _required_decimal(data: Mapping[str, object], key: str, context: str) -> Decimal:
    if key not in data:
        raise BrokerDataError(f"{context} requires {key}")
    return _decimal(data[key], f"{context} {key}")


def _optional_decimal(value: object) -> Decimal | None:
    return None if value is None else _decimal(value, "optional broker decimal")


def _optional_positive_decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    result = _decimal(value, "optional broker price")
    if result <= 0:
        raise BrokerDataError("optional broker price must be positive")
    return result


def _required_positive_decimal(data: Mapping[str, object], key: str, context: str) -> Decimal:
    result = _required_decimal(data, key, context)
    if result <= 0:
        raise BrokerDataError(f"{context} {key} must be positive")
    return result


def _required_nonnegative_decimal_any(
    data: Mapping[str, object], keys: tuple[str, ...], context: str
) -> Decimal:
    result = _decimal(_first(data, keys), f"{context} {keys[0]}")
    if result < 0:
        raise BrokerDataError(f"{context} {keys[0]} must be non-negative")
    return result


def _required_positive_decimal_any(
    data: Mapping[str, object], keys: tuple[str, ...], context: str
) -> Decimal:
    result = _decimal(_first(data, keys), f"{context} {keys[0]}")
    if result <= 0:
        raise BrokerDataError(f"{context} {keys[0]} must be positive")
    return result


def _integer(value: object, context: str) -> int:
    result = _decimal(value, context)
    if result != result.to_integral_value():
        raise BrokerDataError(f"{context} must be an integer")
    return int(result)


def _nonnegative_int(value: object, context: str) -> int:
    result = _integer(value, context)
    if result < 0:
        raise BrokerDataError(f"{context} must be non-negative")
    return result


def _required_int_any(data: Mapping[str, object], keys: tuple[str, ...], context: str) -> int:
    return _integer(_first(data, keys), f"{context} {keys[0]}")


def _required_nonnegative_int_any(
    data: Mapping[str, object], keys: tuple[str, ...], context: str
) -> int:
    return _nonnegative_int(_first(data, keys), f"{context} {keys[0]}")


def _required_positive_int_any(
    data: Mapping[str, object], keys: tuple[str, ...], context: str
) -> int:
    result = _integer(_first(data, keys), f"{context} {keys[0]}")
    if result <= 0:
        raise BrokerDataError(f"{context} {keys[0]} must be positive")
    return result


def _action(value: object) -> BrokerTransactionAction:
    try:
        return BrokerTransactionAction(str(value).upper())
    except ValueError as error:
        raise BrokerDataError("broker transaction action is invalid") from error


def _order_type(value: object) -> OrderType:
    try:
        return OrderType(str(value).upper())
    except ValueError as error:
        raise BrokerDataError("broker order type is invalid") from error


def _required_broker_timestamp(value: object, context: str) -> datetime:
    result = _optional_broker_timestamp(value)
    if result is None:
        raise BrokerDataError(f"{context} is missing or malformed")
    return result


def _optional_broker_timestamp(value: object) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        for pattern in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M", "%Y-%m-%d %H:%M:%S"):
            try:
                parsed = datetime.strptime(text, pattern)
                break
            except ValueError:
                continue
        else:
            raise BrokerDataError("broker timestamp is malformed") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=MARKET_TIMEZONE)
    return parsed.astimezone(MARKET_TIMEZONE)


def _require_aware(value: datetime, name: str) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _decimal_text(value: Decimal | None) -> str:
    if value is None:
        raise ValueError("Decimal value is required")
    return format(value, "f")
