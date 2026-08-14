"""SmartAPI REST candle normalization and minimal NSE cash websocket wrapper."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from zoneinfo import ZoneInfo

import polars as pl
from SmartApi.smartWebSocketV2 import SmartWebSocketV2

from algo_trader.broker.exceptions import BrokerApiError, BrokerDataError
from algo_trader.broker.models import (
    AngelOneCredentials,
    AngelOneSession,
    BrokerInstrument,
    BrokerMarketTick,
)
from algo_trader.data import CANONICAL_CANDLE_COLUMNS

MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")
FIVE_MINUTE_INTERVAL = "FIVE_MINUTE"
NSE_CASH_EXCHANGE_TYPE = 1
LTP_STREAM_MODE = 1
STREAM_CORRELATION_ID = "NATSTRM1"


class AngelOneCandleClient:
    """Explicit SmartConnect candle adapter with canonical half-open output."""

    def __init__(self, sdk_client: object) -> None:
        self._sdk = sdk_client

    def get_five_minute_candles(
        self,
        instrument: BrokerInstrument,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """Normalize official six-field rows without filling, sorting, or deduplication."""
        normalized_start = _boundary(start, "start")
        normalized_end = _boundary(end, "end")
        if normalized_start >= normalized_end:
            raise ValueError("start must be earlier than end")
        params = {
            "exchange": instrument.exchange.value,
            "symboltoken": instrument.symbol_token,
            "interval": FIVE_MINUTE_INTERVAL,
            "fromdate": normalized_start.strftime("%Y-%m-%d %H:%M"),
            "todate": normalized_end.strftime("%Y-%m-%d %H:%M"),
        }
        try:
            response = self._sdk.getCandleData(params)
        except Exception as error:
            raise BrokerApiError("Angel One candle operation failed") from error
        payload = _successful_response(response, "Angel One candles")
        rows = payload.get("data")
        if not isinstance(rows, list):
            raise BrokerDataError("Angel One candle data must be a list")

        normalized_rows = []
        previous: datetime | None = None
        for row in rows:
            if not isinstance(row, Sequence) or isinstance(row, str | bytes) or len(row) != 6:
                raise BrokerDataError("Angel One candle row must have exactly six values")
            timestamp = _broker_timestamp(row[0], "candle timestamp")
            if previous is not None and timestamp <= previous:
                qualifier = "duplicate" if timestamp == previous else "unsorted"
                raise BrokerDataError(f"Angel One candle timestamps are {qualifier}")
            previous = timestamp
            values = tuple(_finite_decimal(value, "candle value") for value in row[1:])
            if any(value <= 0 for value in values[:4]):
                raise BrokerDataError("candle OHLC values must be positive")
            if values[4] < 0:
                raise BrokerDataError("candle volume must be non-negative")
            if normalized_start <= timestamp < normalized_end:
                normalized_rows.append(
                    {
                        "timestamp": timestamp,
                        "open": float(values[0]),
                        "high": float(values[1]),
                        "low": float(values[2]),
                        "close": float(values[3]),
                        "volume": float(values[4]),
                        "symbol": instrument.symbol,
                    }
                )
        schema = {
            "timestamp": pl.Datetime(time_unit="us", time_zone="Asia/Kolkata"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "symbol": pl.String,
        }
        return pl.DataFrame(normalized_rows, schema=schema, strict=False).select(
            CANONICAL_CANDLE_COLUMNS
        )


class AngelOneMarketDataStream:
    """Minimal explicit-connect wrapper over official SmartWebSocketV2 parsed messages."""

    def __init__(
        self,
        credentials: AngelOneCredentials,
        session: AngelOneSession,
        on_tick: Callable[[BrokerMarketTick], None],
        on_error: Callable[[BrokerDataError], None],
        *,
        websocket_factory: Callable[..., object] = SmartWebSocketV2,
    ) -> None:
        self._credentials = credentials
        self._session = session
        self._on_tick_callback = on_tick
        self._on_error_callback = on_error
        self._factory = websocket_factory
        self._socket: object | None = None
        self._instruments_by_token: dict[str, BrokerInstrument] = {}
        self._initial_instruments: tuple[BrokerInstrument, ...] = ()
        self._closed = False

    def configure_initial_subscription(
        self, instruments: Sequence[BrokerInstrument]
    ) -> None:
        """Configure the exact subscription sent by SDK ``on_open``."""
        if self._socket is not None:
            raise RuntimeError("initial subscription must be configured before connect")
        if self._initial_instruments:
            raise RuntimeError("initial subscription is already configured")
        self._initial_instruments = self._validated_instruments(instruments)

    def connect(self) -> None:
        """Construct and connect the SDK socket only on this explicit call."""
        if self._socket is not None:
            raise RuntimeError("market-data stream is already connected")
        try:
            socket = self._factory(
                self._session.jwt_token.get_secret_value(),
                self._credentials.api_key.get_secret_value(),
                self._session.client_code,
                self._session.feed_token.get_secret_value(),
                max_retry_attempt=0,
            )
            socket.on_data = self._handle_data
            socket.on_error = self._handle_sdk_error
            socket.on_open = self._handle_open
            self._socket = socket
            socket.connect()
        except Exception as error:
            self._socket = None
            raise BrokerApiError("Angel One websocket connection failed") from error

    def subscribe(self, instruments: Sequence[BrokerInstrument]) -> None:
        """Subscribe exact unique NSE tokens in deterministic order."""
        socket = self._require_socket()
        selected = self._validated_instruments(instruments)
        self._subscribe_socket(socket, selected)

    def _subscribe_socket(
        self, socket: object, selected: tuple[BrokerInstrument, ...]
    ) -> None:
        tokens = [item.symbol_token for item in selected]
        if any(token in self._instruments_by_token for token in tokens):
            raise BrokerDataError("instrument token is already subscribed")
        token_list = [{"exchangeType": NSE_CASH_EXCHANGE_TYPE, "tokens": tokens}]
        try:
            socket.subscribe(STREAM_CORRELATION_ID, LTP_STREAM_MODE, token_list)
        except Exception as error:
            raise BrokerApiError("Angel One websocket subscription failed") from error
        self._instruments_by_token.update({item.symbol_token: item for item in selected})

    def _handle_open(self, socket: object, *args: object) -> None:
        del args
        if not self._initial_instruments:
            return
        try:
            self._subscribe_socket(socket, self._initial_instruments)
        except (BrokerApiError, BrokerDataError) as error:
            self._on_error_callback(error)

    def unsubscribe(self, instruments: Sequence[BrokerInstrument]) -> None:
        """Unsubscribe an exact deterministic set of currently mapped tokens."""
        socket = self._require_socket()
        selected = tuple(sorted(instruments, key=lambda item: item.symbol_token))
        tokens = [item.symbol_token for item in selected]
        if not tokens or any(token not in self._instruments_by_token for token in tokens):
            raise BrokerDataError("unsubscribe requires currently subscribed instruments")
        token_list = [{"exchangeType": NSE_CASH_EXCHANGE_TYPE, "tokens": tokens}]
        try:
            socket.unsubscribe(STREAM_CORRELATION_ID, LTP_STREAM_MODE, token_list)
        except Exception as error:
            raise BrokerApiError("Angel One websocket unsubscription failed") from error
        for token in tokens:
            del self._instruments_by_token[token]

    def normalize_tick(self, message: object) -> BrokerMarketTick:
        """Normalize one official parsed SDK message, rejecting unknown tokens."""
        if not isinstance(message, Mapping):
            raise BrokerDataError("websocket message must be an object")
        if message.get("exchange_type") != NSE_CASH_EXCHANGE_TYPE:
            raise BrokerDataError("websocket message is not NSE cash")
        token = str(message.get("token") or "")
        instrument = self._instruments_by_token.get(token)
        if instrument is None:
            raise BrokerDataError(f"websocket message has unknown token: {token}")
        raw_price = _integer(message.get("last_traded_price"), "last_traded_price")
        if raw_price <= 0:
            raise BrokerDataError("last_traded_price must be positive")
        raw_timestamp = _integer(message.get("exchange_timestamp"), "exchange_timestamp")
        seconds = raw_timestamp / 1000 if raw_timestamp > 100_000_000_000 else raw_timestamp
        timestamp = datetime.fromtimestamp(seconds, tz=UTC).astimezone(MARKET_TIMEZONE)
        volume_value = message.get("volume_trade_for_the_day")
        volume = None if volume_value is None else _integer(volume_value, "cumulative volume")
        if volume is not None and volume < 0:
            raise BrokerDataError("cumulative volume must be non-negative")
        return BrokerMarketTick(
            instrument=instrument,
            exchange_timestamp=timestamp,
            last_traded_price=Decimal(raw_price) / Decimal("100"),
            cumulative_volume=volume,
        )

    def close(self) -> None:
        """Delegate close at most once; no destructor or reconnect loop exists."""
        if self._socket is None or self._closed:
            return
        try:
            self._socket.close_connection()
        except Exception as error:
            raise BrokerApiError("Angel One websocket close failed") from error
        self._closed = True

    def _handle_data(self, _socket: object, message: object) -> None:
        try:
            tick = self.normalize_tick(message)
        except BrokerDataError as error:
            self._on_error_callback(error)
            return
        self._on_tick_callback(tick)

    def _handle_sdk_error(self, *args: object) -> None:
        self._on_error_callback(BrokerDataError("Angel One websocket reported an error"))

    @staticmethod
    def _validated_instruments(
        instruments: Sequence[BrokerInstrument],
    ) -> tuple[BrokerInstrument, ...]:
        selected = tuple(sorted(instruments, key=lambda item: item.symbol_token))
        if not selected:
            raise ValueError("at least one instrument is required")
        tokens = [item.symbol_token for item in selected]
        if len(tokens) != len(set(tokens)):
            raise BrokerDataError("duplicate instrument token in subscription")
        return selected

    def _require_socket(self) -> object:
        if self._socket is None:
            raise RuntimeError("market-data stream must be connected explicitly")
        return self._socket


def _successful_response(response: object, operation: str) -> Mapping[str, object]:
    if not isinstance(response, Mapping):
        raise BrokerDataError(f"{operation} response must be an object")
    if response.get("status") is not True:
        message = str(response.get("message") or "broker declared failure")
        raise BrokerApiError(f"{operation} failed: {message}")
    return response


def _boundary(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(MARKET_TIMEZONE)


def _broker_timestamp(value: object, context: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise BrokerDataError(f"{context} must be a timestamp string")
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError as error:
        raise BrokerDataError(f"{context} is malformed") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=MARKET_TIMEZONE)
    return parsed.astimezone(MARKET_TIMEZONE)


def _finite_decimal(value: object, context: str) -> Decimal:
    if value is None or isinstance(value, bool):
        raise BrokerDataError(f"{context} must be numeric")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as error:
        raise BrokerDataError(f"{context} must be numeric") from error
    if not result.is_finite():
        raise BrokerDataError(f"{context} must be finite")
    return result


def _integer(value: object, context: str) -> int:
    numeric = _finite_decimal(value, context)
    if numeric != numeric.to_integral_value() or not math.isfinite(float(numeric)):
        raise BrokerDataError(f"{context} must be an integer")
    return int(numeric)
