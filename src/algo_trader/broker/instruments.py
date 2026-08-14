"""Pure Angel One OpenAPI instrument-master parsing and exact NSE EQ resolution."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from urllib.request import urlopen

from algo_trader.broker.exceptions import BrokerApiError, BrokerDataError, BrokerInstrumentError
from algo_trader.broker.models import BrokerInstrument

ANGEL_ONE_INSTRUMENT_MASTER_URL = (
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
)


class AngelOneInstrumentMaster:
    """Deterministic exact NSE cash-equity index."""

    def __init__(self, instruments: Sequence[BrokerInstrument]) -> None:
        selected = tuple(sorted(instruments, key=lambda item: (item.symbol, item.symbol_token)))
        if not selected:
            raise BrokerInstrumentError("instrument master contains no NSE EQ instruments")
        by_symbol: dict[str, list[BrokerInstrument]] = {}
        by_token: dict[str, list[BrokerInstrument]] = {}
        for instrument in selected:
            by_symbol.setdefault(instrument.symbol, []).append(instrument)
            by_token.setdefault(instrument.symbol_token, []).append(instrument)
        self._instruments = selected
        self._by_symbol = {key: tuple(value) for key, value in by_symbol.items()}
        self._by_token = {key: tuple(value) for key, value in by_token.items()}

    @property
    def instruments(self) -> tuple[BrokerInstrument, ...]:
        """Return the deterministic immutable instrument ordering."""
        return self._instruments

    def resolve(self, symbol: str) -> BrokerInstrument:
        """Resolve one exact internal symbol to exactly one NSE ``-EQ`` record."""
        matches = self._by_symbol.get(symbol, ())
        if not matches:
            raise BrokerInstrumentError(f"no exact NSE EQ instrument for symbol: {symbol}")
        if len(matches) != 1:
            raise BrokerInstrumentError(f"ambiguous NSE EQ instrument for symbol: {symbol}")
        return matches[0]

    def resolve_token(self, symbol_token: str) -> BrokerInstrument:
        """Resolve one exact broker token without loose matching."""
        matches = self._by_token.get(str(symbol_token), ())
        if not matches:
            raise BrokerInstrumentError(f"unknown NSE EQ symbol token: {symbol_token}")
        if len(matches) != 1:
            raise BrokerInstrumentError(f"ambiguous NSE EQ symbol token: {symbol_token}")
        return matches[0]


def parse_instrument_master(payload: bytes | str | object) -> AngelOneInstrumentMaster:
    """Parse decoded or JSON OpenAPIScripMaster content without network access."""
    if isinstance(payload, bytes):
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BrokerDataError("instrument master is not valid UTF-8 JSON") from error
    elif isinstance(payload, str):
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as error:
            raise BrokerDataError("instrument master is not valid JSON") from error
    else:
        decoded = payload
    if not isinstance(decoded, list):
        raise BrokerDataError("instrument master root must be a JSON array")

    instruments = []
    for row in decoded:
        if not isinstance(row, Mapping):
            raise BrokerDataError("instrument master rows must be JSON objects")

        # OpenAPIScripMaster is a mixed all-instrument file. Filter irrelevant rows
        # before requiring fields that are mandatory only for our NSE cash-equity
        # subset. This prevents malformed/non-equity rows from aborting the entire
        # master while preserving fail-closed validation for candidate NSE -EQ rows.
        exchange_segment = str(row.get("exch_seg") or "").strip().upper()
        trading_symbol = str(row.get("symbol") or "").strip()
        instrument_type = str(row.get("instrumenttype") or "").strip().upper()

        if exchange_segment != "NSE":
            continue
        if not trading_symbol.endswith("-EQ"):
            continue
        if instrument_type not in {"", "EQ"}:
            continue

        # From this point onward the row claims to be an NSE cash equity, so all
        # required fields remain strict. Bad candidate rows must not be accepted.
        name = _text(row, "name")
        if trading_symbol != f"{name}-EQ":
            continue

        lot_size = _positive_integer(row.get("lotsize"), "lotsize")
        raw_tick_size = _positive_decimal(row.get("tick_size"), "tick_size")
        instruments.append(
            BrokerInstrument(
                symbol=name,
                trading_symbol=trading_symbol,
                symbol_token=_text(row, "token"),
                exchange_segment=exchange_segment,
                lot_size=lot_size,
                tick_size=raw_tick_size / Decimal("100"),
            )
        )
    return AngelOneInstrumentMaster(instruments)


def fetch_instrument_master(
    url: str = ANGEL_ONE_INSTRUMENT_MASTER_URL,
    opener: Callable[..., object] = urlopen,
) -> AngelOneInstrumentMaster:
    """Explicitly fetch the official master; callers control when network occurs."""
    try:
        response = opener(url, timeout=30)
        payload = response.read()
    except Exception as error:
        raise BrokerApiError("instrument-master fetch failed") from error
    return parse_instrument_master(payload)


def _text(row: Mapping[object, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise BrokerDataError(f"instrument master field is missing or invalid: {key}")
    return value.strip()


def _positive_decimal(value: object, name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as error:
        raise BrokerDataError(f"instrument {name} must be numeric") from error
    if not result.is_finite() or result <= 0:
        raise BrokerDataError(f"instrument {name} must be finite and positive")
    return result


def _positive_integer(value: object, name: str) -> int:
    numeric = _positive_decimal(value, name)
    if numeric != numeric.to_integral_value():
        raise BrokerDataError(f"instrument {name} must be an integer")
    return int(numeric)
