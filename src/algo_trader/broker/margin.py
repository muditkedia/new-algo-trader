"""Live Angel margin and frozen broker-derived historical margin assumptions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from importlib.metadata import version
from pathlib import Path

from algo_trader.broker.angel_one import ANGEL_PRODUCT_TYPE, entry_action
from algo_trader.broker.exceptions import (
    BrokerApiError,
    BrokerDataError,
    BrokerSystemicError,
)
from algo_trader.broker.instruments import AngelOneInstrumentMaster
from algo_trader.broker.models import (
    BROKER_ARCHITECTURE_VERSION,
    HISTORICAL_MARGIN_CALCULATION_METHOD,
    HistoricalMarginSnapshot,
    HistoricalMarginSnapshotEntry,
)
from algo_trader.domain import OrderType, Side
from algo_trader.portfolio import (
    AllocationCandidate,
    MarginRequirementQuote,
    PortfolioState,
)

LIVE_MARGIN_PROVIDER_ID = "ANGEL_ONE:LIVE_MARGIN_V1"


class AngelOneLiveMarginProvider:
    """Live broker-derived margin quote using SmartConnect.getMarginApi exactly once."""

    def __init__(
        self,
        sdk_client: object,
        instrument_master: AngelOneInstrumentMaster,
    ) -> None:
        self._sdk = sdk_client
        self._instrument_master = instrument_master

    def quote(
        self,
        candidate: AllocationCandidate,
        state: PortfolioState,
    ) -> MarginRequirementQuote:
        """Return Angel's required margin without leverage assumptions or resizing."""
        if not isinstance(candidate, AllocationCandidate):
            raise TypeError("candidate must be an AllocationCandidate")
        if not isinstance(state, PortfolioState):
            raise TypeError("state must be a PortfolioState")
        order = candidate.order_intent
        instrument = self._instrument_master.resolve(order.signal.symbol)
        payload = {
            "positions": [
                {
                    "exchange": instrument.exchange.value,
                    "productType": ANGEL_PRODUCT_TYPE,
                    "token": instrument.symbol_token,
                    "tradeType": entry_action(order.signal.side).value,
                    "orderType": order.order_type.value,
                    "qty": order.quantity,
                    "price": (
                        0
                        if order.order_type is OrderType.MARKET
                        else _json_number(order.limit_price)
                    ),
                }
            ]
        }
        try:
            response = self._sdk.getMarginApi(payload)
        except Exception as error:
            raise BrokerSystemicError(
                "Angel One live-margin operation failed: "
                + _safe_exception_summary(error)
            ) from None
        if not isinstance(response, Mapping):
            raise BrokerDataError("Angel One live-margin response must be an object")
        if response.get("status") is not True:
            raw_errorcode = response.get("errorcode")
            raw_message = response.get("message")
            status = _safe_diagnostic_text(response.get("status"), "MISSING")
            errorcode = _safe_diagnostic_text(raw_errorcode, "UNSPECIFIED")
            message = _safe_diagnostic_text(
                raw_message,
                "broker declared failure",
            )
            error_type = (
                BrokerSystemicError
                if _is_clearly_systemic(raw_errorcode, raw_message)
                else BrokerApiError
            )
            raise error_type(
                "Angel One live-margin request failed: "
                f"status={status} "
                f"errorcode={errorcode} message={message}"
            )
        data = response.get("data")
        if not isinstance(data, Mapping) or "totalMarginRequired" not in data:
            raise BrokerDataError("Angel One live-margin data is missing totalMarginRequired")
        required_margin = _positive_decimal(
            data["totalMarginRequired"], "totalMarginRequired"
        )
        return MarginRequirementQuote(
            provider_id=LIVE_MARGIN_PROVIDER_ID,
            required_margin=required_margin,
        )


class HistoricalMarginRequirementProvider:
    """Deterministic linear quote from one frozen broker-derived snapshot."""

    def __init__(self, snapshot: HistoricalMarginSnapshot) -> None:
        if not isinstance(snapshot, HistoricalMarginSnapshot):
            raise TypeError("snapshot must be a HistoricalMarginSnapshot")
        self.snapshot = snapshot
        self._entries = {(entry.symbol, entry.side): entry for entry in snapshot.entries}

    def quote(
        self,
        candidate: AllocationCandidate,
        state: PortfolioState,
    ) -> MarginRequirementQuote:
        """Scale requested notional by the exact stored fraction; never by leverage."""
        if not isinstance(candidate, AllocationCandidate):
            raise TypeError("candidate must be an AllocationCandidate")
        if not isinstance(state, PortfolioState):
            raise TypeError("state must be a PortfolioState")
        signal = candidate.order_intent.signal
        entry = self._entries.get((signal.symbol, signal.side))
        if entry is None:
            raise LookupError(
                f"historical margin snapshot lacks {signal.symbol}/{signal.side.value}"
            )
        required = Decimal(candidate.order_intent.requested_notional) * (
            entry.required_margin_fraction
        )
        return MarginRequirementQuote(
            provider_id=f"ANGEL_ONE:HISTORICAL_MARGIN:{self.snapshot.snapshot_id}",
            required_margin=required,
        )


def create_margin_snapshot_entry(
    symbol: str,
    side: Side,
    reference_notional: Decimal,
    broker_required_margin: Decimal,
) -> HistoricalMarginSnapshotEntry:
    """Create one exact broker-derived ratio without storing a leverage constant."""
    validated_notional = _positive_decimal(reference_notional, "reference_notional")
    validated_margin = _positive_decimal(broker_required_margin, "broker_required_margin")
    return HistoricalMarginSnapshotEntry(
        symbol=symbol,
        side=side,
        reference_notional=validated_notional,
        broker_required_margin=validated_margin,
        required_margin_fraction=validated_margin / validated_notional,
    )


def create_historical_margin_snapshot(
    *,
    snapshot_id: str,
    captured_at: datetime,
    source_as_of_date: date,
    entries: Sequence[HistoricalMarginSnapshotEntry],
    broker_sdk_version: str | None = None,
) -> HistoricalMarginSnapshot:
    """Construct a sorted frozen snapshot from explicitly supplied reference entries."""
    return HistoricalMarginSnapshot(
        snapshot_id=snapshot_id,
        broker_sdk_version=broker_sdk_version or version("smartapi-python"),
        broker_architecture_version=BROKER_ARCHITECTURE_VERSION,
        captured_at=captured_at,
        source_as_of_date=source_as_of_date,
        calculation_method=HISTORICAL_MARGIN_CALCULATION_METHOD,
        entries=tuple(sorted(entries, key=lambda item: (item.symbol, item.side))),
    )


def save_historical_margin_snapshot(
    snapshot: HistoricalMarginSnapshot,
    output_path: Path,
) -> Path:
    """Write deterministic fingerprinted JSON without overwriting an artifact."""
    if not isinstance(snapshot, HistoricalMarginSnapshot):
        raise TypeError("snapshot must be a HistoricalMarginSnapshot")
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(f"historical margin snapshot already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_payload = snapshot.model_dump(
        mode="json",
        exclude_computed_fields=True,
    )
    canonical_snapshot = _canonical_json(snapshot_payload)
    envelope = {
        "fingerprint_algorithm": "sha256",
        "snapshot_fingerprint": hashlib.sha256(canonical_snapshot.encode("utf-8")).hexdigest(),
        "snapshot": snapshot_payload,
    }
    path.write_text(_canonical_json(envelope), encoding="utf-8", newline="\n")
    return path


def load_historical_margin_snapshot(input_path: Path) -> HistoricalMarginSnapshot:
    """Verify the canonical snapshot fingerprint before returning an immutable model."""
    path = Path(input_path)
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BrokerDataError("historical margin snapshot is unreadable") from error
    if not isinstance(envelope, Mapping):
        raise BrokerDataError("historical margin snapshot envelope must be an object")
    payload = envelope.get("snapshot")
    fingerprint = envelope.get("snapshot_fingerprint")
    if envelope.get("fingerprint_algorithm") != "sha256" or not isinstance(
        payload, Mapping
    ):
        raise BrokerDataError("historical margin snapshot envelope is invalid")
    expected = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    if fingerprint != expected:
        raise BrokerDataError("historical margin snapshot fingerprint verification failed")
    try:
        return HistoricalMarginSnapshot.model_validate(payload)
    except Exception as error:
        raise BrokerDataError("historical margin snapshot data is invalid") from error


def _positive_decimal(value: object, context: str) -> Decimal:
    if value is None or isinstance(value, bool):
        raise BrokerDataError(f"{context} must be numeric")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as error:
        raise BrokerDataError(f"{context} must be numeric") from error
    if not result.is_finite() or result <= 0:
        raise BrokerDataError(f"{context} must be finite and positive")
    return result


def _json_number(value: Decimal | None) -> int | float:
    if value is None:
        raise BrokerDataError("LIMIT margin request requires limit_price")
    if value == value.to_integral_value():
        return int(value)
    return float(value)


def _safe_exception_summary(error: Exception) -> str:
    return f"{type(error).__name__}: {_safe_diagnostic_text(str(error), 'no detail')}"


def _safe_diagnostic_text(value: object, fallback: str) -> str:
    raw_text = fallback if value is None else str(value)
    text = " ".join(raw_text.split()) or fallback
    lowered = text.lower()
    sensitive_markers = (
        "authorization",
        "bearer",
        "api_key",
        "apikey",
        "client_code",
        "feedtoken",
        "refreshtoken",
        "jwt",
        "password",
        "privatekey",
        "totp",
        "token",
    )
    if any(marker in lowered for marker in sensitive_markers) or any(
        marker in text for marker in ("{", "}", "[", "]")
    ):
        return "[sensitive detail redacted]"
    return text[:300] + ("..." if len(text) > 300 else "")


def _is_clearly_systemic(errorcode: object, message: object) -> bool:
    detail = f"{errorcode} {message}".lower()
    return any(
        marker in detail
        for marker in (
            "access denied",
            "authentication",
            "forbidden",
            "internal server",
            "rate limit",
            "service unavailable",
            "session",
            "timeout",
            "token",
            "too many requests",
            "unauthorized",
        )
    )


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
