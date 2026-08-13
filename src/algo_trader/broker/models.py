"""Immutable broker-neutral models for Angel One integration evidence."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StringConstraints,
    computed_field,
    model_validator,
)

from algo_trader.domain import OrderType, Side

BROKER_ARCHITECTURE_VERSION = "1"
HISTORICAL_MARGIN_CALCULATION_METHOD = "BROKER_DERIVED_LINEAR_MARGIN_FRACTION_V1"
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveDecimal = Annotated[Decimal, Field(gt=0, allow_inf_nan=False)]
NonNegativeDecimal = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]
FiniteDecimal = Annotated[Decimal, Field(allow_inf_nan=False)]


class FrozenBrokerModel(BaseModel):
    """Validation policy for immutable broker records."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class BrokerExchange(StrEnum):
    """Broker exchanges supported by Broker v1."""

    NSE = "NSE"


class BrokerTransactionAction(StrEnum):
    """Explicit execution action, independent of position side."""

    BUY = "BUY"
    SELL = "SELL"


class BrokerOrderState(StrEnum):
    """Conservative normalized broker order lifecycle."""

    SUBMITTED = "SUBMITTED"
    OPEN = "OPEN"
    PENDING = "PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"


class AngelOneCredentials(FrozenBrokerModel):
    """Caller-supplied secrets; Broker never reads an environment or .env file."""

    api_key: SecretStr = Field(min_length=1)
    client_code: NonEmptyStr
    pin: SecretStr = Field(min_length=1)
    totp_secret: SecretStr = Field(min_length=1)


class AngelOneSession(FrozenBrokerModel):
    """Normalized immutable authenticated-session tokens."""

    client_code: NonEmptyStr
    jwt_token: SecretStr = Field(min_length=1)
    refresh_token: SecretStr = Field(min_length=1)
    feed_token: SecretStr = Field(min_length=1)
    authenticated_at: datetime
    sdk_version: NonEmptyStr

    @model_validator(mode="after")
    def validate_time(self) -> AngelOneSession:
        _require_aware(self.authenticated_at, "authenticated_at")
        return self


class BrokerInstrument(FrozenBrokerModel):
    """Exact NSE cash-equity mapping from internal symbol to Angel token."""

    symbol: NonEmptyStr
    trading_symbol: NonEmptyStr
    symbol_token: NonEmptyStr
    exchange: BrokerExchange = BrokerExchange.NSE
    exchange_segment: NonEmptyStr = "NSE"
    lot_size: int = Field(strict=True, gt=0)
    tick_size: PositiveDecimal

    @model_validator(mode="after")
    def validate_nse_equity(self) -> BrokerInstrument:
        if self.exchange is not BrokerExchange.NSE or self.exchange_segment != "NSE":
            raise ValueError("Broker v1 instruments must be NSE cash equities")
        if self.trading_symbol != f"{self.symbol}-EQ":
            raise ValueError("trading_symbol must be the exact internal-symbol EQ contract")
        return self


class BrokerOrderRequest(FrozenBrokerModel):
    """Explicit ordinary NSE intraday order request."""

    client_order_id: NonEmptyStr
    instrument: BrokerInstrument
    transaction_action: BrokerTransactionAction
    order_type: OrderType
    quantity: int = Field(strict=True, gt=0)
    limit_price: PositiveDecimal | None = None
    submitted_at: datetime
    scrip_consent: bool = False

    @model_validator(mode="after")
    def validate_request(self) -> BrokerOrderRequest:
        _require_aware(self.submitted_at, "submitted_at")
        if self.order_type is OrderType.MARKET and self.limit_price is not None:
            raise ValueError("MARKET orders cannot have limit_price")
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("LIMIT orders require limit_price")
        return self


class BrokerOrderAcknowledgement(FrozenBrokerModel):
    """Placement acknowledgement only; it is never evidence of a fill."""

    client_order_id: NonEmptyStr
    broker_order_tag: NonEmptyStr
    broker_order_id: NonEmptyStr
    unique_order_id: str | None = None
    acknowledged_at: datetime
    raw_status: bool
    raw_message: str | None = None
    raw_error_code: str | None = None

    @model_validator(mode="after")
    def validate_time(self) -> BrokerOrderAcknowledgement:
        _require_aware(self.acknowledged_at, "acknowledged_at")
        return self


class BrokerCancellationAcknowledgement(FrozenBrokerModel):
    """Cancellation request acknowledgement, not final no-fill proof."""

    broker_order_id: NonEmptyStr
    acknowledged_at: datetime
    raw_status: bool
    raw_message: str | None = None
    raw_error_code: str | None = None

    @model_validator(mode="after")
    def validate_time(self) -> BrokerCancellationAcknowledgement:
        _require_aware(self.acknowledged_at, "acknowledged_at")
        return self


class BrokerOrderSnapshot(FrozenBrokerModel):
    """Normalized broker order evidence, including partial fills and raw state."""

    broker_order_id: NonEmptyStr
    unique_order_id: str | None
    broker_order_tag: str | None
    instrument: BrokerInstrument
    transaction_action: BrokerTransactionAction
    order_type: OrderType
    requested_quantity: int = Field(strict=True, gt=0)
    filled_quantity: int = Field(strict=True, ge=0)
    remaining_quantity: int = Field(strict=True, ge=0)
    average_price: NonNegativeDecimal
    state: BrokerOrderState
    raw_status: str | None
    raw_order_status: str | None
    raw_text: str | None
    updated_at: datetime | None
    exchange_timestamp: datetime | None

    @model_validator(mode="after")
    def validate_quantities(self) -> BrokerOrderSnapshot:
        if self.filled_quantity > self.requested_quantity:
            raise ValueError("filled_quantity cannot exceed requested_quantity")
        if self.remaining_quantity > self.requested_quantity:
            raise ValueError("remaining_quantity cannot exceed requested_quantity")
        if self.updated_at is not None:
            _require_aware(self.updated_at, "updated_at")
        if self.exchange_timestamp is not None:
            _require_aware(self.exchange_timestamp, "exchange_timestamp")
        return self


class BrokerTradeFill(FrozenBrokerModel):
    """One actual broker trade-book fill; multiple fills per order are preserved."""

    broker_order_id: NonEmptyStr
    fill_id: NonEmptyStr
    instrument: BrokerInstrument
    transaction_action: BrokerTransactionAction
    fill_timestamp: datetime
    fill_price: PositiveDecimal
    quantity: int = Field(strict=True, gt=0)

    @model_validator(mode="after")
    def validate_time(self) -> BrokerTradeFill:
        _require_aware(self.fill_timestamp, "fill_timestamp")
        return self


class BrokerPosition(FrozenBrokerModel):
    """Current broker position evidence for reconciliation and safety."""

    instrument: BrokerInstrument
    product_type: NonEmptyStr
    net_quantity: int = Field(strict=True)
    buy_quantity: int = Field(strict=True, ge=0)
    sell_quantity: int = Field(strict=True, ge=0)
    buy_average_price: NonNegativeDecimal
    sell_average_price: NonNegativeDecimal
    net_average_price: FiniteDecimal | None = None


class BrokerFunds(FrozenBrokerModel):
    """External Angel RMS truth, distinct from internal PortfolioState capital."""

    net: FiniteDecimal
    available_cash: FiniteDecimal
    available_intraday_payin: FiniteDecimal | None = None
    available_limit_margin: FiniteDecimal | None = None
    collateral: FiniteDecimal | None = None
    m2m_unrealized: FiniteDecimal | None = None
    m2m_realized: FiniteDecimal | None = None
    utilised_debits: FiniteDecimal | None = None
    utilised_span: FiniteDecimal | None = None
    utilised_exposure: FiniteDecimal | None = None


class BrokerQuote(FrozenBrokerModel):
    """One observed immutable broker quote."""

    instrument: BrokerInstrument
    observed_at: datetime
    ltp: PositiveDecimal
    open: PositiveDecimal | None = None
    high: PositiveDecimal | None = None
    low: PositiveDecimal | None = None
    close: PositiveDecimal | None = None

    @model_validator(mode="after")
    def validate_time(self) -> BrokerQuote:
        _require_aware(self.observed_at, "observed_at")
        return self


class BrokerMarketTick(FrozenBrokerModel):
    """Normalized NSE cash websocket tick in Decimal rupees."""

    instrument: BrokerInstrument
    exchange_timestamp: datetime
    last_traded_price: PositiveDecimal
    cumulative_volume: int | None = Field(default=None, strict=True, ge=0)

    @model_validator(mode="after")
    def validate_time(self) -> BrokerMarketTick:
        _require_aware(self.exchange_timestamp, "exchange_timestamp")
        return self


class HistoricalMarginSnapshotEntry(FrozenBrokerModel):
    """Broker-derived symbol/side reference margin ratio."""

    symbol: NonEmptyStr
    side: Side
    reference_notional: PositiveDecimal
    broker_required_margin: PositiveDecimal
    required_margin_fraction: PositiveDecimal

    @model_validator(mode="after")
    def validate_fraction(self) -> HistoricalMarginSnapshotEntry:
        if self.required_margin_fraction != self.broker_required_margin / self.reference_notional:
            raise ValueError("required_margin_fraction must equal margin / reference notional")
        return self

    @computed_field
    @property
    def leverage_equivalent(self) -> Decimal:
        """Diagnostic only; never the historical calculation source."""
        return self.reference_notional / self.broker_required_margin


class HistoricalMarginSnapshot(FrozenBrokerModel):
    """Frozen broker-derived current margin assumption for reproducible research."""

    snapshot_id: NonEmptyStr
    broker_id: Literal["ANGEL_ONE"] = "ANGEL_ONE"
    broker_sdk_version: NonEmptyStr
    broker_architecture_version: Literal["1"] = BROKER_ARCHITECTURE_VERSION
    captured_at: datetime
    source_as_of_date: date
    calculation_method: Literal[
        "BROKER_DERIVED_LINEAR_MARGIN_FRACTION_V1"
    ] = HISTORICAL_MARGIN_CALCULATION_METHOD
    entries: tuple[HistoricalMarginSnapshotEntry, ...]

    @model_validator(mode="after")
    def validate_snapshot(self) -> HistoricalMarginSnapshot:
        _require_aware(self.captured_at, "captured_at")
        if not self.entries:
            raise ValueError("historical margin snapshot entries must not be empty")
        keys = [(entry.symbol, entry.side) for entry in self.entries]
        if len(keys) != len(set(keys)):
            raise ValueError("historical margin snapshot symbol/side entries must be unique")
        if self.entries != tuple(sorted(self.entries, key=lambda item: (item.symbol, item.side))):
            raise ValueError("historical margin snapshot entries must be deterministically sorted")
        return self


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
