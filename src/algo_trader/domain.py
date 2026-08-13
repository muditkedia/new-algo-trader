"""Validated, broker-neutral records shared by the trading domain."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Any

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_serializer,
    model_validator,
)

MIN_NOTIONAL = 50_000
MAX_NOTIONAL = 100_000
NOTIONAL_INCREMENT = 5_000


class Side(StrEnum):
    """Direction of a strategy signal or position."""

    LONG = "LONG"
    SHORT = "SHORT"


class SignalStatus(StrEnum):
    """Lifecycle result of a valid strategy signal."""

    GENERATED = "GENERATED"
    EXECUTED = "EXECUTED"
    CAPACITY_REJECTED = "CAPACITY_REJECTED"


class ExitReason(StrEnum):
    """Small, strategy-neutral set of reasons for closing a position."""

    TARGET_REACHED = "TARGET_REACHED"
    STOP_LOSS = "STOP_LOSS"
    TIME_EXIT = "TIME_EXIT"
    MANUAL = "MANUAL"
    STRATEGY_EXIT = "STRATEGY_EXIT"


class OrderType(StrEnum):
    """Broker-neutral order styles required by an execution adapter."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"


def _validate_aware_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value


def _validate_notional(value: int) -> int:
    if value % NOTIONAL_INCREMENT != 0:
        raise ValueError(f"notional must be in increments of {NOTIONAL_INCREMENT}")
    return value


def _freeze_snapshot(value: Any) -> Any:
    """Detach snapshot data from callers and make nested containers immutable."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_snapshot(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_snapshot(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze_snapshot(item) for item in value)
    return deepcopy(value)


def _thaw_snapshot(value: Any) -> Any:
    """Return ordinary containers when Pydantic serializes an immutable snapshot."""
    if isinstance(value, Mapping):
        return {key: _thaw_snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_thaw_snapshot(item) for item in value]
    if isinstance(value, set | frozenset):
        return [_thaw_snapshot(item) for item in value]
    return value


AwareDateTime = Annotated[datetime, AfterValidator(_validate_aware_datetime)]
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Notional = Annotated[
    int,
    Field(strict=True, ge=MIN_NOTIONAL, le=MAX_NOTIONAL),
    AfterValidator(_validate_notional),
]
PositiveDecimal = Annotated[Decimal, Field(gt=0, allow_inf_nan=False)]
FiniteDecimal = Annotated[Decimal, Field(allow_inf_nan=False)]


class DomainModel(BaseModel):
    """Common validation policy for immutable domain records."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Signal(DomainModel):
    """A strategy decision containing only information known when generated."""

    strategy_id: NonEmptyStr
    strategy_version: NonEmptyStr
    symbol: NonEmptyStr
    timestamp: AwareDateTime
    side: Side
    strategy_parameters: Mapping[str, Any] = Field(default_factory=dict)
    feature_snapshot: Mapping[str, Any] = Field(default_factory=dict)
    status: SignalStatus = SignalStatus.GENERATED

    @model_validator(mode="after")
    def freeze_generation_snapshots(self) -> Signal:
        object.__setattr__(self, "strategy_parameters", _freeze_snapshot(self.strategy_parameters))
        object.__setattr__(self, "feature_snapshot", _freeze_snapshot(self.feature_snapshot))
        return self

    @field_serializer("strategy_parameters", "feature_snapshot")
    def serialize_snapshot(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return _thaw_snapshot(value)


class MLScore(DomainModel):
    """Advisory prediction captured before a trade outcome is known."""

    model_version: NonEmptyStr
    quality_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    calibrated_probability: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    predicted_net_return: float = Field(
        allow_inf_nan=False,
        description="Predicted net return fraction; 0.005 represents +0.5%.",
    )
    recommended_notional: Notional


class OrderIntent(DomainModel):
    """Broker-neutral request to execute a generated strategy signal."""

    signal: Signal
    timestamp: AwareDateTime
    quantity: int = Field(strict=True, gt=0)
    requested_notional: Notional
    order_type: OrderType = OrderType.MARKET
    limit_price: PositiveDecimal | None = None

    @model_validator(mode="after")
    def validate_execution_request(self) -> OrderIntent:
        if self.signal.status is not SignalStatus.GENERATED:
            raise ValueError("an order intent requires a signal with GENERATED status")
        if self.timestamp < self.signal.timestamp:
            raise ValueError("order intent timestamp cannot precede the signal timestamp")
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("a limit order requires limit_price")
        if self.order_type is OrderType.MARKET and self.limit_price is not None:
            raise ValueError("a market order cannot specify limit_price")
        return self


class Fill(DomainModel):
    """A broker-neutral actual or simulated execution fill."""

    timestamp: AwareDateTime
    price: PositiveDecimal
    quantity: int = Field(strict=True, gt=0)
    slippage_per_unit: FiniteDecimal = Decimal("0")
    is_simulated: bool


class Trade(DomainModel):
    """A completed executed trade or capacity-rejected shadow trade."""

    signal: Signal
    ml_score: MLScore
    target_notional: Notional
    entry_fill: Fill
    exit_fill: Fill
    gross_pnl: FiniteDecimal
    total_costs: Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]
    net_pnl: FiniteDecimal
    mfe_return: Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]
    mae_return: Annotated[Decimal, Field(le=0, allow_inf_nan=False)]
    exit_reason: ExitReason
    is_shadow: bool = False

    @property
    def actual_entry_notional(self) -> Decimal:
        """Actual filled entry value, before costs."""
        return self.entry_fill.price * self.entry_fill.quantity

    @property
    def gross_return(self) -> Decimal:
        """Gross P&L as a fraction of actual entry notional."""
        return self.gross_pnl / self.actual_entry_notional

    @property
    def net_return(self) -> Decimal:
        """Net P&L as a fraction of actual entry notional."""
        return self.net_pnl / self.actual_entry_notional

    @model_validator(mode="after")
    def validate_completed_trade(self) -> Trade:
        if self.entry_fill.timestamp < self.signal.timestamp:
            raise ValueError("entry fill timestamp cannot precede signal timestamp")
        if self.exit_fill.timestamp < self.entry_fill.timestamp:
            raise ValueError("exit fill timestamp cannot precede entry fill timestamp")
        if self.exit_fill.quantity != self.entry_fill.quantity:
            raise ValueError("entry and exit fill quantities must match")
        if self.net_pnl != self.gross_pnl - self.total_costs:
            raise ValueError("net_pnl must equal gross_pnl minus total_costs")
        if self.target_notional != self.ml_score.recommended_notional:
            raise ValueError("target_notional must equal the ML-recommended notional")

        if self.is_shadow:
            if self.signal.status is not SignalStatus.CAPACITY_REJECTED:
                raise ValueError("a shadow trade requires a CAPACITY_REJECTED signal")
            if not self.entry_fill.is_simulated or not self.exit_fill.is_simulated:
                raise ValueError("a shadow trade requires simulated entry and exit fills")
        elif self.signal.status is not SignalStatus.EXECUTED:
            raise ValueError("a non-shadow trade requires an EXECUTED signal")

        return self
