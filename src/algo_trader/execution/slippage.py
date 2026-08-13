"""Deterministic, broker-neutral execution slippage models."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Protocol, runtime_checkable

_BASIS_POINTS_DENOMINATOR = Decimal("10000")


class ExecutionAction(StrEnum):
    """Transaction action, intentionally distinct from LONG/SHORT position side."""

    BUY = "BUY"
    SELL = "SELL"


@runtime_checkable
class SlippageModel(Protocol):
    """Structural contract for deterministic execution-price adjustment."""

    def apply(self, raw_price: Decimal, action: ExecutionAction) -> Decimal:
        """Return the final execution price after adverse slippage."""
        ...


@dataclass(frozen=True, slots=True)
class NoSlippage:
    """Leave the raw execution price unchanged."""

    def apply(self, raw_price: Decimal, action: ExecutionAction) -> Decimal:
        _validate_application(raw_price, action)
        return raw_price


@dataclass(frozen=True, slots=True)
class FixedBasisPointsSlippage:
    """Apply a fixed adverse percentage to every execution price."""

    basis_points: Decimal

    def __post_init__(self) -> None:
        try:
            normalized = (
                self.basis_points
                if isinstance(self.basis_points, Decimal)
                else Decimal(str(self.basis_points))
            )
        except (InvalidOperation, ValueError) as error:
            raise ValueError("basis_points must be a finite non-negative Decimal") from error
        if not normalized.is_finite() or normalized < 0:
            raise ValueError("basis_points must be a finite non-negative Decimal")
        object.__setattr__(self, "basis_points", normalized)

    def apply(self, raw_price: Decimal, action: ExecutionAction) -> Decimal:
        _validate_application(raw_price, action)
        adverse_amount = raw_price * self.basis_points / _BASIS_POINTS_DENOMINATOR
        adjusted_price = (
            raw_price + adverse_amount
            if action is ExecutionAction.BUY
            else raw_price - adverse_amount
        )
        if adjusted_price <= 0:
            raise ValueError("slippage-adjusted execution price must be positive")
        return adjusted_price


def _validate_application(raw_price: Decimal, action: ExecutionAction) -> None:
    if not isinstance(raw_price, Decimal):
        raise TypeError("raw_price must be a Decimal")
    if not raw_price.is_finite() or raw_price <= 0:
        raise ValueError("raw_price must be finite and positive")
    if not isinstance(action, ExecutionAction):
        raise TypeError("action must be an ExecutionAction")
