"""Leakage-resistant signal-time feature extraction."""

from __future__ import annotations

import math
from decimal import Decimal

from algo_trader.domain import Signal
from algo_trader.ml.models import MetaFeatureSchema


class FeatureIntegrityError(ValueError):
    """Raised when a selected signal-time model feature is unusable."""


def extract_meta_features(signal: Signal, schema: MetaFeatureSchema) -> tuple[float, ...]:
    """Extract only explicitly selected top-level finite numeric features."""
    if not isinstance(signal, Signal):
        raise TypeError("signal must be a Signal")
    if not isinstance(schema, MetaFeatureSchema):
        raise TypeError("schema must be a MetaFeatureSchema")
    values: list[float] = []
    for name in schema.feature_names:
        if name not in signal.feature_snapshot:
            raise FeatureIntegrityError(f"required model feature is missing: {name}")
        value = signal.feature_snapshot[name]
        if isinstance(value, bool) or not isinstance(value, int | float | Decimal):
            raise FeatureIntegrityError(f"model feature must be a numeric scalar: {name}")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise FeatureIntegrityError(f"model feature must be finite: {name}")
        values.append(numeric)
    return tuple(values)
