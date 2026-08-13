"""Deterministic complete BacktestRunResult fingerprinting."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from decimal import Decimal
from enum import Enum

from algo_trader.backtest import BacktestRunResult


def fingerprint_backtest_result(result: BacktestRunResult) -> str:
    """Return SHA-256 over canonical JSON for the complete immutable result."""
    if not isinstance(result, BacktestRunResult):
        raise TypeError("result must be a BacktestRunResult")
    payload = _canonicalize(result.model_dump(mode="python"))
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _canonicalize(value: object) -> object:
    if isinstance(value, Decimal):
        if value == 0:
            return "0"
        normalized = value.normalize()
        return format(normalized, "f")
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _canonicalize(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_canonicalize(item) for item in value]
    return value
