"""Canonical deterministic Runtime identities and configuration fingerprints."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

from algo_trader.portfolio import AllocationCandidate, CandidateIdentity
from algo_trader.runtime.models import RuntimeConfig, RuntimeOrderLeg


def candidate_identity_payload(identity: CandidateIdentity) -> list[object]:
    """Return the frozen portfolio identity in canonical JSON-safe form."""
    return [_canonical_value(value) for value in identity]


def candidate_fingerprint(candidate: AllocationCandidate) -> str:
    """Hash only the frozen duplicate/reservation identity, never ML snapshots."""
    if not isinstance(candidate, AllocationCandidate):
        raise TypeError("candidate must be an AllocationCandidate")
    payload = _canonical_json(candidate_identity_payload(candidate.identity))
    return hashlib.sha256(payload).hexdigest()


def runtime_client_order_id(
    runtime_session_id: str,
    candidate_identity: CandidateIdentity,
    leg: RuntimeOrderLeg,
    attempt: int = 1,
) -> str:
    """Create a stable ID from session, candidate, leg, and deterministic attempt."""
    if not isinstance(runtime_session_id, str) or not runtime_session_id.strip():
        raise ValueError("runtime_session_id must be a non-empty string")
    if not isinstance(leg, RuntimeOrderLeg):
        raise TypeError("leg must be a RuntimeOrderLeg")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt <= 0:
        raise ValueError("attempt must be a positive integer")
    if leg is RuntimeOrderLeg.ENTRY and attempt != 1:
        raise ValueError("ENTRY order attempt must be 1")
    payload = {
        "candidate_identity": candidate_identity_payload(candidate_identity),
        "leg": leg.value,
        "attempt": attempt,
        "runtime_session_id": runtime_session_id.strip(),
        "runtime_version": "1",
    }
    return f"NAT-RUNTIME-{hashlib.sha256(_canonical_json(payload)).hexdigest()}"


def runtime_config_fingerprint(config: RuntimeConfig) -> str:
    """Hash non-secret configuration provenance deterministically."""
    if not isinstance(config, RuntimeConfig):
        raise TypeError("config must be a RuntimeConfig")
    payload = config.model_dump(mode="json")
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _canonical_value(value: Any) -> object:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
