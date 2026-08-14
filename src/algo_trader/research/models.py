"""Immutable generic research-orchestration records."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator


class FrozenResearchModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ResearchDecisionRecord(FrozenResearchModel):
    """Human decision linked to canonical run evidence without copied metrics."""

    decision_id: str
    source_run_ids: tuple[str, ...]
    strategy_id: str
    strategy_version: str
    research_scope_id: str
    decision: str
    diagnosis: str
    changes_authorized: tuple[str, ...]
    changes_rejected: tuple[str, ...]
    next_action: str
    recorded_at: datetime
    git_commit: str

    @field_validator(
        "decision_id",
        "strategy_id",
        "strategy_version",
        "research_scope_id",
        "decision",
        "diagnosis",
        "next_action",
        "git_commit",
    )
    @classmethod
    def require_nonblank(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("research decision text fields must be nonblank")
        return value.strip()

    @field_validator(
        "source_run_ids",
        "changes_authorized",
        "changes_rejected",
        mode="before",
    )
    @classmethod
    def normalize_strings(cls, value: object) -> tuple[str, ...]:
        if isinstance(value, str) or value is None:
            raise TypeError("research decision collections must be iterables of strings")
        selected = tuple(value)  # type: ignore[arg-type]
        normalized = tuple(item.strip() for item in selected if isinstance(item, str))
        if len(normalized) != len(selected) or any(not item for item in normalized):
            raise ValueError("research decision collection values must be nonblank strings")
        if len(normalized) != len(set(normalized)):
            raise ValueError("research decision collections must not contain duplicates")
        return normalized
