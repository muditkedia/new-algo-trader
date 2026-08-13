"""Immutable out-of-sample governance models."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
StrictDate = Annotated[date, Field(strict=True)]


class FrozenOOSModel(BaseModel):
    """Validation policy for immutable OOS governance records."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class OOSWindowState(StrEnum):
    """Complete monotonic lifecycle for ordinary OOS and sealed windows."""

    AVAILABLE = "AVAILABLE"
    TESTED = "TESTED"
    CONSUMED = "CONSUMED"
    TRAINING_ALLOWED = "TRAINING_ALLOWED"
    SEALED_HOLDOUT = "SEALED_HOLDOUT"


class OOSAuditContext(FrozenOOSModel):
    """Caller-supplied provenance for one persistent state change."""

    event_id: NonEmptyStr
    occurred_at: datetime
    git_commit: NonEmptyStr

    @model_validator(mode="after")
    def validate_timestamp(self) -> OOSAuditContext:
        if self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")
        return self


class OOSWindowSpec(FrozenOOSModel):
    """Caller-selected half-open ordinary OOS date partition."""

    window_id: NonEmptyStr
    start_date: StrictDate
    end_date: StrictDate

    @model_validator(mode="after")
    def validate_range(self) -> OOSWindowSpec:
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be earlier than end_date")
        return self


class OOSWindow(OOSWindowSpec):
    """One persisted OOS date partition and its current state."""

    state: OOSWindowState


class OOSDateRange(FrozenOOSModel):
    """One half-open date range approved for a governance use."""

    start_date: StrictDate
    end_date: StrictDate

    @model_validator(mode="after")
    def validate_range(self) -> OOSDateRange:
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be earlier than end_date")
        return self


class OOSPlan(FrozenOOSModel):
    """One research-lineage plan over a fixed historical data horizon."""

    research_scope_id: NonEmptyStr
    plan_id: NonEmptyStr
    protocol_version: NonEmptyStr
    data_start_date: StrictDate
    data_end_exclusive: StrictDate
    development_start_date: StrictDate
    development_end_exclusive: StrictDate
    sealed_holdout_start_date: StrictDate
    sealed_holdout_end_exclusive: StrictDate
    oos_windows: tuple[OOSWindow, ...]
    sealed_holdout: OOSWindow
    creation_audit: OOSAuditContext

    @model_validator(mode="after")
    def validate_partition(self) -> OOSPlan:
        from algo_trader.oos.planning import shift_calendar_months

        if self.data_start_date >= self.data_end_exclusive:
            raise ValueError("data_start_date must be earlier than data_end_exclusive")
        expected_holdout_start = shift_calendar_months(
            self.data_end_exclusive,
            -12,
        )
        if self.sealed_holdout_start_date != expected_holdout_start:
            raise ValueError("sealed holdout must start exactly 12 calendar months before data end")
        if self.sealed_holdout_end_exclusive != self.data_end_exclusive:
            raise ValueError("sealed holdout must end at data_end_exclusive")
        if self.data_start_date >= self.sealed_holdout_start_date:
            raise ValueError("data horizon must leave a non-empty pre-holdout range")
        if not self.oos_windows:
            raise ValueError("at least one ordinary OOS window is required")

        identifiers = [window.window_id for window in self.oos_windows]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("ordinary OOS window IDs must be unique inside a plan")
        if self.sealed_holdout.window_id in identifiers:
            raise ValueError("sealed holdout window ID must be unique inside a plan")

        ordered = tuple(sorted(self.oos_windows, key=lambda window: window.start_date))
        if ordered != self.oos_windows:
            raise ValueError("ordinary OOS windows must be in chronological order")
        if any(window.state is OOSWindowState.SEALED_HOLDOUT for window in ordered):
            raise ValueError("ordinary OOS windows cannot use SEALED_HOLDOUT state")
        if ordered[0].start_date <= self.data_start_date:
            raise ValueError("first ordinary OOS must leave non-empty development data")
        if self.development_start_date != self.data_start_date:
            raise ValueError("development range must start at data_start_date")
        if self.development_end_exclusive != ordered[0].start_date:
            raise ValueError("development range must end at the first ordinary OOS")

        previous_end = ordered[0].start_date
        for window in ordered:
            if window.start_date != previous_end:
                raise ValueError("ordinary OOS windows must be contiguous without overlap or gaps")
            if not (
                self.data_start_date < window.start_date < window.end_date
                <= self.sealed_holdout_start_date
            ):
                raise ValueError("ordinary OOS window is outside the pre-holdout data range")
            previous_end = window.end_date
        if previous_end != self.sealed_holdout_start_date:
            raise ValueError("final ordinary OOS must end exactly at sealed holdout start")

        if (
            self.sealed_holdout.start_date != self.sealed_holdout_start_date
            or self.sealed_holdout.end_date != self.sealed_holdout_end_exclusive
            or self.sealed_holdout.state is not OOSWindowState.SEALED_HOLDOUT
        ):
            raise ValueError("sealed_holdout must match the plan's protected final partition")
        return self


class OOSTestRecord(FrozenOOSModel):
    """Compact immutable provenance for one registered OOS backtest."""

    research_scope_id: NonEmptyStr
    plan_id: NonEmptyStr
    window_id: NonEmptyStr
    backtest_run_id: NonEmptyStr
    backtest_git_commit: NonEmptyStr
    backtester_version: NonEmptyStr
    backtest_window_start: datetime
    backtest_window_end: datetime
    cost_policy_id: NonEmptyStr
    brokerage_plan: NonEmptyStr
    symbols: tuple[str, ...]
    strategy_versions: tuple[tuple[str, str], ...]
    ml_model_versions: tuple[str, ...]
    result_fingerprint: NonEmptyStr
    registration_audit: OOSAuditContext


class OOSTransitionRecord(FrozenOOSModel):
    """Persisted plan-creation or monotonic window-transition provenance."""

    event_id: NonEmptyStr
    occurred_at: datetime
    git_commit: NonEmptyStr
    research_scope_id: NonEmptyStr
    plan_id: NonEmptyStr
    window_id: str | None
    from_state: OOSWindowState | None
    to_state: OOSWindowState | None
    event_type: NonEmptyStr
