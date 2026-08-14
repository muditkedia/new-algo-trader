"""Deterministic construction of chronological OOS plans."""

from __future__ import annotations

import calendar
from collections.abc import Iterable
from datetime import date, datetime

from algo_trader.oos.models import (
    STANDARD_OOS_PARTITION_POLICY,
    OOSAuditContext,
    OOSPartitionPolicy,
    OOSPlan,
    OOSWindow,
    OOSWindowSpec,
    OOSWindowState,
)

SEALED_HOLDOUT_WINDOW_ID = "sealed-holdout"
DEFAULT_OOS_PROTOCOL_VERSION = "3"


def shift_calendar_months(value: date, months: int) -> date:
    """Shift a date by calendar months, clamping to the target month end."""
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError("value must be a date")
    if isinstance(months, bool) or not isinstance(months, int):
        raise TypeError("months must be an integer")
    month_index = value.year * 12 + value.month - 1 + months
    target_year, zero_based_month = divmod(month_index, 12)
    target_month = zero_based_month + 1
    target_day = min(value.day, calendar.monthrange(target_year, target_month)[1])
    return date(target_year, target_month, target_day)


def create_oos_plan(
    *,
    research_scope_id: str,
    plan_id: str,
    strategy_ids: Iterable[str],
    data_start_date: date,
    data_end_exclusive: date,
    oos_windows: Iterable[OOSWindowSpec],
    audit_context: OOSAuditContext,
    protocol_version: str = DEFAULT_OOS_PROTOCOL_VERSION,
    partition_policy: OOSPartitionPolicy | None = None,
) -> OOSPlan:
    """Create one immutable plan with caller-sized ordinary OOS windows."""
    if strategy_ids is None or isinstance(strategy_ids, str):
        raise TypeError("strategy_ids must be a non-string iterable of strategy IDs")
    try:
        selected_strategy_ids = tuple(strategy_ids)
    except TypeError as error:
        raise TypeError(
            "strategy_ids must be a non-string iterable of strategy IDs"
        ) from error
    if isinstance(data_start_date, datetime) or not isinstance(data_start_date, date):
        raise TypeError("data_start_date must be a date")
    if isinstance(data_end_exclusive, datetime) or not isinstance(
        data_end_exclusive, date
    ):
        raise TypeError("data_end_exclusive must be a date")
    if partition_policy is not None and not isinstance(
        partition_policy,
        OOSPartitionPolicy,
    ):
        raise TypeError("partition_policy must be an OOSPartitionPolicy or None")
    selected = tuple(oos_windows)
    if any(not isinstance(window, OOSWindowSpec) for window in selected):
        raise TypeError("all oos_windows must be OOSWindowSpec instances")
    ordered = tuple(sorted(selected, key=lambda window: window.start_date))
    holdout_months = (
        partition_policy.sealed_holdout_months
        if partition_policy is not None
        else STANDARD_OOS_PARTITION_POLICY.sealed_holdout_months
    )
    holdout_start = shift_calendar_months(data_end_exclusive, -holdout_months)
    ordinary = tuple(
        OOSWindow(
            window_id=window.window_id,
            start_date=window.start_date,
            end_date=window.end_date,
            state=OOSWindowState.AVAILABLE,
        )
        for window in ordered
    )
    development_end = ordinary[0].start_date if ordinary else data_start_date
    sealed_holdout = OOSWindow(
        window_id=SEALED_HOLDOUT_WINDOW_ID,
        start_date=holdout_start,
        end_date=data_end_exclusive,
        state=OOSWindowState.SEALED_HOLDOUT,
    )
    return OOSPlan(
        research_scope_id=research_scope_id,
        plan_id=plan_id,
        protocol_version=protocol_version,
        partition_policy=partition_policy,
        strategy_ids=selected_strategy_ids,
        data_start_date=data_start_date,
        data_end_exclusive=data_end_exclusive,
        development_start_date=data_start_date,
        development_end_exclusive=development_end,
        sealed_holdout_start_date=holdout_start,
        sealed_holdout_end_exclusive=data_end_exclusive,
        oos_windows=ordinary,
        sealed_holdout=sealed_holdout,
        creation_audit=audit_context,
    )


def build_standard_oos_windows(
    *,
    earliest_oos_start_date: date,
    sealed_holdout_start_date: date,
    policy: OOSPartitionPolicy = STANDARD_OOS_PARTITION_POLICY,
) -> tuple[OOSWindowSpec, ...]:
    """Build contiguous forward OOS windows under the frozen standard policy."""
    _require_date(earliest_oos_start_date, "earliest_oos_start_date")
    _require_date(sealed_holdout_start_date, "sealed_holdout_start_date")
    if not isinstance(policy, OOSPartitionPolicy):
        raise TypeError("policy must be an OOSPartitionPolicy")
    if earliest_oos_start_date >= sealed_holdout_start_date:
        raise ValueError(
            "earliest_oos_start_date must be earlier than sealed_holdout_start_date"
        )
    if not _meets_minimum_duration(
        earliest_oos_start_date,
        sealed_holdout_start_date,
        policy,
    ):
        raise ValueError("ordinary OOS horizon must span at least one minimum window")

    ranges: list[tuple[date, date]] = []
    cursor = earliest_oos_start_date
    while True:
        target_end = shift_calendar_months(cursor, policy.target_window_months)
        if target_end > sealed_holdout_start_date:
            break
        ranges.append((cursor, target_end))
        cursor = target_end
        if cursor == sealed_holdout_start_date:
            break

    if cursor != sealed_holdout_start_date:
        if _valid_window_duration(cursor, sealed_holdout_start_date, policy):
            ranges.append((cursor, sealed_holdout_start_date))
        else:
            if not ranges:
                raise ValueError("ordinary OOS horizon cannot satisfy window bounds")
            rebalance_start, _ = ranges.pop()
            split = _find_rebalanced_split(
                rebalance_start,
                sealed_holdout_start_date,
                policy,
            )
            ranges.extend(
                (
                    (rebalance_start, split),
                    (split, sealed_holdout_start_date),
                )
            )

    windows = tuple(
        OOSWindowSpec(
            window_id=f"oos-{ordinal:03d}",
            start_date=start_date,
            end_date=end_date,
        )
        for ordinal, (start_date, end_date) in enumerate(ranges, start=1)
    )
    _validate_standard_windows(
        windows,
        earliest_oos_start_date,
        sealed_holdout_start_date,
        policy,
    )
    return windows


def create_standard_oos_plan(
    *,
    research_scope_id: str,
    plan_id: str,
    strategy_ids: Iterable[str],
    data_start_date: date,
    data_end_exclusive: date,
    earliest_oos_start_date: date,
    audit_context: OOSAuditContext,
    policy: OOSPartitionPolicy = STANDARD_OOS_PARTITION_POLICY,
    protocol_version: str = DEFAULT_OOS_PROTOCOL_VERSION,
) -> OOSPlan:
    """Create a policy-bound standard plan while retaining the low-level API."""
    _require_date(data_start_date, "data_start_date")
    _require_date(data_end_exclusive, "data_end_exclusive")
    _require_date(earliest_oos_start_date, "earliest_oos_start_date")
    if not isinstance(policy, OOSPartitionPolicy):
        raise TypeError("policy must be an OOSPartitionPolicy")
    if not data_start_date < earliest_oos_start_date:
        raise ValueError("earliest_oos_start_date must be after data_start_date")
    holdout_start = shift_calendar_months(
        data_end_exclusive,
        -policy.sealed_holdout_months,
    )
    if earliest_oos_start_date >= holdout_start:
        raise ValueError(
            "earliest_oos_start_date must be earlier than sealed holdout start"
        )
    return create_oos_plan(
        research_scope_id=research_scope_id,
        plan_id=plan_id,
        strategy_ids=strategy_ids,
        data_start_date=data_start_date,
        data_end_exclusive=data_end_exclusive,
        oos_windows=build_standard_oos_windows(
            earliest_oos_start_date=earliest_oos_start_date,
            sealed_holdout_start_date=holdout_start,
            policy=policy,
        ),
        audit_context=audit_context,
        protocol_version=protocol_version,
        partition_policy=policy,
    )


def _find_rebalanced_split(
    start_date: date,
    end_date: date,
    policy: OOSPartitionPolicy,
) -> date:
    preferred_months = max(
        policy.minimum_window_months,
        policy.target_window_months - 1,
    )
    candidate_months = (
        preferred_months,
        *(
            months
            for months in range(
                policy.minimum_window_months,
                policy.maximum_window_months + 1,
            )
            if months != preferred_months
        ),
    )
    for months in candidate_months:
        split = shift_calendar_months(start_date, months)
        if _valid_window_duration(start_date, split, policy) and _valid_window_duration(
            split,
            end_date,
            policy,
        ):
            return split
    raise ValueError("ordinary OOS tail cannot be rebalanced within window bounds")


def _meets_minimum_duration(
    start_date: date,
    end_date: date,
    policy: OOSPartitionPolicy,
) -> bool:
    return end_date >= shift_calendar_months(
        start_date,
        policy.minimum_window_months,
    )


def _valid_window_duration(
    start_date: date,
    end_date: date,
    policy: OOSPartitionPolicy,
) -> bool:
    return (
        _meets_minimum_duration(start_date, end_date, policy)
        and end_date
        <= shift_calendar_months(start_date, policy.maximum_window_months)
    )


def _validate_standard_windows(
    windows: tuple[OOSWindowSpec, ...],
    expected_start: date,
    expected_end: date,
    policy: OOSPartitionPolicy,
) -> None:
    if not windows:
        raise ValueError("at least one standard OOS window is required")
    if windows[0].start_date != expected_start or windows[-1].end_date != expected_end:
        raise ValueError("standard OOS windows must cover the exact requested horizon")
    if len({window.window_id for window in windows}) != len(windows):
        raise ValueError("standard OOS window IDs must be unique")
    previous_end = expected_start
    for window in windows:
        if window.start_date != previous_end:
            raise ValueError("standard OOS windows must be contiguous")
        if not _valid_window_duration(window.start_date, window.end_date, policy):
            raise ValueError("standard OOS window is outside policy duration bounds")
        previous_end = window.end_date


def _require_date(value: object, name: str) -> None:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(f"{name} must be a date")
