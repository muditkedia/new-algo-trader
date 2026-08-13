"""Deterministic construction of chronological OOS plans."""

from __future__ import annotations

import calendar
from collections.abc import Iterable
from datetime import date, datetime

from algo_trader.oos.models import (
    OOSAuditContext,
    OOSPlan,
    OOSWindow,
    OOSWindowSpec,
    OOSWindowState,
)

SEALED_HOLDOUT_WINDOW_ID = "sealed-holdout"
DEFAULT_OOS_PROTOCOL_VERSION = "1"


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
    data_start_date: date,
    data_end_exclusive: date,
    oos_windows: Iterable[OOSWindowSpec],
    audit_context: OOSAuditContext,
    protocol_version: str = DEFAULT_OOS_PROTOCOL_VERSION,
) -> OOSPlan:
    """Create one immutable plan with caller-sized ordinary OOS windows."""
    if isinstance(data_start_date, datetime) or not isinstance(data_start_date, date):
        raise TypeError("data_start_date must be a date")
    if isinstance(data_end_exclusive, datetime) or not isinstance(
        data_end_exclusive, date
    ):
        raise TypeError("data_end_exclusive must be a date")
    selected = tuple(oos_windows)
    if any(not isinstance(window, OOSWindowSpec) for window in selected):
        raise TypeError("all oos_windows must be OOSWindowSpec instances")
    ordered = tuple(sorted(selected, key=lambda window: window.start_date))
    holdout_start = shift_calendar_months(data_end_exclusive, -12)
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
