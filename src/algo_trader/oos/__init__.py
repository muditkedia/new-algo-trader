"""Persistent research-lineage out-of-sample governance."""

from algo_trader.oos.fingerprint import fingerprint_backtest_result
from algo_trader.oos.models import (
    STANDARD_OOS_PARTITION_POLICY,
    OOSAuditContext,
    OOSDateRange,
    OOSPartitionPolicy,
    OOSPlan,
    OOSTestRecord,
    OOSTransitionRecord,
    OOSWindow,
    OOSWindowSpec,
    OOSWindowState,
)
from algo_trader.oos.planning import (
    DEFAULT_OOS_PROTOCOL_VERSION,
    SEALED_HOLDOUT_WINDOW_ID,
    build_standard_oos_windows,
    create_oos_plan,
    create_standard_oos_plan,
    shift_calendar_months,
)
from algo_trader.oos.registry import OOSRegistry
from algo_trader.oos.universe import (
    derive_equity_data_horizon,
    select_historically_available_equities,
)

__all__ = [
    "DEFAULT_OOS_PROTOCOL_VERSION",
    "SEALED_HOLDOUT_WINDOW_ID",
    "STANDARD_OOS_PARTITION_POLICY",
    "OOSAuditContext",
    "OOSDateRange",
    "OOSPartitionPolicy",
    "OOSPlan",
    "OOSRegistry",
    "OOSTestRecord",
    "OOSTransitionRecord",
    "OOSWindow",
    "OOSWindowSpec",
    "OOSWindowState",
    "build_standard_oos_windows",
    "create_oos_plan",
    "create_standard_oos_plan",
    "derive_equity_data_horizon",
    "fingerprint_backtest_result",
    "shift_calendar_months",
    "select_historically_available_equities",
]
