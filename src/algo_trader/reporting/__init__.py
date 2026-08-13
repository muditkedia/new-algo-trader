"""Deterministic analytics and derivative reporting artifacts."""

from algo_trader.reporting.analytics import (
    REPORTING_VERSION,
    ReportingIntegrityError,
    build_report,
)
from algo_trader.reporting.models import (
    AcceptanceAssessment,
    CostSummary,
    DailyPerformance,
    EquityPoint,
    ExitReasonMetrics,
    PerformanceMetrics,
    ProfitFactor,
    ReportBundle,
    ReportContext,
    ReportProvenance,
    RequestOutcomeSummary,
    ShadowMetrics,
    StrategyTradeMetrics,
    SymbolTradeMetrics,
)
from algo_trader.reporting.outputs import write_excel_report, write_visual_report
from algo_trader.reporting.tables import (
    REPORT_TABLE_FILENAMES,
    report_tables,
    write_report_dataset,
)

__all__ = [
    "REPORTING_VERSION",
    "REPORT_TABLE_FILENAMES",
    "AcceptanceAssessment",
    "CostSummary",
    "DailyPerformance",
    "EquityPoint",
    "ExitReasonMetrics",
    "PerformanceMetrics",
    "ProfitFactor",
    "ReportBundle",
    "ReportContext",
    "ReportProvenance",
    "ReportingIntegrityError",
    "RequestOutcomeSummary",
    "ShadowMetrics",
    "StrategyTradeMetrics",
    "SymbolTradeMetrics",
    "build_report",
    "report_tables",
    "write_excel_report",
    "write_report_dataset",
    "write_visual_report",
]
