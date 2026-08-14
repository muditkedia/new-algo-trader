"""Immutable analytical models for deterministic backtest reporting."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from algo_trader.backtest import BacktestTradeRecord
from algo_trader.domain import ExitReason

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
FiniteDecimal = Annotated[Decimal, Field(allow_inf_nan=False)]
NonNegativeDecimal = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]


class FrozenReportingModel(BaseModel):
    """Validation policy shared by immutable reporting records."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ReportContext(FrozenReportingModel):
    """Caller-owned deterministic context for one report build."""

    report_id: NonEmptyStr
    generated_at: datetime
    trading_dates: tuple[date, ...]

    @field_validator("trading_dates", mode="before")
    @classmethod
    def reject_datetime_dates(cls, values: object) -> object:
        if not isinstance(values, (tuple, list)):
            raise TypeError("trading_dates must be an ordered sequence of date objects")
        if any(type(value) is not date for value in values):
            raise TypeError("trading_dates must contain date objects, not datetime values")
        return values

    @model_validator(mode="after")
    def validate_context(self) -> ReportContext:
        if self.generated_at.tzinfo is None or self.generated_at.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware")
        if not self.trading_dates:
            raise ValueError("trading_dates must contain at least one date")
        if any(
            isinstance(value, datetime) or not isinstance(value, date)
            for value in self.trading_dates
        ):
            raise TypeError("trading_dates must contain date objects, not datetime values")
        if len(self.trading_dates) != len(set(self.trading_dates)):
            raise ValueError("trading_dates must not contain duplicates")
        if self.trading_dates != tuple(sorted(self.trading_dates)):
            raise ValueError("trading_dates must be chronological")
        return self


class ProfitFactor(FrozenReportingModel):
    """Finite, positive-unbounded, or undefined after-cost profit factor."""

    value: NonNegativeDecimal | None
    is_unbounded: bool = False
    is_undefined: bool = False

    @model_validator(mode="after")
    def validate_representation(self) -> ProfitFactor:
        states = int(self.value is not None) + int(self.is_unbounded) + int(self.is_undefined)
        if states != 1:
            raise ValueError("profit factor must have exactly one representation")
        return self


class PerformanceMetrics(FrozenReportingModel):
    """Actual realized portfolio metrics; shadow economics are excluded."""

    starting_capital: FiniteDecimal
    ending_capital: FiniteDecimal
    net_profit: FiniteDecimal
    total_return: FiniteDecimal
    cagr: FiniteDecimal | None
    actual_trade_count: int = Field(ge=0)
    winning_trade_count: int = Field(ge=0)
    losing_trade_count: int = Field(ge=0)
    breakeven_trade_count: int = Field(ge=0)
    win_rate: NonNegativeDecimal | None
    gross_positive_net_pnl: NonNegativeDecimal
    gross_negative_net_pnl_absolute: NonNegativeDecimal
    net_profit_factor: ProfitFactor
    average_net_pnl_per_trade: FiniteDecimal | None
    average_net_return_per_trade: FiniteDecimal | None
    median_net_return_per_trade: FiniteDecimal | None
    best_trade_net_pnl: FiniteDecimal | None
    worst_trade_net_pnl: FiniteDecimal | None
    best_trade_net_return: FiniteDecimal | None
    worst_trade_net_return: FiniteDecimal | None
    total_costs: NonNegativeDecimal
    average_cost_per_trade: NonNegativeDecimal | None
    costs_as_pct_of_gross_profit: NonNegativeDecimal | None
    average_mfe_return: NonNegativeDecimal | None
    average_mae_return: FiniteDecimal | None
    maximum_realized_drawdown: NonNegativeDecimal
    maximum_realized_drawdown_pct: NonNegativeDecimal
    evaluation_trading_days: int = Field(gt=0)
    actual_trades_per_day: NonNegativeDecimal
    average_quality_score: NonNegativeDecimal | None


class ShadowMetrics(FrozenReportingModel):
    """Hypothetical diagnostics with no actual capital impact."""

    economic_status: str = "HYPOTHETICAL - NO ACTUAL CAPITAL IMPACT"
    shadow_trade_count: int = Field(ge=0)
    winning_trade_count: int = Field(ge=0)
    losing_trade_count: int = Field(ge=0)
    breakeven_trade_count: int = Field(ge=0)
    hypothetical_net_pnl: FiniteDecimal
    average_net_pnl: FiniteDecimal | None
    average_net_return: FiniteDecimal | None
    win_rate: NonNegativeDecimal | None
    profit_factor: ProfitFactor
    hypothetical_total_costs: NonNegativeDecimal
    average_quality_score: NonNegativeDecimal | None
    best_net_return: FiniteDecimal | None
    worst_net_return: FiniteDecimal | None


class RequestOutcomeSummary(FrozenReportingModel):
    """Exhaustive terminal request funnel."""

    total_requests: int = Field(ge=0)
    completed_actual: int = Field(ge=0)
    completed_shadow: int = Field(ge=0)
    allocated_entry_not_filled: int = Field(ge=0)
    shadow_entry_not_filled: int = Field(ge=0)
    capital_exhausted: int = Field(ge=0)
    capacity_rejected_request_count: int = Field(ge=0)
    actual_completion_rate: NonNegativeDecimal | None
    capacity_rejection_rate: NonNegativeDecimal | None
    no_fill_rate: NonNegativeDecimal | None


class StrategyTradeMetrics(FrozenReportingModel):
    """Descriptive completed-trade metrics for one strategy version."""

    strategy_id: NonEmptyStr
    strategy_version: NonEmptyStr
    economic_status: NonEmptyStr
    trade_count: int = Field(ge=0)
    wins: int = Field(ge=0)
    losses: int = Field(ge=0)
    breakeven: int = Field(ge=0)
    long_trade_count: int = Field(ge=0)
    short_trade_count: int = Field(ge=0)
    net_pnl: FiniteDecimal
    average_net_pnl: FiniteDecimal | None
    average_net_return: FiniteDecimal | None
    win_rate: NonNegativeDecimal | None
    profit_factor: ProfitFactor
    total_costs: NonNegativeDecimal
    average_quality_score: NonNegativeDecimal | None


class SymbolTradeMetrics(FrozenReportingModel):
    """Descriptive actual completed-trade metrics for one symbol."""

    symbol: NonEmptyStr
    trade_count: int = Field(ge=0)
    wins: int = Field(ge=0)
    losses: int = Field(ge=0)
    breakeven: int = Field(ge=0)
    long_count: int = Field(ge=0)
    short_count: int = Field(ge=0)
    net_pnl: FiniteDecimal
    average_net_return: FiniteDecimal | None
    win_rate: NonNegativeDecimal | None
    profit_factor: ProfitFactor
    total_costs: NonNegativeDecimal


class ExitReasonMetrics(FrozenReportingModel):
    """Stable completed-trade aggregation for one exit reason."""

    exit_reason: ExitReason
    economic_status: NonEmptyStr
    trade_count: int = Field(ge=0)
    net_pnl: FiniteDecimal
    average_net_pnl: FiniteDecimal | None
    average_net_return: FiniteDecimal | None


class CostSummary(FrozenReportingModel):
    """Aggregated retained transaction-cost breakdowns for one economic class."""

    economic_status: NonEmptyStr
    entry_turnover: NonNegativeDecimal
    exit_turnover: NonNegativeDecimal
    total_turnover: NonNegativeDecimal
    brokerage: NonNegativeDecimal
    exchange_transaction_charge: NonNegativeDecimal
    sebi_turnover_fee: NonNegativeDecimal
    ipft: NonNegativeDecimal
    stt: NonNegativeDecimal
    stamp_duty: NonNegativeDecimal
    gst: NonNegativeDecimal
    total_costs: NonNegativeDecimal
    component_percentages: dict[str, NonNegativeDecimal | None]


class EquityPoint(FrozenReportingModel):
    """One actual realized-capital point, never an intraday MTM value."""

    timestamp: datetime
    realized_capital: FiniteDecimal
    running_peak_capital: FiniteDecimal
    drawdown_amount: NonNegativeDecimal
    drawdown_pct: NonNegativeDecimal
    group_net_pnl: FiniteDecimal
    cumulative_net_pnl: FiniteDecimal


class DailyPerformance(FrozenReportingModel):
    """Actual daily results over a caller-supplied evaluation trading date."""

    trading_date: date
    actual_trade_count: int = Field(ge=0)
    winning_trades: int = Field(ge=0)
    losing_trades: int = Field(ge=0)
    breakeven_trades: int = Field(ge=0)
    gross_pnl: FiniteDecimal
    total_costs: NonNegativeDecimal
    net_pnl: FiniteDecimal
    cumulative_net_pnl: FiniteDecimal
    realized_end_capital: FiniteDecimal
    average_trade_net_return: FiniteDecimal | None


class AcceptanceAssessment(FrozenReportingModel):
    """Strict numerical research-target assessment, not strategy approval."""

    cagr_pass: bool
    win_rate_pass: bool
    profit_factor_pass: bool
    average_net_return_pass: bool
    frequency_target_met: bool
    hard_quantitative_targets_pass: bool


class ReportProvenance(FrozenReportingModel):
    """Explicit reproducibility identifiers retained by every report."""

    report_id: NonEmptyStr
    reporting_version: NonEmptyStr
    generated_at: datetime
    source_backtest_fingerprint: NonEmptyStr
    run_id: NonEmptyStr
    git_commit: NonEmptyStr
    backtester_version: NonEmptyStr
    window_start: datetime
    window_end: datetime
    cost_policy_id: NonEmptyStr
    cost_policy_source_as_of_date: date
    brokerage_plan: NonEmptyStr
    starting_capital: FiniteDecimal
    ending_capital: FiniteDecimal
    capital_exhausted: bool
    symbols: tuple[str, ...]
    strategy_versions: tuple[tuple[str, str], ...]
    ml_model_versions: tuple[str, ...]
    evaluation_trading_dates: tuple[date, ...]
    research_scope_id: str | None = None
    plan_id: str | None = None
    window_id: str | None = None
    oos_result_fingerprint: str | None = None


class ReportBundle(FrozenReportingModel):
    """Canonical validated in-memory reporting truth derived from a backtest."""

    provenance: ReportProvenance
    performance: PerformanceMetrics
    shadow_metrics: ShadowMetrics
    request_outcomes: RequestOutcomeSummary
    actual_costs: CostSummary
    shadow_costs: CostSummary
    equity_curve: tuple[EquityPoint, ...]
    daily_performance: tuple[DailyPerformance, ...]
    actual_strategy_breakdown: tuple[StrategyTradeMetrics, ...]
    shadow_strategy_breakdown: tuple[StrategyTradeMetrics, ...]
    symbol_breakdown: tuple[SymbolTradeMetrics, ...]
    actual_exit_reason_breakdown: tuple[ExitReasonMetrics, ...]
    shadow_exit_reason_breakdown: tuple[ExitReasonMetrics, ...]
    acceptance: AcceptanceAssessment
    actual_trade_records: tuple[BacktestTradeRecord, ...]
    shadow_trade_records: tuple[BacktestTradeRecord, ...]


class ReportComparisonRow(FrozenReportingModel):
    """One source-backed run/window row in a deterministic comparison."""

    report_id: NonEmptyStr
    run_id: NonEmptyStr
    backtester_version: NonEmptyStr
    reporting_version: NonEmptyStr
    research_scope_id: str | None = None
    plan_id: str | None = None
    window_id: str | None = None
    window_start: datetime
    window_end: datetime
    actual_trade_count: int = Field(ge=0)
    actual_trades_per_day: NonNegativeDecimal
    net_pnl: FiniteDecimal
    ending_capital: FiniteDecimal
    cagr: FiniteDecimal | None
    win_rate: NonNegativeDecimal | None
    profit_factor: ProfitFactor
    average_net_return: FiniteDecimal | None
    max_drawdown: NonNegativeDecimal
    total_actual_costs: NonNegativeDecimal
    long_trade_count: int = Field(ge=0)
    short_trade_count: int = Field(ge=0)
    cagr_pass: bool
    win_rate_pass: bool
    profit_factor_pass: bool
    average_net_return_pass: bool
    frequency_target_met: bool
    hard_quantitative_targets_pass: bool


class ReportComparisonBundle(FrozenReportingModel):
    """Chronological report comparison without pooled pseudo-backtest metrics."""

    is_oos: bool
    research_scope_id: str | None = None
    plan_id: str | None = None
    rows: tuple[ReportComparisonRow, ...]

    @model_validator(mode="after")
    def validate_nonempty(self) -> ReportComparisonBundle:
        if not self.rows:
            raise ValueError("report comparison must contain at least one row")
        return self
