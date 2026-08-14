"""Deterministic read-only analytics over immutable backtest results."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime, time, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from algo_trader.backtest import (
    BacktestRequestOutcome,
    BacktestRunResult,
    BacktestTradeRecord,
)
from algo_trader.domain import ExitReason, Side
from algo_trader.oos import OOSTestRecord, fingerprint_backtest_result
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

REPORTING_VERSION = "3"
MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")
ZERO = Decimal("0")


class ReportingIntegrityError(ValueError):
    """Raised when retained backtest facts cannot form a truthful report."""


def _sum(values: Iterable[Decimal]) -> Decimal:
    return sum(values, start=ZERO)


def _mean(values: tuple[Decimal, ...]) -> Decimal | None:
    return _sum(values) / len(values) if values else None


def _median(values: tuple[Decimal, ...]) -> Decimal | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / Decimal("2")


def _ratio(numerator: int | Decimal, denominator: int | Decimal) -> Decimal:
    return Decimal(numerator) / Decimal(denominator)


def _quality(record: BacktestTradeRecord) -> Decimal:
    return Decimal(str(record.trade.ml_score.quality_score))


def _identity_sort_key(identity) -> tuple[object, ...]:
    """Mirror the backtester allocation-identity ordering exactly."""
    return (
        identity[0],
        identity[1],
        identity[2],
        identity[3].value,
        identity[4],
        identity[5],
        identity[6].value,
        int(identity[7] is not None),
        identity[7] if identity[7] is not None else ZERO,
        identity[8],
        identity[9],
    )


def _record_sort_key(record: BacktestTradeRecord) -> tuple[object, ...]:
    """Use the same realized-trade order as HistoricalBacktester."""
    return (
        record.trade.exit_fill.timestamp,
        _identity_sort_key(record.allocation_identity),
    )


def _realization_groups(
    records: tuple[BacktestTradeRecord, ...],
) -> tuple[tuple[datetime, Decimal], ...]:
    """Reproduce HistoricalBacktester's timestamp-group capital realization arithmetic.

    Decimal arithmetic is context-sensitive and therefore not associative. The
    backtester intentionally realizes all exits sharing one timestamp as a group,
    ordered by allocation identity, then applies that group P&L to capital. Reporting
    must use that identical reduction path instead of independently flattening or
    reordering trade P&L, otherwise exact reconciliation can fail by rounding dust.
    """
    by_timestamp: dict[datetime, list[BacktestTradeRecord]] = defaultdict(list)
    for record in records:
        by_timestamp[record.trade.exit_fill.timestamp].append(record)

    groups: list[tuple[datetime, Decimal]] = []
    for timestamp in sorted(by_timestamp):
        ordered = sorted(
            by_timestamp[timestamp],
            key=lambda record: _identity_sort_key(record.allocation_identity),
        )
        groups.append(
            (
                timestamp,
                _sum(record.trade.net_pnl for record in ordered),
            )
        )
    return tuple(groups)


def _realized_ending_capital(
    records: tuple[BacktestTradeRecord, ...],
    starting_capital: Decimal,
) -> Decimal:
    capital = starting_capital
    for _, group_pnl in _realization_groups(records):
        capital += group_pnl
    return capital


def _profit_factor(records: tuple[BacktestTradeRecord, ...]) -> ProfitFactor:
    positive = _sum(record.trade.net_pnl for record in records if record.trade.net_pnl > 0)
    negative = -_sum(record.trade.net_pnl for record in records if record.trade.net_pnl < 0)
    if negative > 0:
        return ProfitFactor(value=positive / negative)
    if positive > 0:
        return ProfitFactor(value=None, is_unbounded=True)
    return ProfitFactor(value=None, is_undefined=True)


def _validate_result(
    result: BacktestRunResult,
    context: ReportContext,
) -> tuple[tuple[BacktestTradeRecord, ...], tuple[BacktestTradeRecord, ...]]:
    if not isinstance(result, BacktestRunResult):
        raise TypeError("result must be a BacktestRunResult")
    if not isinstance(context, ReportContext):
        raise TypeError("context must be a ReportContext")
    for name, value in (
        ("window_start", result.window_start),
        ("window_end", result.window_end),
    ):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ReportingIntegrityError(f"{name} must be timezone-aware")
    if result.window_start >= result.window_end:
        raise ReportingIntegrityError("backtest window must have positive elapsed duration")
    if result.starting_capital <= 0:
        raise ReportingIntegrityError("starting_capital must be positive")
    window_start = result.window_start.astimezone(MARKET_TIMEZONE)
    window_end = result.window_end.astimezone(MARKET_TIMEZONE)
    for day in context.trading_dates:
        day_start = datetime.combine(day, time.min, tzinfo=MARKET_TIMEZONE)
        day_end = day_start + timedelta(days=1)
        if not (day_start < window_end and day_end > window_start):
            raise ReportingIntegrityError(
                "every trading_date must intersect the half-open backtest window"
            )

    actual = tuple(sorted(result.actual_trade_records, key=_record_sort_key))
    shadow = tuple(sorted(result.shadow_trade_records, key=_record_sort_key))
    if any(record.trade.is_shadow for record in actual):
        raise ReportingIntegrityError("actual_trade_records cannot contain shadow trades")
    if any(not record.trade.is_shadow for record in shadow):
        raise ReportingIntegrityError("shadow_trade_records must contain only shadow trades")

    realized_ending_capital = _realized_ending_capital(actual, result.starting_capital)
    if realized_ending_capital != result.ending_capital:
        raise ReportingIntegrityError(
            "backtester realization groups must reconcile exactly to ending_capital"
        )
    trade_costs = _sum(record.trade.total_costs for record in actual)
    retained_costs = _sum(record.round_trip_cost_breakdown.total for record in actual)
    if trade_costs != retained_costs:
        raise ReportingIntegrityError(
            "actual trade total_costs must equal retained round-trip cost totals exactly"
        )
    known_dates = set(context.trading_dates)
    if any(
        record.trade.exit_fill.timestamp.astimezone(MARKET_TIMEZONE).date() not in known_dates
        for record in actual
    ):
        raise ReportingIntegrityError(
            "every actual trade exit date must be present in context.trading_dates"
        )
    return actual, shadow


def _verify_oos(
    result: BacktestRunResult,
    record: OOSTestRecord | None,
    fingerprint: str,
    context: ReportContext,
) -> dict[str, str | None]:
    empty = {
        "research_scope_id": None,
        "plan_id": None,
        "window_id": None,
        "oos_result_fingerprint": None,
    }
    if record is None:
        if context.oos_result_fingerprint is None:
            return empty
        if context.oos_result_fingerprint != fingerprint:
            raise ReportingIntegrityError(
                "pre-registration OOS result fingerprint does not match backtest result"
            )
        return {
            "research_scope_id": context.research_scope_id,
            "plan_id": context.plan_id,
            "window_id": context.window_id,
            "oos_result_fingerprint": context.oos_result_fingerprint,
        }
    if context.oos_result_fingerprint is not None:
        raise ValueError("OOS provenance must come from either context or record, not both")
    if not isinstance(record, OOSTestRecord):
        raise TypeError("oos_test_record must be an OOSTestRecord or None")
    comparisons = {
        "backtest_run_id": (record.backtest_run_id, result.run_id),
        "git_commit": (record.backtest_git_commit, result.git_commit),
        "backtester_version": (record.backtester_version, result.backtester_version),
        "window_start": (record.backtest_window_start, result.window_start),
        "window_end": (record.backtest_window_end, result.window_end),
        "cost_policy_id": (record.cost_policy_id, result.cost_policy_id),
        "brokerage_plan": (record.brokerage_plan, result.brokerage_plan.value),
        "symbols": (record.symbols, result.symbols),
        "strategy_versions": (record.strategy_versions, result.strategy_versions),
        "ml_model_versions": (record.ml_model_versions, result.ml_model_versions),
    }
    mismatches = [name for name, values in comparisons.items() if values[0] != values[1]]
    if mismatches:
        raise ReportingIntegrityError(
            f"OOS provenance does not match backtest result: {', '.join(mismatches)}"
        )
    if record.result_fingerprint != fingerprint:
        raise ReportingIntegrityError("OOS result fingerprint does not match backtest result")
    return {
        "research_scope_id": record.research_scope_id,
        "plan_id": record.plan_id,
        "window_id": record.window_id,
        "oos_result_fingerprint": record.result_fingerprint,
    }


def _cost_summary(
    records: tuple[BacktestTradeRecord, ...],
    economic_status: str,
) -> CostSummary:
    entry_turnover = _sum(record.round_trip_cost_breakdown.entry.turnover for record in records)
    exit_turnover = _sum(record.round_trip_cost_breakdown.exit.turnover for record in records)
    names = (
        "brokerage",
        "exchange_transaction_charge",
        "sebi_turnover_fee",
        "ipft",
        "stt",
        "stamp_duty",
        "gst",
    )
    components = {
        name: _sum(
            getattr(record.round_trip_cost_breakdown.entry, name)
            + getattr(record.round_trip_cost_breakdown.exit, name)
            for record in records
        )
        for name in names
    }
    total = _sum(components.values())
    percentages = {
        name: value / total if total > 0 else None for name, value in components.items()
    }
    return CostSummary(
        economic_status=economic_status,
        entry_turnover=entry_turnover,
        exit_turnover=exit_turnover,
        total_turnover=entry_turnover + exit_turnover,
        **components,
        total_costs=total,
        component_percentages=percentages,
    )


def _equity_curve(
    records: tuple[BacktestTradeRecord, ...],
    result: BacktestRunResult,
) -> tuple[EquityPoint, ...]:
    capital = result.starting_capital
    peak = capital
    cumulative = ZERO
    points = [
        EquityPoint(
            timestamp=result.window_start,
            realized_capital=capital,
            running_peak_capital=peak,
            drawdown_amount=ZERO,
            drawdown_pct=ZERO,
            group_net_pnl=ZERO,
            cumulative_net_pnl=ZERO,
        )
    ]
    for timestamp, group_pnl in _realization_groups(records):
        capital += group_pnl
        cumulative = capital - result.starting_capital
        peak = max(peak, capital)
        drawdown = peak - capital
        points.append(
            EquityPoint(
                timestamp=timestamp,
                realized_capital=capital,
                running_peak_capital=peak,
                drawdown_amount=drawdown,
                drawdown_pct=drawdown / peak if peak > 0 else ZERO,
                group_net_pnl=group_pnl,
                cumulative_net_pnl=cumulative,
            )
        )
    if points[-1].realized_capital != result.ending_capital:
        raise ReportingIntegrityError("final realized equity must equal ending_capital")
    return tuple(points)


def _daily_performance(
    records: tuple[BacktestTradeRecord, ...],
    result: BacktestRunResult,
    context: ReportContext,
) -> tuple[DailyPerformance, ...]:
    grouped: dict[object, list[BacktestTradeRecord]] = defaultdict(list)
    for record in records:
        day = record.trade.exit_fill.timestamp.astimezone(MARKET_TIMEZONE).date()
        grouped[day].append(record)

    groups_by_day: dict[object, list[Decimal]] = defaultdict(list)
    for timestamp, group_pnl in _realization_groups(records):
        day = timestamp.astimezone(MARKET_TIMEZONE).date()
        groups_by_day[day].append(group_pnl)

    cumulative = ZERO
    realized_capital = result.starting_capital
    rows = []
    for day in context.trading_dates:
        day_records = tuple(sorted(grouped[day], key=_record_sort_key))
        day_start_capital = realized_capital
        for group_pnl in groups_by_day[day]:
            realized_capital += group_pnl
        cumulative = realized_capital - result.starting_capital
        net_pnl = realized_capital - day_start_capital
        returns = tuple(record.trade.net_return for record in day_records)
        rows.append(
            DailyPerformance(
                trading_date=day,
                actual_trade_count=len(day_records),
                winning_trades=sum(record.trade.net_pnl > 0 for record in day_records),
                losing_trades=sum(record.trade.net_pnl < 0 for record in day_records),
                breakeven_trades=sum(record.trade.net_pnl == 0 for record in day_records),
                gross_pnl=_sum(record.trade.gross_pnl for record in day_records),
                total_costs=_sum(record.trade.total_costs for record in day_records),
                net_pnl=net_pnl,
                cumulative_net_pnl=cumulative,
                realized_end_capital=realized_capital,
                average_trade_net_return=_mean(returns),
            )
        )
    return tuple(rows)


def _trade_counts(
    records: tuple[BacktestTradeRecord, ...],
) -> tuple[int, int, int]:
    return (
        sum(record.trade.net_pnl > 0 for record in records),
        sum(record.trade.net_pnl < 0 for record in records),
        sum(record.trade.net_pnl == 0 for record in records),
    )


def _strategy_metrics(
    records: tuple[BacktestTradeRecord, ...],
    economic_status: str,
) -> tuple[StrategyTradeMetrics, ...]:
    grouped: dict[tuple[str, str], list[BacktestTradeRecord]] = defaultdict(list)
    for record in records:
        signal = record.trade.signal
        grouped[(signal.strategy_id, signal.strategy_version)].append(record)
    output = []
    for (strategy_id, strategy_version), values in sorted(grouped.items()):
        group = tuple(values)
        wins, losses, breakeven = _trade_counts(group)
        output.append(
            StrategyTradeMetrics(
                strategy_id=strategy_id,
                strategy_version=strategy_version,
                economic_status=economic_status,
                trade_count=len(group),
                wins=wins,
                losses=losses,
                breakeven=breakeven,
                long_trade_count=sum(record.trade.signal.side is Side.LONG for record in group),
                short_trade_count=sum(record.trade.signal.side is Side.SHORT for record in group),
                net_pnl=_sum(record.trade.net_pnl for record in group),
                average_net_pnl=_mean(tuple(record.trade.net_pnl for record in group)),
                average_net_return=_mean(tuple(record.trade.net_return for record in group)),
                win_rate=_ratio(wins, len(group)),
                profit_factor=_profit_factor(group),
                total_costs=_sum(record.trade.total_costs for record in group),
                average_quality_score=_mean(tuple(_quality(record) for record in group)),
            )
        )
    return tuple(output)


def _symbol_metrics(
    records: tuple[BacktestTradeRecord, ...],
) -> tuple[SymbolTradeMetrics, ...]:
    grouped: dict[str, list[BacktestTradeRecord]] = defaultdict(list)
    for record in records:
        grouped[record.trade.signal.symbol].append(record)
    output = []
    for symbol, values in sorted(grouped.items()):
        group = tuple(values)
        wins, losses, breakeven = _trade_counts(group)
        output.append(
            SymbolTradeMetrics(
                symbol=symbol,
                trade_count=len(group),
                wins=wins,
                losses=losses,
                breakeven=breakeven,
                long_count=sum(record.trade.signal.side is Side.LONG for record in group),
                short_count=sum(record.trade.signal.side is Side.SHORT for record in group),
                net_pnl=_sum(record.trade.net_pnl for record in group),
                average_net_return=_mean(tuple(record.trade.net_return for record in group)),
                win_rate=_ratio(wins, len(group)),
                profit_factor=_profit_factor(group),
                total_costs=_sum(record.trade.total_costs for record in group),
            )
        )
    return tuple(output)


def _exit_reason_metrics(
    records: tuple[BacktestTradeRecord, ...],
    economic_status: str,
) -> tuple[ExitReasonMetrics, ...]:
    grouped: dict[ExitReason, list[BacktestTradeRecord]] = defaultdict(list)
    for record in records:
        grouped[record.trade.exit_reason].append(record)
    output = []
    for reason in ExitReason:
        group = tuple(grouped[reason])
        output.append(
            ExitReasonMetrics(
                exit_reason=reason,
                economic_status=economic_status,
                trade_count=len(group),
                net_pnl=_sum(record.trade.net_pnl for record in group),
                average_net_pnl=_mean(tuple(record.trade.net_pnl for record in group)),
                average_net_return=_mean(tuple(record.trade.net_return for record in group)),
            )
        )
    return tuple(output)


def _request_summary(result: BacktestRunResult) -> RequestOutcomeSummary:
    counts = Counter(request.outcome for request in result.request_results)
    total = len(result.request_results)
    actual = counts[BacktestRequestOutcome.COMPLETED_ACTUAL]
    shadow = counts[BacktestRequestOutcome.COMPLETED_SHADOW]
    allocated_no_fill = counts[BacktestRequestOutcome.ALLOCATED_ENTRY_NOT_FILLED]
    shadow_no_fill = counts[BacktestRequestOutcome.SHADOW_ENTRY_NOT_FILLED]
    exhausted = counts[BacktestRequestOutcome.CAPITAL_EXHAUSTED]
    if actual + shadow + allocated_no_fill + shadow_no_fill + exhausted != total:
        raise ReportingIntegrityError("request outcome counts must reconcile to total_requests")
    rejected = shadow + shadow_no_fill
    return RequestOutcomeSummary(
        total_requests=total,
        completed_actual=actual,
        completed_shadow=shadow,
        allocated_entry_not_filled=allocated_no_fill,
        shadow_entry_not_filled=shadow_no_fill,
        capital_exhausted=exhausted,
        capacity_rejected_request_count=rejected,
        actual_completion_rate=_ratio(actual, total) if total else None,
        capacity_rejection_rate=_ratio(rejected, total) if total else None,
        no_fill_rate=_ratio(allocated_no_fill + shadow_no_fill, total) if total else None,
    )


def _calculate_cagr(result: BacktestRunResult) -> Decimal | None:
    if result.ending_capital <= 0:
        return None
    elapsed_seconds = (result.window_end - result.window_start).total_seconds()
    elapsed_days = Decimal(str(elapsed_seconds)) / Decimal("86400")
    if elapsed_days <= 0:
        raise ReportingIntegrityError("backtest window must have positive elapsed duration")
    ratio = float(result.ending_capital / result.starting_capital)
    exponent = float(Decimal("365.2425") / elapsed_days)
    return Decimal(str(ratio**exponent - 1))


def build_report(
    result: BacktestRunResult,
    context: ReportContext,
    oos_test_record: OOSTestRecord | None = None,
) -> ReportBundle:
    """Build deterministic validated analytics without mutating source records."""
    actual, shadow = _validate_result(result, context)
    fingerprint = fingerprint_backtest_result(result)
    oos_values = _verify_oos(result, oos_test_record, fingerprint, context)
    actual_costs = _cost_summary(actual, "ACTUAL")
    shadow_costs = _cost_summary(shadow, "SHADOW / HYPOTHETICAL - NO ACTUAL CAPITAL IMPACT")
    equity = _equity_curve(actual, result)
    daily = _daily_performance(actual, result, context)
    wins, losses, breakeven = _trade_counts(actual)
    actual_count = len(actual)
    returns = tuple(record.trade.net_return for record in actual)
    pnl_values = tuple(record.trade.net_pnl for record in actual)
    gross_positive = _sum(value for value in pnl_values if value > 0)
    gross_negative = -_sum(value for value in pnl_values if value < 0)
    gross_profit = _sum(record.trade.gross_pnl for record in actual if record.trade.gross_pnl > 0)
    total_return = (result.ending_capital - result.starting_capital) / result.starting_capital
    cagr = _calculate_cagr(result)
    max_drawdown = max(point.drawdown_amount for point in equity)
    max_drawdown_pct = max(point.drawdown_pct for point in equity)
    profit_factor = _profit_factor(actual)
    performance = PerformanceMetrics(
        starting_capital=result.starting_capital,
        ending_capital=result.ending_capital,
        net_profit=result.ending_capital - result.starting_capital,
        total_return=total_return,
        cagr=cagr,
        actual_trade_count=actual_count,
        winning_trade_count=wins,
        losing_trade_count=losses,
        breakeven_trade_count=breakeven,
        win_rate=_ratio(wins, actual_count) if actual_count else None,
        gross_positive_net_pnl=gross_positive,
        gross_negative_net_pnl_absolute=gross_negative,
        net_profit_factor=profit_factor,
        average_net_pnl_per_trade=_mean(pnl_values),
        average_net_return_per_trade=_mean(returns),
        median_net_return_per_trade=_median(returns),
        best_trade_net_pnl=max(pnl_values) if pnl_values else None,
        worst_trade_net_pnl=min(pnl_values) if pnl_values else None,
        best_trade_net_return=max(returns) if returns else None,
        worst_trade_net_return=min(returns) if returns else None,
        total_costs=actual_costs.total_costs,
        average_cost_per_trade=actual_costs.total_costs / actual_count if actual_count else None,
        costs_as_pct_of_gross_profit=(
            actual_costs.total_costs / gross_profit if gross_profit > 0 else None
        ),
        average_mfe_return=_mean(tuple(record.trade.mfe_return for record in actual)),
        average_mae_return=_mean(tuple(record.trade.mae_return for record in actual)),
        maximum_realized_drawdown=max_drawdown,
        maximum_realized_drawdown_pct=max_drawdown_pct,
        evaluation_trading_days=len(context.trading_dates),
        actual_trades_per_day=_ratio(actual_count, len(context.trading_dates)),
        average_quality_score=_mean(tuple(_quality(record) for record in actual)),
    )
    shadow_wins, shadow_losses, shadow_breakeven = _trade_counts(shadow)
    shadow_returns = tuple(record.trade.net_return for record in shadow)
    shadow_pnl = tuple(record.trade.net_pnl for record in shadow)
    shadow_metrics = ShadowMetrics(
        shadow_trade_count=len(shadow),
        winning_trade_count=shadow_wins,
        losing_trade_count=shadow_losses,
        breakeven_trade_count=shadow_breakeven,
        hypothetical_net_pnl=_sum(shadow_pnl),
        average_net_pnl=_mean(shadow_pnl),
        average_net_return=_mean(shadow_returns),
        win_rate=_ratio(shadow_wins, len(shadow)) if shadow else None,
        profit_factor=_profit_factor(shadow),
        hypothetical_total_costs=shadow_costs.total_costs,
        average_quality_score=_mean(tuple(_quality(record) for record in shadow)),
        best_net_return=max(shadow_returns) if shadow_returns else None,
        worst_net_return=min(shadow_returns) if shadow_returns else None,
    )
    pf_pass = profit_factor.is_unbounded or (
        profit_factor.value is not None and profit_factor.value > Decimal("2")
    )
    cagr_pass = cagr is not None and cagr > Decimal("0.20")
    win_rate_pass = performance.win_rate is not None and performance.win_rate > Decimal("0.50")
    average_return_pass = (
        performance.average_net_return_per_trade is not None
        and performance.average_net_return_per_trade > Decimal("0.005")
    )
    acceptance = AcceptanceAssessment(
        cagr_pass=cagr_pass,
        win_rate_pass=win_rate_pass,
        profit_factor_pass=pf_pass,
        average_net_return_pass=average_return_pass,
        frequency_target_met=performance.actual_trades_per_day >= Decimal("2"),
        hard_quantitative_targets_pass=(
            cagr_pass and win_rate_pass and pf_pass and average_return_pass
        ),
    )
    provenance = ReportProvenance(
        report_id=context.report_id,
        reporting_version=REPORTING_VERSION,
        generated_at=context.generated_at,
        source_backtest_fingerprint=fingerprint,
        run_id=result.run_id,
        git_commit=result.git_commit,
        backtester_version=result.backtester_version,
        window_start=result.window_start,
        window_end=result.window_end,
        cost_policy_id=result.cost_policy_id,
        cost_policy_source_as_of_date=result.cost_policy_source_as_of_date,
        brokerage_plan=result.brokerage_plan.value,
        starting_capital=result.starting_capital,
        ending_capital=result.ending_capital,
        capital_exhausted=result.capital_exhausted,
        symbols=tuple(sorted(result.symbols)),
        strategy_versions=tuple(sorted(result.strategy_versions)),
        ml_model_versions=tuple(sorted(result.ml_model_versions)),
        evaluation_trading_dates=context.trading_dates,
        **oos_values,
    )
    return ReportBundle(
        provenance=provenance,
        performance=performance,
        shadow_metrics=shadow_metrics,
        request_outcomes=_request_summary(result),
        actual_costs=actual_costs,
        shadow_costs=shadow_costs,
        equity_curve=equity,
        daily_performance=daily,
        actual_strategy_breakdown=_strategy_metrics(actual, "ACTUAL"),
        shadow_strategy_breakdown=_strategy_metrics(
            shadow, "SHADOW / HYPOTHETICAL - NO ACTUAL CAPITAL IMPACT"
        ),
        symbol_breakdown=_symbol_metrics(actual),
        actual_exit_reason_breakdown=_exit_reason_metrics(actual, "ACTUAL"),
        shadow_exit_reason_breakdown=_exit_reason_metrics(
            shadow, "SHADOW / HYPOTHETICAL - NO ACTUAL CAPITAL IMPACT"
        ),
        acceptance=acceptance,
        actual_trade_records=actual,
        shadow_trade_records=shadow,
    )
