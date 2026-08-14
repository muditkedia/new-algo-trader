"""Canonical deterministic Polars tables and Parquet reporting dataset."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from algo_trader.backtest import BacktestTradeRecord
from algo_trader.domain import Side
from algo_trader.reporting.models import ProfitFactor, ReportBundle

DECIMAL = pl.Decimal(precision=38, scale=28)
MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")
ZERO = Decimal("0")

REPORT_TABLE_FILENAMES = {
    "summary": "summary.parquet",
    "actual_trades": "actual_trades.parquet",
    "shadow_trades": "shadow_trades.parquet",
    "request_outcomes": "request_outcomes.parquet",
    "equity_curve": "equity_curve.parquet",
    "daily_performance": "daily_performance.parquet",
    "actual_strategy_breakdown": "actual_strategy_breakdown.parquet",
    "shadow_strategy_breakdown": "shadow_strategy_breakdown.parquet",
    "symbol_breakdown": "symbol_breakdown.parquet",
    "actual_cost_breakdown": "actual_cost_breakdown.parquet",
    "shadow_cost_breakdown": "shadow_cost_breakdown.parquet",
    "actual_exit_reason_breakdown": "actual_exit_reason_breakdown.parquet",
    "shadow_exit_reason_breakdown": "shadow_exit_reason_breakdown.parquet",
    "provenance": "provenance.parquet",
    "cumulative_pnl": "cumulative_pnl.parquet",
    "monthly_performance": "monthly_performance.parquet",
    "side_performance": "side_performance.parquet",
    "time_of_day_performance": "time_of_day_performance.parquet",
    "holding_time_distribution": "holding_time_distribution.parquet",
    "rolling_trade_metrics": "rolling_trade_metrics.parquet",
    "trade_diagnostics": "trade_diagnostics.parquet",
    "cost_impact": "cost_impact.parquet",
    "outcome_funnel": "outcome_funnel.parquet",
    "actual_shadow_comparison": "actual_shadow_comparison.parquet",
}


def _frame(rows: list[dict[str, Any]], schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=schema, strict=False).select(*schema)


def _profit_factor_columns(value: ProfitFactor) -> dict[str, object]:
    return {
        "profit_factor": value.value,
        "profit_factor_is_unbounded": value.is_unbounded,
        "profit_factor_is_undefined": value.is_undefined,
    }


def _trade_rows(
    report: ReportBundle,
    records: tuple[BacktestTradeRecord, ...],
    economic_status: str,
) -> list[dict[str, object]]:
    rows = []
    for record in records:
        trade = record.trade
        candidate = record.allocation_decision.candidate
        order = candidate.order_intent
        costs = record.round_trip_cost_breakdown
        component_values = {
            name: getattr(costs.entry, name) + getattr(costs.exit, name)
            for name in (
                "brokerage",
                "exchange_transaction_charge",
                "sebi_turnover_fee",
                "ipft",
                "stt",
                "stamp_duty",
                "gst",
            )
        }
        rows.append(
            {
                "report_id": report.provenance.report_id,
                "run_id": report.provenance.run_id,
                "strategy_id": trade.signal.strategy_id,
                "strategy_version": trade.signal.strategy_version,
                "symbol": trade.signal.symbol,
                "side": trade.signal.side.value,
                "signal_timestamp": trade.signal.timestamp,
                "order_timestamp": order.timestamp,
                "entry_timestamp": trade.entry_fill.timestamp,
                "exit_timestamp": trade.exit_fill.timestamp,
                "quantity": trade.entry_fill.quantity,
                "target_notional": trade.target_notional,
                "actual_entry_notional": trade.actual_entry_notional,
                "entry_price": trade.entry_fill.price,
                "exit_price": trade.exit_fill.price,
                "gross_pnl": trade.gross_pnl,
                "total_costs": trade.total_costs,
                "net_pnl": trade.net_pnl,
                "gross_return": trade.gross_return,
                "net_return": trade.net_return,
                "mfe_return": trade.mfe_return,
                "mae_return": trade.mae_return,
                "exit_reason": trade.exit_reason.value,
                "ml_model_version": trade.ml_score.model_version,
                "ml_quality_score": trade.ml_score.quality_score,
                "ml_calibrated_probability": trade.ml_score.calibrated_probability,
                "ml_predicted_net_return": trade.ml_score.predicted_net_return,
                "ml_recommended_notional": trade.ml_score.recommended_notional,
                "margin_provider_id": record.allocation_decision.margin_quote.provider_id,
                "required_margin": record.allocation_decision.margin_quote.required_margin,
                "cost_policy_id": record.cost_policy_id,
                "entry_turnover": costs.entry.turnover,
                "exit_turnover": costs.exit.turnover,
                **component_values,
                "is_shadow": trade.is_shadow,
                "economic_status": economic_status,
            }
        )
    return rows


TRADE_SCHEMA = {
    "report_id": pl.String,
    "run_id": pl.String,
    "strategy_id": pl.String,
    "strategy_version": pl.String,
    "symbol": pl.String,
    "side": pl.String,
    "signal_timestamp": pl.Datetime(time_unit="us", time_zone="Asia/Kolkata"),
    "order_timestamp": pl.Datetime(time_unit="us", time_zone="Asia/Kolkata"),
    "entry_timestamp": pl.Datetime(time_unit="us", time_zone="Asia/Kolkata"),
    "exit_timestamp": pl.Datetime(time_unit="us", time_zone="Asia/Kolkata"),
    "quantity": pl.Int64,
    "target_notional": pl.Int64,
    "actual_entry_notional": DECIMAL,
    "entry_price": DECIMAL,
    "exit_price": DECIMAL,
    "gross_pnl": DECIMAL,
    "total_costs": DECIMAL,
    "net_pnl": DECIMAL,
    "gross_return": DECIMAL,
    "net_return": DECIMAL,
    "mfe_return": DECIMAL,
    "mae_return": DECIMAL,
    "exit_reason": pl.String,
    "ml_model_version": pl.String,
    "ml_quality_score": pl.Float64,
    "ml_calibrated_probability": pl.Float64,
    "ml_predicted_net_return": pl.Float64,
    "ml_recommended_notional": pl.Int64,
    "margin_provider_id": pl.String,
    "required_margin": DECIMAL,
    "cost_policy_id": pl.String,
    "entry_turnover": DECIMAL,
    "exit_turnover": DECIMAL,
    "brokerage": DECIMAL,
    "exchange_transaction_charge": DECIMAL,
    "sebi_turnover_fee": DECIMAL,
    "ipft": DECIMAL,
    "stt": DECIMAL,
    "stamp_duty": DECIMAL,
    "gst": DECIMAL,
    "is_shadow": pl.Boolean,
    "economic_status": pl.String,
}


def _summary_table(report: ReportBundle) -> pl.DataFrame:
    metrics = report.performance
    return _frame(
        [
            {
                "report_id": report.provenance.report_id,
                "run_id": report.provenance.run_id,
                **metrics.model_dump(mode="python", exclude={"net_profit_factor"}),
                **_profit_factor_columns(metrics.net_profit_factor),
                **report.acceptance.model_dump(mode="python"),
            }
        ],
        {
            "report_id": pl.String,
            "run_id": pl.String,
            "starting_capital": DECIMAL,
            "ending_capital": DECIMAL,
            "net_profit": DECIMAL,
            "total_return": DECIMAL,
            "cagr": DECIMAL,
            "actual_trade_count": pl.Int64,
            "winning_trade_count": pl.Int64,
            "losing_trade_count": pl.Int64,
            "breakeven_trade_count": pl.Int64,
            "win_rate": DECIMAL,
            "gross_positive_net_pnl": DECIMAL,
            "gross_negative_net_pnl_absolute": DECIMAL,
            "average_net_pnl_per_trade": DECIMAL,
            "average_net_return_per_trade": DECIMAL,
            "median_net_return_per_trade": DECIMAL,
            "best_trade_net_pnl": DECIMAL,
            "worst_trade_net_pnl": DECIMAL,
            "best_trade_net_return": DECIMAL,
            "worst_trade_net_return": DECIMAL,
            "total_costs": DECIMAL,
            "average_cost_per_trade": DECIMAL,
            "costs_as_pct_of_gross_profit": DECIMAL,
            "average_mfe_return": DECIMAL,
            "average_mae_return": DECIMAL,
            "maximum_realized_drawdown": DECIMAL,
            "maximum_realized_drawdown_pct": DECIMAL,
            "evaluation_trading_days": pl.Int64,
            "actual_trades_per_day": DECIMAL,
            "average_quality_score": DECIMAL,
            "profit_factor": DECIMAL,
            "profit_factor_is_unbounded": pl.Boolean,
            "profit_factor_is_undefined": pl.Boolean,
            "cagr_pass": pl.Boolean,
            "win_rate_pass": pl.Boolean,
            "profit_factor_pass": pl.Boolean,
            "average_net_return_pass": pl.Boolean,
            "frequency_target_met": pl.Boolean,
            "hard_quantitative_targets_pass": pl.Boolean,
        },
    )


def _strategy_table(report: ReportBundle, *, shadow: bool) -> pl.DataFrame:
    values = report.shadow_strategy_breakdown if shadow else report.actual_strategy_breakdown
    rows = []
    for value in values:
        rows.append(
            {
                **value.model_dump(mode="python", exclude={"profit_factor"}),
                **_profit_factor_columns(value.profit_factor),
            }
        )
    return _frame(
        rows,
        {
            "strategy_id": pl.String,
            "strategy_version": pl.String,
            "economic_status": pl.String,
            "trade_count": pl.Int64,
            "wins": pl.Int64,
            "losses": pl.Int64,
            "breakeven": pl.Int64,
            "long_trade_count": pl.Int64,
            "short_trade_count": pl.Int64,
            "net_pnl": DECIMAL,
            "average_net_pnl": DECIMAL,
            "average_net_return": DECIMAL,
            "win_rate": DECIMAL,
            "total_costs": DECIMAL,
            "average_quality_score": DECIMAL,
            "profit_factor": DECIMAL,
            "profit_factor_is_unbounded": pl.Boolean,
            "profit_factor_is_undefined": pl.Boolean,
        },
    )


def _cost_table(summary: object) -> pl.DataFrame:
    rows = []
    for name in (
        "entry_turnover",
        "exit_turnover",
        "total_turnover",
        "brokerage",
        "exchange_transaction_charge",
        "sebi_turnover_fee",
        "ipft",
        "stt",
        "stamp_duty",
        "gst",
        "total_costs",
    ):
        is_component = name in summary.component_percentages
        rows.append(
            {
                "economic_status": summary.economic_status,
                "component": name,
                "amount": getattr(summary, name),
                "percentage_of_total_costs": (
                    summary.component_percentages[name] if is_component else None
                ),
                "is_cost_component": is_component,
            }
        )
    return _frame(
        rows,
        {
            "economic_status": pl.String,
            "component": pl.String,
            "amount": DECIMAL,
            "percentage_of_total_costs": DECIMAL,
            "is_cost_component": pl.Boolean,
        },
    )


def _exit_table(values: tuple[object, ...]) -> pl.DataFrame:
    return _frame(
        [
            {
                **value.model_dump(mode="python"),
                "exit_reason": value.exit_reason.value,
            }
            for value in values
        ],
        {
            "exit_reason": pl.String,
            "economic_status": pl.String,
            "trade_count": pl.Int64,
            "net_pnl": DECIMAL,
            "average_net_pnl": DECIMAL,
            "average_net_return": DECIMAL,
        },
    )


def _factor_from_records(records: tuple[BacktestTradeRecord, ...]) -> ProfitFactor:
    positive = sum(
        (record.trade.net_pnl for record in records if record.trade.net_pnl > 0),
        start=ZERO,
    )
    negative = -sum(
        (record.trade.net_pnl for record in records if record.trade.net_pnl < 0),
        start=ZERO,
    )
    if negative > 0:
        return ProfitFactor(value=positive / negative)
    if positive > 0:
        return ProfitFactor(value=None, is_unbounded=True)
    return ProfitFactor(value=None, is_undefined=True)


def _aggregate_rows(
    groups: list[tuple[str, tuple[BacktestTradeRecord, ...]]],
    label_name: str,
) -> pl.DataFrame:
    rows = []
    for label, records in groups:
        wins = sum(record.trade.net_pnl > 0 for record in records)
        factor = _factor_from_records(records)
        rows.append(
            {
                label_name: label,
                "trade_count": len(records),
                "net_pnl": sum(
                    (record.trade.net_pnl for record in records), start=ZERO
                ),
                "win_rate": Decimal(wins) / len(records) if records else None,
                "average_net_return": (
                    sum(
                        (record.trade.net_return for record in records), start=ZERO
                    )
                    / len(records)
                    if records
                    else None
                ),
                "total_costs": sum(
                    (record.trade.total_costs for record in records), start=ZERO
                ),
                **_profit_factor_columns(factor),
            }
        )
    return _frame(
        rows,
        {
            label_name: pl.String,
            "trade_count": pl.Int64,
            "net_pnl": DECIMAL,
            "win_rate": DECIMAL,
            "average_net_return": DECIMAL,
            "total_costs": DECIMAL,
            "profit_factor": DECIMAL,
            "profit_factor_is_unbounded": pl.Boolean,
            "profit_factor_is_undefined": pl.Boolean,
        },
    )


def _diagnostic_tables(report: ReportBundle) -> dict[str, pl.DataFrame]:
    records = report.actual_trade_records
    cumulative_gross = ZERO
    cumulative_net = ZERO
    cumulative_cost = ZERO
    cumulative_rows = []
    diagnostic_rows = []
    for trade_number, record in enumerate(records, start=1):
        trade = record.trade
        cumulative_gross += trade.gross_pnl
        cumulative_net += trade.net_pnl
        cumulative_cost += trade.total_costs
        cumulative_rows.append(
            {
                "trade_number": trade_number,
                "exit_timestamp": trade.exit_fill.timestamp,
                "cumulative_gross_pnl": cumulative_gross,
                "cumulative_net_pnl": cumulative_net,
                "cumulative_cost_drag": cumulative_cost,
            }
        )
        holding_minutes = Decimal(
            str((trade.exit_fill.timestamp - trade.entry_fill.timestamp).total_seconds())
        ) / Decimal("60")
        diagnostic_rows.append(
            {
                "trade_number": trade_number,
                "symbol": trade.signal.symbol,
                "side": trade.signal.side.value,
                "signal_timestamp": trade.signal.timestamp,
                "entry_timestamp": trade.entry_fill.timestamp,
                "exit_timestamp": trade.exit_fill.timestamp,
                "holding_minutes": holding_minutes,
                "mfe_return": trade.mfe_return,
                "mae_return": trade.mae_return,
                "gross_return": trade.gross_return,
                "net_return": trade.net_return,
            }
        )

    grouped_month: dict[str, list[BacktestTradeRecord]] = defaultdict(list)
    grouped_side: dict[Side, list[BacktestTradeRecord]] = defaultdict(list)
    grouped_time: dict[str, list[BacktestTradeRecord]] = defaultdict(list)
    for record in records:
        trade = record.trade
        local_exit = trade.exit_fill.timestamp.astimezone(MARKET_TIMEZONE)
        grouped_month[f"{local_exit.year:04d}-{local_exit.month:02d}"].append(record)
        grouped_side[trade.signal.side].append(record)
        local_entry = trade.entry_fill.timestamp.astimezone(MARKET_TIMEZONE)
        bucket_minute = 30 * (local_entry.minute // 30)
        grouped_time[f"{local_entry.hour:02d}:{bucket_minute:02d}"].append(record)

    monthly = _aggregate_rows(
        [(key, tuple(grouped_month[key])) for key in sorted(grouped_month)],
        "month",
    )
    side = _aggregate_rows(
        [(value.value, tuple(grouped_side[value])) for value in Side],
        "side",
    )
    time_of_day = _aggregate_rows(
        [(key, tuple(grouped_time[key])) for key in sorted(grouped_time)],
        "entry_time_bucket_ist",
    )

    rolling_rows = []
    for end in range(20, len(records) + 1):
        window = records[end - 20 : end]
        wins = sum(record.trade.net_pnl > 0 for record in window)
        factor = _factor_from_records(window)
        rolling_rows.append(
            {
                "trade_number": end,
                "window_end_timestamp": window[-1].trade.exit_fill.timestamp,
                "rolling_win_rate": Decimal(wins) / Decimal("20"),
                "rolling_average_net_return": sum(
                    (record.trade.net_return for record in window), start=ZERO
                )
                / Decimal("20"),
                **_profit_factor_columns(factor),
            }
        )

    holding_bounds = (
        ("0-5", Decimal("0"), Decimal("5")),
        (">5-15", Decimal("5"), Decimal("15")),
        (">15-30", Decimal("15"), Decimal("30")),
        (">30-60", Decimal("30"), Decimal("60")),
        (">60-120", Decimal("60"), Decimal("120")),
        (">120", Decimal("120"), None),
    )
    holding_rows = []
    holding_values = [row["holding_minutes"] for row in diagnostic_rows]
    for label, lower, upper in holding_bounds:
        count = sum(
            value >= lower
            and (value <= upper if lower == 0 and upper is not None else True)
            and (value > lower if lower > 0 else True)
            and (upper is None or value <= upper)
            for value in holding_values
        )
        holding_rows.append({"bucket": label, "trade_count": count})

    costs = report.actual_costs
    gross_profit = sum(
        (record.trade.gross_pnl for record in records if record.trade.gross_pnl > 0),
        start=ZERO,
    )
    gross_loss = -sum(
        (record.trade.gross_pnl for record in records if record.trade.gross_pnl < 0),
        start=ZERO,
    )
    cost_rows = [
        {"component": "gross_profit", "amount": gross_profit},
        {"component": "gross_loss_absolute", "amount": gross_loss},
        *[
            {"component": name, "amount": getattr(costs, name)}
            for name in costs.component_percentages
        ],
        {"component": "final_net_pnl", "amount": report.performance.net_profit},
    ]
    request = report.request_outcomes
    funnel_rows = [
        {"stage": "generated_requests", "count": request.total_requests},
        {
            "stage": "allocated",
            "count": request.completed_actual + request.allocated_entry_not_filled,
        },
        {
            "stage": "capacity_rejected",
            "count": request.capacity_rejected_request_count,
        },
        {
            "stage": "filled",
            "count": request.completed_actual + request.completed_shadow,
        },
        {
            "stage": "not_filled",
            "count": request.allocated_entry_not_filled
            + request.shadow_entry_not_filled,
        },
        {
            "stage": "completed",
            "count": request.completed_actual + request.completed_shadow,
        },
    ]
    shadow = report.shadow_metrics
    comparison_rows = [
        {
            "economic_status": "ACTUAL",
            "trade_count": report.performance.actual_trade_count,
            "net_pnl": report.performance.net_profit,
            "total_costs": report.performance.total_costs,
            "win_rate": report.performance.win_rate,
            "average_net_return": report.performance.average_net_return_per_trade,
            **_profit_factor_columns(report.performance.net_profit_factor),
        },
        {
            "economic_status": shadow.economic_status,
            "trade_count": shadow.shadow_trade_count,
            "net_pnl": shadow.hypothetical_net_pnl,
            "total_costs": shadow.hypothetical_total_costs,
            "win_rate": shadow.win_rate,
            "average_net_return": shadow.average_net_return,
            **_profit_factor_columns(shadow.profit_factor),
        },
    ]
    return {
        "cumulative_pnl": _frame(
            cumulative_rows,
            {
                "trade_number": pl.Int64,
                "exit_timestamp": pl.Datetime(
                    time_unit="us", time_zone="Asia/Kolkata"
                ),
                "cumulative_gross_pnl": DECIMAL,
                "cumulative_net_pnl": DECIMAL,
                "cumulative_cost_drag": DECIMAL,
            },
        ),
        "monthly_performance": monthly,
        "side_performance": side,
        "time_of_day_performance": time_of_day,
        "holding_time_distribution": _frame(
            holding_rows, {"bucket": pl.String, "trade_count": pl.Int64}
        ),
        "rolling_trade_metrics": _frame(
            rolling_rows,
            {
                "trade_number": pl.Int64,
                "window_end_timestamp": pl.Datetime(
                    time_unit="us", time_zone="Asia/Kolkata"
                ),
                "rolling_win_rate": DECIMAL,
                "rolling_average_net_return": DECIMAL,
                "profit_factor": DECIMAL,
                "profit_factor_is_unbounded": pl.Boolean,
                "profit_factor_is_undefined": pl.Boolean,
            },
        ),
        "trade_diagnostics": _frame(
            diagnostic_rows,
            {
                "trade_number": pl.Int64,
                "symbol": pl.String,
                "side": pl.String,
                "signal_timestamp": pl.Datetime(
                    time_unit="us", time_zone="Asia/Kolkata"
                ),
                "entry_timestamp": pl.Datetime(
                    time_unit="us", time_zone="Asia/Kolkata"
                ),
                "exit_timestamp": pl.Datetime(
                    time_unit="us", time_zone="Asia/Kolkata"
                ),
                "holding_minutes": DECIMAL,
                "mfe_return": DECIMAL,
                "mae_return": DECIMAL,
                "gross_return": DECIMAL,
                "net_return": DECIMAL,
            },
        ),
        "cost_impact": _frame(
            cost_rows, {"component": pl.String, "amount": DECIMAL}
        ),
        "outcome_funnel": _frame(
            funnel_rows, {"stage": pl.String, "count": pl.Int64}
        ),
        "actual_shadow_comparison": _frame(
            comparison_rows,
            {
                "economic_status": pl.String,
                "trade_count": pl.Int64,
                "net_pnl": DECIMAL,
                "total_costs": DECIMAL,
                "win_rate": DECIMAL,
                "average_net_return": DECIMAL,
                "profit_factor": DECIMAL,
                "profit_factor_is_unbounded": pl.Boolean,
                "profit_factor_is_undefined": pl.Boolean,
            },
        ),
    }


def report_tables(report: ReportBundle) -> Mapping[str, pl.DataFrame]:
    """Return stable canonical Polars tables derived only from ``report``."""
    if not isinstance(report, ReportBundle):
        raise TypeError("report must be a ReportBundle")
    request = report.request_outcomes
    provenance = report.provenance
    symbol_rows = []
    for value in report.symbol_breakdown:
        symbol_rows.append(
            {
                **value.model_dump(mode="python", exclude={"profit_factor"}),
                **_profit_factor_columns(value.profit_factor),
            }
        )
    tables = {
        "summary": _summary_table(report),
        "actual_trades": _frame(
            _trade_rows(report, report.actual_trade_records, "ACTUAL"), TRADE_SCHEMA
        ),
        "shadow_trades": _frame(
            _trade_rows(
                report,
                report.shadow_trade_records,
                "SHADOW / HYPOTHETICAL - NO ACTUAL CAPITAL IMPACT",
            ),
            TRADE_SCHEMA,
        ),
        "request_outcomes": _frame(
            [request.model_dump(mode="python")],
            {
                "total_requests": pl.Int64,
                "completed_actual": pl.Int64,
                "completed_shadow": pl.Int64,
                "allocated_entry_not_filled": pl.Int64,
                "shadow_entry_not_filled": pl.Int64,
                "capital_exhausted": pl.Int64,
                "capacity_rejected_request_count": pl.Int64,
                "actual_completion_rate": DECIMAL,
                "capacity_rejection_rate": DECIMAL,
                "no_fill_rate": DECIMAL,
            },
        ),
        "equity_curve": _frame(
            [value.model_dump(mode="python") for value in report.equity_curve],
            {
                "timestamp": pl.Datetime(time_unit="us", time_zone="Asia/Kolkata"),
                "realized_capital": DECIMAL,
                "running_peak_capital": DECIMAL,
                "drawdown_amount": DECIMAL,
                "drawdown_pct": DECIMAL,
                "group_net_pnl": DECIMAL,
                "cumulative_net_pnl": DECIMAL,
            },
        ),
        "daily_performance": _frame(
            [value.model_dump(mode="python") for value in report.daily_performance],
            {
                "trading_date": pl.Date,
                "actual_trade_count": pl.Int64,
                "winning_trades": pl.Int64,
                "losing_trades": pl.Int64,
                "breakeven_trades": pl.Int64,
                "gross_pnl": DECIMAL,
                "total_costs": DECIMAL,
                "net_pnl": DECIMAL,
                "cumulative_net_pnl": DECIMAL,
                "realized_end_capital": DECIMAL,
                "average_trade_net_return": DECIMAL,
            },
        ),
        "actual_strategy_breakdown": _strategy_table(report, shadow=False),
        "shadow_strategy_breakdown": _strategy_table(report, shadow=True),
        "symbol_breakdown": _frame(
            symbol_rows,
            {
                "symbol": pl.String,
                "trade_count": pl.Int64,
                "wins": pl.Int64,
                "losses": pl.Int64,
                "breakeven": pl.Int64,
                "long_count": pl.Int64,
                "short_count": pl.Int64,
                "net_pnl": DECIMAL,
                "average_net_return": DECIMAL,
                "win_rate": DECIMAL,
                "total_costs": DECIMAL,
                "profit_factor": DECIMAL,
                "profit_factor_is_unbounded": pl.Boolean,
                "profit_factor_is_undefined": pl.Boolean,
            },
        ),
        "actual_cost_breakdown": _cost_table(report.actual_costs),
        "shadow_cost_breakdown": _cost_table(report.shadow_costs),
        "actual_exit_reason_breakdown": _exit_table(report.actual_exit_reason_breakdown),
        "shadow_exit_reason_breakdown": _exit_table(report.shadow_exit_reason_breakdown),
        "provenance": _frame(
            [
                {
                    **provenance.model_dump(
                        mode="python",
                        exclude={
                            "symbols",
                            "strategy_versions",
                            "ml_model_versions",
                            "evaluation_trading_dates",
                        },
                    ),
                    "symbols": json.dumps(provenance.symbols, separators=(",", ":")),
                    "strategy_versions": json.dumps(
                        provenance.strategy_versions, separators=(",", ":")
                    ),
                    "ml_model_versions": json.dumps(
                        provenance.ml_model_versions, separators=(",", ":")
                    ),
                    "evaluation_trading_dates": json.dumps(
                        [value.isoformat() for value in provenance.evaluation_trading_dates],
                        separators=(",", ":"),
                    ),
                }
            ],
            {
                "report_id": pl.String,
                "reporting_version": pl.String,
                "generated_at": pl.Datetime(time_unit="us", time_zone="Asia/Kolkata"),
                "source_backtest_fingerprint": pl.String,
                "run_id": pl.String,
                "git_commit": pl.String,
                "backtester_version": pl.String,
                "window_start": pl.Datetime(time_unit="us", time_zone="Asia/Kolkata"),
                "window_end": pl.Datetime(time_unit="us", time_zone="Asia/Kolkata"),
                "cost_policy_id": pl.String,
                "cost_policy_source_as_of_date": pl.Date,
                "brokerage_plan": pl.String,
                "starting_capital": DECIMAL,
                "ending_capital": DECIMAL,
                "capital_exhausted": pl.Boolean,
                "symbols": pl.String,
                "strategy_versions": pl.String,
                "ml_model_versions": pl.String,
                "evaluation_trading_dates": pl.String,
                "research_scope_id": pl.String,
                "plan_id": pl.String,
                "window_id": pl.String,
                "oos_result_fingerprint": pl.String,
            },
        ),
    }
    tables.update(_diagnostic_tables(report))
    return tables


def write_report_dataset(
    report: ReportBundle,
    output_directory: Path,
) -> tuple[Path, ...]:
    """Write the canonical Parquet dataset without overwriting report artifacts."""
    directory = Path(output_directory)
    existing = [
        directory / name
        for name in REPORT_TABLE_FILENAMES.values()
        if (directory / name).exists()
    ]
    if existing:
        raise FileExistsError(f"report dataset already contains artifact: {existing[0]}")
    directory.mkdir(parents=True, exist_ok=True)
    tables = report_tables(report)
    paths = []
    for name, filename in REPORT_TABLE_FILENAMES.items():
        path = directory / filename
        tables[name].write_parquet(path)
        paths.append(path)
    return tuple(paths)
