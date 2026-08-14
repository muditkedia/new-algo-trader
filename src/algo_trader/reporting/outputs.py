"""Derivative Excel and Matplotlib presentation artifacts for reports."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from algo_trader.reporting.models import ProfitFactor, ReportBundle
from algo_trader.reporting.tables import report_tables

MONEY_FORMAT = "₹#,##0.00;[Red]-₹#,##0.00"
PERCENT_FORMAT = "0.00%"
HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
HEADER_FONT = Font(color="FFFFFF", bold=True)
PASS_FILL = PatternFill("solid", fgColor="C6EFCE")
FAIL_FILL = PatternFill("solid", fgColor="FFC7CE")


def _excel_value(value: object) -> object:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value
    if isinstance(value, (tuple, list, dict, set, frozenset)):
        return str(value)
    return value


def _write_table(sheet: Worksheet, rows: list[dict[str, Any]]) -> None:
    if not rows:
        sheet["A1"] = "No records"
        sheet.freeze_panes = "A2"
        return
    headers = list(rows[0])
    sheet.append(headers)
    for cell in sheet[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
    for row in rows:
        sheet.append([_excel_value(row[header]) for header in headers])
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for index, header in enumerate(headers, start=1):
        values = [str(row.get(header, "")) for row in rows[:100]]
        width = min(max([len(header), *(len(value) for value in values)]) + 2, 35)
        sheet.column_dimensions[get_column_letter(index)].width = width
        lowered = header.lower()
        money_tokens = (
            "capital",
            "pnl",
            "cost",
            "turnover",
            "price",
            "margin",
            "notional",
            "amount",
        )
        if any(token in lowered for token in money_tokens):
            for cell in sheet[get_column_letter(index)][1:]:
                if isinstance(cell.value, int | float):
                    cell.number_format = MONEY_FORMAT
        elif any(token in lowered for token in ("return", "rate", "pct", "percentage")):
            for cell in sheet[get_column_letter(index)][1:]:
                if isinstance(cell.value, int | float):
                    cell.number_format = PERCENT_FORMAT


def _pf_display(value: ProfitFactor) -> Decimal | str:
    if value.is_unbounded:
        return "UNBOUNDED"
    if value.is_undefined:
        return "UNDEFINED"
    return value.value if value.value is not None else "UNDEFINED"


def _dashboard(workbook: Workbook, report: ReportBundle) -> Worksheet:
    sheet = workbook.active
    sheet.title = "Dashboard"
    sheet.sheet_view.showGridLines = False
    sheet["A1"] = "NEW ALGO TRADER — BACKTEST REPORT"
    sheet["A1"].font = Font(size=16, bold=True, color="1F4E78")
    sheet["A2"] = "ACTUAL REALIZED PORTFOLIO PERFORMANCE"
    sheet["A2"].font = Font(bold=True)
    performance = report.performance
    metrics = [
        ("Starting Capital", performance.starting_capital, MONEY_FORMAT),
        ("Ending Capital", performance.ending_capital, MONEY_FORMAT),
        ("Net Profit", performance.net_profit, MONEY_FORMAT),
        ("Total Return", performance.total_return, PERCENT_FORMAT),
        ("CAGR", performance.cagr, PERCENT_FORMAT),
        ("Actual Trades", performance.actual_trade_count, "0"),
        ("Win Rate", performance.win_rate, PERCENT_FORMAT),
        ("Net Profit Factor", _pf_display(performance.net_profit_factor), "0.00"),
        ("Avg Net Return/Trade", performance.average_net_return_per_trade, PERCENT_FORMAT),
        ("Max Realized Drawdown %", performance.maximum_realized_drawdown_pct, PERCENT_FORMAT),
        ("Trades/Day", performance.actual_trades_per_day, "0.00"),
        ("Total Costs", performance.total_costs, MONEY_FORMAT),
        ("Capital Exhausted", report.provenance.capital_exhausted, "General"),
    ]
    for row_index, (label, value, number_format) in enumerate(metrics, start=4):
        sheet.cell(row_index, 1, label).font = Font(bold=True)
        cell = sheet.cell(row_index, 2, _excel_value(value))
        cell.number_format = number_format

    sheet["D2"] = "HARD QUANTITATIVE TARGETS"
    sheet["D2"].font = Font(bold=True)
    statuses = [
        ("CAGR >20%", report.acceptance.cagr_pass),
        ("Win Rate >50%", report.acceptance.win_rate_pass),
        ("PF >2", report.acceptance.profit_factor_pass),
        ("Avg Net Return >0.5%", report.acceptance.average_net_return_pass),
        ("All Hard Targets", report.acceptance.hard_quantitative_targets_pass),
        ("FREQUENCY IDEAL >=2/day", report.acceptance.frequency_target_met),
    ]
    for row_index, (label, passed) in enumerate(statuses, start=4):
        sheet.cell(row_index, 4, label).font = Font(bold=True)
        sheet.cell(row_index, 5, "PASS" if passed else "FAIL")
    sheet.conditional_formatting.add(
        "E4:E9", CellIsRule(operator="equal", formula=['"PASS"'], fill=PASS_FILL)
    )
    sheet.conditional_formatting.add(
        "E4:E9", CellIsRule(operator="equal", formula=['"FAIL"'], fill=FAIL_FILL)
    )

    sheet["G2"] = "REPRODUCIBILITY"
    sheet["G2"].font = Font(bold=True)
    provenance = [
        ("report_id", report.provenance.report_id),
        ("run_id", report.provenance.run_id),
        ("git_commit", report.provenance.git_commit),
        ("reporting_version", report.provenance.reporting_version),
        ("backtester_version", report.provenance.backtester_version),
        ("cost_policy_id", report.provenance.cost_policy_id),
    ]
    if report.provenance.research_scope_id is not None:
        provenance.extend(
            [
                ("research_scope_id", report.provenance.research_scope_id),
                ("plan_id", report.provenance.plan_id),
                ("window_id", report.provenance.window_id),
            ]
        )
    for row_index, (label, value) in enumerate(provenance, start=4):
        sheet.cell(row_index, 7, label).font = Font(bold=True)
        sheet.cell(row_index, 8, value)
    for column, width in {"A": 28, "B": 18, "D": 28, "E": 12, "G": 22, "H": 28}.items():
        sheet.column_dimensions[column].width = width
    sheet.freeze_panes = "A4"
    return sheet


def _chart_data_sheet(
    workbook: Workbook,
    report: ReportBundle,
) -> tuple[Worksheet, dict[str, tuple[int, int, int]]]:
    sheet = workbook.create_sheet("Chart Data")
    positions: dict[str, tuple[int, int, int]] = {}
    row = 1

    def add(name: str, headers: tuple[str, str], values: list[tuple[object, object]]) -> None:
        nonlocal row
        start = row
        sheet.cell(row, 1, headers[0])
        sheet.cell(row, 2, headers[1])
        for first, second in values:
            row += 1
            sheet.cell(row, 1, _excel_value(first))
            sheet.cell(row, 2, _excel_value(second))
        positions[name] = (start, start + len(values), len(values))
        row += 2

    add(
        "equity",
        ("Exit Timestamp", "Realized Capital"),
        [(point.timestamp, point.realized_capital) for point in report.equity_curve],
    )
    add(
        "drawdown",
        ("Exit Timestamp", "Realized Drawdown"),
        [(point.timestamp, point.drawdown_pct) for point in report.equity_curve],
    )
    add(
        "daily",
        ("Trading Date", "Daily Net P&L"),
        [(point.trading_date, point.net_pnl) for point in report.daily_performance],
    )
    request = report.request_outcomes
    add(
        "requests",
        ("Request Outcome", "Count"),
        [
            ("Completed Actual", request.completed_actual),
            ("Completed Shadow", request.completed_shadow),
            ("Allocated No Fill", request.allocated_entry_not_filled),
            ("Shadow No Fill", request.shadow_entry_not_filled),
            ("Capital Exhausted", request.capital_exhausted),
        ],
    )
    add(
        "costs",
        ("Actual Cost Component", "Amount"),
        [
            (name, getattr(report.actual_costs, name))
            for name in report.actual_costs.component_percentages
        ],
    )
    add(
        "strategies",
        ("Strategy / Version", "Actual Net P&L"),
        [
            (f"{value.strategy_id} / {value.strategy_version}", value.net_pnl)
            for value in report.actual_strategy_breakdown
        ],
    )
    add(
        "exits",
        ("Exit Reason", "Actual Trade Count"),
        [
            (value.exit_reason.value, value.trade_count)
            for value in report.actual_exit_reason_breakdown
        ],
    )
    add(
        "symbols",
        ("Symbol", "Actual Net P&L"),
        [(value.symbol, value.net_pnl) for value in report.symbol_breakdown],
    )
    sheet.sheet_state = "hidden"
    return sheet, positions


def _add_chart(
    dashboard: Worksheet,
    source: Worksheet,
    bounds: tuple[int, int, int],
    title: str,
    anchor: str,
    *,
    line: bool,
    percent: bool = False,
) -> None:
    start, end, count = bounds
    if count == 0:
        return
    chart = LineChart() if line else BarChart()
    chart.title = title
    chart.height = 7
    chart.width = 12
    chart.y_axis.title = "Percent" if percent else "Value"
    chart.x_axis.title = source.cell(start, 1).value
    chart.add_data(Reference(source, min_col=2, min_row=start, max_row=end), titles_from_data=True)
    chart.set_categories(Reference(source, min_col=1, min_row=start + 1, max_row=end))
    if not line:
        chart.type = "col"
    dashboard.add_chart(chart, anchor)


def write_excel_report(report: ReportBundle, output_path: Path) -> Path:
    """Write a professional derivative workbook from canonical report tables."""
    if not isinstance(report, ReportBundle):
        raise TypeError("report must be a ReportBundle")
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(f"Excel report already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tables = report_tables(report)
    workbook = Workbook()
    dashboard = _dashboard(workbook, report)
    sheet_tables = (
        ("Summary", "summary"),
        ("Actual Trades", "actual_trades"),
        ("Shadow Trades", "shadow_trades"),
        ("Daily Performance", "daily_performance"),
        ("Requests", "request_outcomes"),
        ("Equity Curve", "equity_curve"),
        ("Strategy Breakdown", "actual_strategy_breakdown"),
        ("Shadow Strategy", "shadow_strategy_breakdown"),
        ("Symbol Breakdown", "symbol_breakdown"),
        ("Costs", "actual_cost_breakdown"),
        ("Shadow Costs", "shadow_cost_breakdown"),
        ("Exit Reasons", "actual_exit_reason_breakdown"),
        ("Shadow Exit Reasons", "shadow_exit_reason_breakdown"),
        ("Provenance", "provenance"),
    )
    for sheet_name, table_name in sheet_tables:
        _write_table(workbook.create_sheet(sheet_name), tables[table_name].to_dicts())
    chart_data, positions = _chart_data_sheet(workbook, report)
    _add_chart(
        dashboard,
        chart_data,
        positions["equity"],
        "Realized Portfolio Equity",
        "A20",
        line=True,
    )
    _add_chart(
        dashboard,
        chart_data,
        positions["drawdown"],
        "Realized Capital Drawdown",
        "N20",
        line=True,
        percent=True,
    )
    _add_chart(dashboard, chart_data, positions["daily"], "Daily Actual Net P&L", "A35", line=False)
    _add_chart(
        dashboard,
        chart_data,
        positions["requests"],
        "Request Outcome Funnel",
        "N35",
        line=False,
    )
    if report.actual_costs.total_costs > 0:
        _add_chart(
            dashboard,
            chart_data,
            positions["costs"],
            "Actual Transaction Cost Composition",
            "A50",
            line=False,
        )
    _add_chart(
        dashboard,
        chart_data,
        positions["strategies"],
        "Actual Strategy Net P&L",
        "N50",
        line=False,
    )
    _add_chart(
        workbook["Exit Reasons"],
        chart_data,
        positions["exits"],
        "Actual Exit Reason Counts",
        "H2",
        line=False,
    )
    if len(report.symbol_breakdown) <= 20:
        _add_chart(
            workbook["Symbol Breakdown"],
            chart_data,
            positions["symbols"],
            "Actual Symbol Net P&L",
            "Q2",
            line=False,
        )
    workbook.save(path)
    return path


def _save_plot(path: Path, title: str, draw: Any) -> None:
    if path.exists():
        raise FileExistsError(f"visual report path already exists: {path}")
    figure, axis = plt.subplots(figsize=(10, 5), dpi=120)
    draw(axis)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _empty(axis: Any, message: str = "No completed actual trades") -> None:
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_xticks([])
    axis.set_yticks([])


def write_visual_report(report: ReportBundle, output_directory: Path) -> tuple[Path, ...]:
    """Write deterministic headless PNG views from the same canonical tables."""
    if not isinstance(report, ReportBundle):
        raise TypeError("report must be a ReportBundle")
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    filenames = [
        "realized_equity_curve.png",
        "realized_drawdown.png",
        "daily_net_pnl.png",
        "trade_net_returns.png",
        "strategy_net_pnl.png",
        "cost_composition.png",
        "request_outcomes.png",
        "exit_reason_counts.png",
    ]
    if report.shadow_trade_records:
        filenames.append("shadow_trade_net_returns.png")
    paths_to_write = tuple(directory / filename for filename in filenames)
    existing = tuple(path for path in paths_to_write if path.exists())
    if existing:
        raise FileExistsError(f"visual report path already exists: {existing[0]}")
    tables = report_tables(report)
    paths: list[Path] = []

    def save(filename: str, title: str, draw: Any) -> None:
        path = directory / filename
        _save_plot(path, title, draw)
        paths.append(path)

    equity = tables["equity_curve"].to_dicts()
    save(
        "realized_equity_curve.png",
        "Realized Portfolio Equity (Actual Only)",
        lambda axis: (
            axis.plot(
                [row["timestamp"] for row in equity],
                [float(row["realized_capital"]) for row in equity],
            ),
            axis.set_xlabel("Exit timestamp"),
            axis.set_ylabel("Realized capital (INR)"),
            axis.grid(alpha=0.25),
        ),
    )
    save(
        "realized_drawdown.png",
        "Realized Capital Drawdown (Actual Only)",
        lambda axis: (
            axis.plot(
                [row["timestamp"] for row in equity],
                [float(row["drawdown_pct"]) * 100 for row in equity],
            ),
            axis.set_xlabel("Exit timestamp"),
            axis.set_ylabel("Drawdown (%)"),
            axis.grid(alpha=0.25),
        ),
    )
    daily = tables["daily_performance"].to_dicts()
    save(
        "daily_net_pnl.png",
        "Daily Actual Net P&L",
        lambda axis: (
            axis.bar(
                [row["trading_date"].isoformat() for row in daily],
                [float(row["net_pnl"]) for row in daily],
            ),
            axis.set_xlabel("Evaluation trading date"),
            axis.set_ylabel("Net P&L (INR)"),
            axis.tick_params(axis="x", rotation=45),
            axis.axhline(0, color="black", linewidth=0.8),
        ),
    )
    trades = tables["actual_trades"].to_dicts()

    def trade_returns(axis: Any) -> None:
        if not trades:
            _empty(axis)
            return
        axis.plot(
            range(1, len(trades) + 1),
            [float(row["net_return"]) * 100 for row in trades],
            marker="o",
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xlabel("Chronological actual trade number")
        axis.set_ylabel("Net return (%)")

    save("trade_net_returns.png", "Actual Trade Net Returns", trade_returns)
    strategies = tables["actual_strategy_breakdown"].to_dicts()

    def strategy_pnl(axis: Any) -> None:
        if not strategies:
            _empty(axis)
            return
        axis.bar(
            [f"{row['strategy_id']} / {row['strategy_version']}" for row in strategies],
            [float(row["net_pnl"]) for row in strategies],
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_ylabel("Actual net P&L (INR)")
        axis.tick_params(axis="x", rotation=30)

    save("strategy_net_pnl.png", "Actual Strategy Net P&L", strategy_pnl)
    costs = [row for row in tables["actual_cost_breakdown"].to_dicts() if row["is_cost_component"]]

    def cost_composition(axis: Any) -> None:
        if not costs or all(row["amount"] == 0 for row in costs):
            _empty(axis, "No actual transaction costs")
            return
        axis.bar([row["component"] for row in costs], [float(row["amount"]) for row in costs])
        axis.set_ylabel("Actual cost (INR)")
        axis.tick_params(axis="x", rotation=30)

    save("cost_composition.png", "Actual Transaction Cost Composition", cost_composition)
    request = report.request_outcomes
    request_values = [
        ("Completed Actual", request.completed_actual),
        ("Completed Shadow", request.completed_shadow),
        ("Allocated No Fill", request.allocated_entry_not_filled),
        ("Shadow No Fill", request.shadow_entry_not_filled),
        ("Capital Exhausted", request.capital_exhausted),
    ]
    save(
        "request_outcomes.png",
        "Request Outcome Funnel",
        lambda axis: (
            axis.bar([name for name, _ in request_values], [value for _, value in request_values]),
            axis.set_ylabel("Request count"),
            axis.tick_params(axis="x", rotation=30),
        ),
    )
    exits = tables["actual_exit_reason_breakdown"].to_dicts()
    save(
        "exit_reason_counts.png",
        "Actual Exit Reason Counts",
        lambda axis: (
            axis.bar([row["exit_reason"] for row in exits], [row["trade_count"] for row in exits]),
            axis.set_ylabel("Actual trade count"),
            axis.tick_params(axis="x", rotation=30),
        ),
    )
    if report.shadow_trade_records:
        shadow = tables["shadow_trades"].to_dicts()
        save(
            "shadow_trade_net_returns.png",
            "SHADOW / HYPOTHETICAL — NO ACTUAL CAPITAL IMPACT",
            lambda axis: (
                axis.plot(
                    range(1, len(shadow) + 1),
                    [float(row["net_return"]) * 100 for row in shadow],
                    marker="o",
                ),
                axis.axhline(0, color="black", linewidth=0.8),
                axis.set_xlabel("Chronological shadow trade number"),
                axis.set_ylabel("Hypothetical net return (%)"),
            ),
        )
    return tuple(paths)
