"""Deterministic sparse multi-objective Strategy Optimizer."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, time
from decimal import Decimal
from statistics import pstdev
from types import MappingProxyType
from zoneinfo import ZoneInfo

import optuna

from algo_trader.ml.contracts import StrategyParameterEvaluator
from algo_trader.ml.models import (
    CategoricalParameterSpec,
    FloatParameterSpec,
    IntParameterSpec,
    OptimizationTrialResult,
    OptimizationTrialState,
    ParameterSpec,
    StrategyOptimizationResult,
    StrategyOptimizerConfig,
)
from algo_trader.oos import OOSRegistry
from algo_trader.reporting import ReportBundle

STRATEGY_OPTIMIZER_VERSION = "2"
MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")
OBJECTIVE_DIRECTIONS = (
    "maximize",
    "maximize",
    "maximize",
    "maximize",
    "maximize",
    "minimize",
    "minimize",
    "minimize",
    "minimize",
)
ZERO = Decimal("0")


class OptimizationIntegrityError(ValueError):
    """Raised when a supplied strategy evaluation violates its contract."""


def parameter_distance(
    parameters: Mapping[str, int | float | str | bool],
    parameter_specs: tuple[ParameterSpec, ...],
) -> Decimal:
    """Return mean normalized movement across every eligible parameter."""
    if not parameter_specs:
        raise ValueError("parameter_specs must not be empty")
    distances = []
    for spec in parameter_specs:
        value = parameters[spec.name]
        if isinstance(spec, FloatParameterSpec | IntParameterSpec):
            distance = abs(Decimal(str(value)) - Decimal(str(spec.baseline_value))) / (
                Decimal(str(spec.high)) - Decimal(str(spec.low))
            )
        else:
            distance = ZERO if value == spec.baseline_value else Decimal("1")
        distances.append(distance)
    return sum(distances, start=ZERO) / len(distances)


def optimize_strategy_parameters(
    config: StrategyOptimizerConfig,
    evaluator: StrategyParameterEvaluator,
    oos_registry: OOSRegistry,
) -> StrategyOptimizationResult:
    """Return deterministic complete and Pareto trials without selecting a winner."""
    if not isinstance(config, StrategyOptimizerConfig):
        raise TypeError("config must be a StrategyOptimizerConfig")
    if not isinstance(oos_registry, OOSRegistry):
        raise TypeError("oos_registry must be an OOSRegistry")
    if not callable(getattr(evaluator, "evaluate", None)):
        raise TypeError("evaluator must implement StrategyParameterEvaluator")

    for selected in config.evaluation_ranges:
        oos_registry.assert_training_range_allowed(
            config.research_scope_id,
            config.plan_id,
            selected.start_date,
            selected.end_date,
        )

    sampler = optuna.samplers.TPESampler(seed=config.random_seed)
    study = optuna.create_study(
        directions=list(OBJECTIVE_DIRECTIONS),
        sampler=sampler,
        study_name=(
            f"strategy-optimizer-v{STRATEGY_OPTIMIZER_VERSION}:"
            f"{config.research_scope_id}:{config.plan_id}:{config.random_seed}"
        ),
    )
    baseline_flags = {f"change::{spec.name}": False for spec in config.parameter_specs}
    study.enqueue_trial(baseline_flags)
    completed: dict[int, OptimizationTrialResult] = {}

    def objective(trial: optuna.Trial) -> tuple[float, ...]:
        parameters, changed, distance = _prepare_trial_parameters(trial, config)
        reports = evaluator.evaluate(MappingProxyType(parameters.copy()), config.evaluation_ranges)
        validated = _validate_reports(config, reports)
        values = _objective_values(validated, distance, config.low_quality_threshold)
        completed[trial.number] = OptimizationTrialResult(
            trial_number=trial.number,
            state=OptimizationTrialState.COMPLETE,
            parameters=parameters,
            changed_parameter_names=changed,
            parameter_distance=distance,
            pf_score=values[0],
            average_net_return=values[1],
            average_cagr=values[2],
            average_win_rate=values[3],
            average_trades_per_day=values[4],
            worst_max_drawdown_pct=values[5],
            instability=values[6],
            low_quality_trade_fraction=values[7],
            evaluation_window_count=len(validated),
            source_report_ids=tuple(report.provenance.report_id for report in validated),
            source_run_ids=tuple(report.provenance.run_id for report in validated),
            source_backtest_fingerprints=tuple(
                report.provenance.source_backtest_fingerprint for report in validated
            ),
        )
        return tuple(float(value) for value in values)

    study.optimize(objective, n_trials=config.n_trials, n_jobs=1)
    completed_trials = tuple(completed[number] for number in sorted(completed))
    if 0 not in completed:
        raise OptimizationIntegrityError("the unchanged baseline trial must complete")
    pareto_numbers = sorted(trial.number for trial in study.best_trials)
    pareto_trials = tuple(completed[number] for number in pareto_numbers if number in completed)
    return StrategyOptimizationResult(
        optimizer_version=STRATEGY_OPTIMIZER_VERSION,
        research_scope_id=config.research_scope_id,
        plan_id=config.plan_id,
        strategy_id=config.strategy_id,
        random_seed=config.random_seed,
        max_changed_parameters=config.max_changed_parameters,
        low_quality_threshold=config.low_quality_threshold,
        optuna_version=optuna.__version__,
        objective_directions=OBJECTIVE_DIRECTIONS,
        completed_trials=completed_trials,
        pareto_trials=pareto_trials,
        baseline_trial=completed[0],
    )


def _prepare_trial_parameters(
    trial: optuna.Trial,
    config: StrategyOptimizerConfig,
) -> tuple[dict[str, int | float | str | bool], tuple[str, ...], Decimal]:
    parameters = dict(config.baseline_parameters)
    for spec in config.parameter_specs:
        change = trial.suggest_categorical(f"change::{spec.name}", [False, True])
        if change:
            parameters[spec.name] = _suggest_value(trial, spec)
    changed = tuple(
        spec.name
        for spec in config.parameter_specs
        if parameters[spec.name] != spec.baseline_value
    )
    if len(changed) > config.max_changed_parameters:
        raise optuna.TrialPruned("trial exceeds max_changed_parameters")
    return parameters, changed, parameter_distance(parameters, config.parameter_specs)


def _suggest_value(
    trial: optuna.Trial,
    spec: ParameterSpec,
) -> int | float | str | bool:
    if isinstance(spec, FloatParameterSpec):
        return trial.suggest_float(
            f"value::{spec.name}",
            spec.low,
            spec.high,
            step=spec.step,
            log=spec.log,
        )
    if isinstance(spec, IntParameterSpec):
        return trial.suggest_int(
            f"value::{spec.name}",
            spec.low,
            spec.high,
            step=spec.step,
            log=spec.log,
        )
    if isinstance(spec, CategoricalParameterSpec):
        return trial.suggest_categorical(f"value::{spec.name}", list(spec.choices))
    raise TypeError("unsupported ParameterSpec")


def _validate_reports(
    config: StrategyOptimizerConfig,
    reports: object,
) -> tuple[ReportBundle, ...]:
    if not isinstance(reports, tuple) or any(
        not isinstance(item, ReportBundle) for item in reports
    ):
        raise OptimizationIntegrityError("evaluator must return a tuple of ReportBundle values")
    if len(reports) != len(config.evaluation_ranges):
        raise OptimizationIntegrityError("evaluator must return exactly one report per range")
    for report, selected in zip(reports, config.evaluation_ranges, strict=True):
        expected_start = datetime.combine(selected.start_date, time.min, MARKET_TIMEZONE)
        expected_end = datetime.combine(selected.end_date, time.min, MARKET_TIMEZONE)
        if (
            report.provenance.window_start != expected_start
            or report.provenance.window_end != expected_end
        ):
            raise OptimizationIntegrityError("report window must exactly match evaluation range")
        if report.provenance.research_scope_id not in (None, config.research_scope_id):
            raise OptimizationIntegrityError("report research scope does not match optimizer")
        if report.provenance.plan_id not in (None, config.plan_id):
            raise OptimizationIntegrityError("report plan does not match optimizer")
        if any(
            strategy_id != config.strategy_id
            for strategy_id, _ in report.provenance.strategy_versions
        ):
            raise OptimizationIntegrityError("report provenance contains an unrelated strategy")
        if any(
            record.trade.signal.strategy_id != config.strategy_id
            for record in report.actual_trade_records
        ):
            raise OptimizationIntegrityError("report contains an unrelated actual strategy")
    return reports


def _objective_values(
    reports: tuple[ReportBundle, ...],
    distance: Decimal,
    low_quality_threshold: float,
) -> tuple[Decimal, ...]:
    pf_scores = []
    average_returns = []
    cagrs = []
    win_rates = []
    frequencies = []
    drawdowns = []
    actual_records = []
    for report in reports:
        performance = report.performance
        factor = performance.net_profit_factor
        if factor.is_undefined:
            raise optuna.TrialPruned("profit factor is undefined")
        if factor.is_unbounded:
            pf_scores.append(Decimal("1"))
        elif factor.value is not None:
            pf_scores.append(factor.value / (Decimal("1") + factor.value))
        if performance.average_net_return_per_trade is None:
            raise optuna.TrialPruned("average net return is undefined")
        if performance.cagr is None:
            raise optuna.TrialPruned("CAGR is undefined")
        if performance.win_rate is None:
            raise optuna.TrialPruned("win rate is undefined")
        average_returns.append(performance.average_net_return_per_trade)
        cagrs.append(performance.cagr)
        win_rates.append(performance.win_rate)
        frequencies.append(performance.actual_trades_per_day)
        drawdowns.append(performance.maximum_realized_drawdown_pct)
        actual_records.extend(report.actual_trade_records)
    low_quality = sum(
        record.trade.ml_score.quality_score < low_quality_threshold
        for record in actual_records
    )
    if not actual_records:
        raise optuna.TrialPruned("low-quality fraction requires completed actual trades")
    instability = Decimal(pstdev(average_returns)) if len(average_returns) > 1 else ZERO
    return (
        _mean(tuple(pf_scores)),
        _mean(tuple(average_returns)),
        _mean(tuple(cagrs)),
        _mean(tuple(win_rates)),
        _mean(tuple(frequencies)),
        max(drawdowns),
        instability,
        Decimal(low_quality) / len(actual_records),
        distance,
    )


def _mean(values: tuple[Decimal, ...]) -> Decimal:
    return sum(values, start=ZERO) / len(values)
