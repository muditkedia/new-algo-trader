"""Read-only evaluation of scores embedded in completed backtest trades."""

from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal

from algo_trader.backtest import BacktestRequestOutcome, BacktestRunResult, BacktestTradeRecord
from algo_trader.ml.models import (
    MetaModelEvaluation,
    QualityBucketMetrics,
    SizeBucketMetrics,
)
from algo_trader.oos import OOSTestRecord, fingerprint_backtest_result

ZERO = Decimal("0")
QUALITY_BOUNDARIES = (
    (Decimal("0.0"), Decimal("0.2"), False),
    (Decimal("0.2"), Decimal("0.4"), False),
    (Decimal("0.4"), Decimal("0.6"), False),
    (Decimal("0.6"), Decimal("0.8"), False),
    (Decimal("0.8"), Decimal("1.0"), True),
)


class MetaModelEvaluationIntegrityError(ValueError):
    """Raised when optional registered-OOS provenance does not match."""


def evaluate_trade_meta_model(
    backtest_result: BacktestRunResult,
    model_version: str,
    oos_test_record: OOSTestRecord | None = None,
) -> MetaModelEvaluation:
    """Evaluate embedded model scores over completed actual and shadow labels."""
    if not isinstance(backtest_result, BacktestRunResult):
        raise TypeError("backtest_result must be a BacktestRunResult")
    if not isinstance(model_version, str) or not model_version.strip():
        raise ValueError("model_version must be a non-empty string")
    fingerprint = fingerprint_backtest_result(backtest_result)
    oos_values = _verify_oos(backtest_result, oos_test_record, fingerprint)
    actual = tuple(
        record
        for record in backtest_result.actual_trade_records
        if record.trade.ml_score.model_version == model_version
    )
    shadow = tuple(
        record
        for record in backtest_result.shadow_trade_records
        if record.trade.ml_score.model_version == model_version
    )
    records = tuple(sorted((*actual, *shadow), key=_record_key))
    labels = tuple(int(record.trade.net_pnl > 0) for record in records)
    probabilities = tuple(
        Decimal(str(record.trade.ml_score.calibrated_probability)) for record in records
    )
    predictions = tuple(
        Decimal(str(record.trade.ml_score.predicted_net_return)) for record in records
    )
    realized = tuple(record.trade.net_return for record in records)
    count = len(records)
    outcome_counts = Counter(
        request.outcome
        for request in backtest_result.request_results
        if request.request.candidate.ml_score.model_version == model_version
    )
    quality_buckets = _quality_buckets(records)
    size_buckets = _size_buckets(records)
    quality_returns = tuple(
        bucket.average_realized_net_return
        for bucket in quality_buckets
        if bucket.count > 0 and bucket.average_realized_net_return is not None
    )
    quality_wins = tuple(
        bucket.empirical_win_rate
        for bucket in quality_buckets
        if bucket.count > 0 and bucket.empirical_win_rate is not None
    )
    size_returns = tuple(bucket.average_realized_net_return for bucket in size_buckets)
    return MetaModelEvaluation(
        model_version=model_version,
        source_backtest_fingerprint=fingerprint,
        labeled_count=count,
        actual_labeled_count=len(actual),
        shadow_labeled_count=len(shadow),
        profitable_count=sum(labels),
        empirical_win_rate=_mean(tuple(Decimal(value) for value in labels)),
        brier_score=_mean(
            tuple(
                (probability - label) ** 2
                for probability, label in zip(probabilities, labels, strict=True)
            )
        ),
        predicted_net_return_mae=_mean(
            tuple(
                abs(predicted - actual_return)
                for predicted, actual_return in zip(predictions, realized, strict=True)
            )
        ),
        mean_predicted_net_return=_mean(predictions),
        mean_realized_net_return=_mean(realized),
        allocated_entry_not_filled_count=outcome_counts[
            BacktestRequestOutcome.ALLOCATED_ENTRY_NOT_FILLED
        ],
        shadow_entry_not_filled_count=outcome_counts[
            BacktestRequestOutcome.SHADOW_ENTRY_NOT_FILLED
        ],
        capital_exhausted_count=outcome_counts[BacktestRequestOutcome.CAPITAL_EXHAUSTED],
        quality_buckets=quality_buckets,
        size_buckets=size_buckets,
        quality_return_monotonic=_non_decreasing(quality_returns),
        quality_win_rate_monotonic=_non_decreasing(quality_wins),
        size_return_monotonic=_non_decreasing(size_returns),
        **oos_values,
    )


def _quality_buckets(
    records: tuple[BacktestTradeRecord, ...],
) -> tuple[QualityBucketMetrics, ...]:
    grouped: dict[int, list[BacktestTradeRecord]] = defaultdict(list)
    for record in records:
        quality = Decimal(str(record.trade.ml_score.quality_score))
        index = min(int(quality / Decimal("0.2")), 4)
        grouped[index].append(record)
    output = []
    for index, (lower, upper, inclusive) in enumerate(QUALITY_BOUNDARIES):
        values = tuple(grouped[index])
        labels = tuple(Decimal(record.trade.net_pnl > 0) for record in values)
        output.append(
            QualityBucketMetrics(
                lower_bound=lower,
                upper_bound=upper,
                upper_inclusive=inclusive,
                count=len(values),
                actual_count=sum(not record.trade.is_shadow for record in values),
                shadow_count=sum(record.trade.is_shadow for record in values),
                average_quality_score=_mean(
                    tuple(Decimal(str(record.trade.ml_score.quality_score)) for record in values)
                ),
                average_calibrated_probability=_mean(
                    tuple(
                        Decimal(str(record.trade.ml_score.calibrated_probability))
                        for record in values
                    )
                ),
                empirical_win_rate=_mean(labels),
                average_predicted_net_return=_mean(
                    tuple(
                        Decimal(str(record.trade.ml_score.predicted_net_return))
                        for record in values
                    )
                ),
                average_realized_net_return=_mean(
                    tuple(record.trade.net_return for record in values)
                ),
            )
        )
    return tuple(output)


def _size_buckets(
    records: tuple[BacktestTradeRecord, ...],
) -> tuple[SizeBucketMetrics, ...]:
    grouped: dict[int, list[BacktestTradeRecord]] = defaultdict(list)
    for record in records:
        grouped[record.trade.ml_score.recommended_notional].append(record)
    output = []
    for notional, values in sorted(grouped.items()):
        group = tuple(values)
        output.append(
            SizeBucketMetrics(
                recommended_notional=notional,
                count=len(group),
                average_quality_score=_required_mean(
                    tuple(Decimal(str(record.trade.ml_score.quality_score)) for record in group)
                ),
                empirical_win_rate=_required_mean(
                    tuple(Decimal(record.trade.net_pnl > 0) for record in group)
                ),
                average_realized_net_return=_required_mean(
                    tuple(record.trade.net_return for record in group)
                ),
            )
        )
    return tuple(output)


def _verify_oos(
    result: BacktestRunResult,
    record: OOSTestRecord | None,
    fingerprint: str,
) -> dict[str, str | None]:
    if record is None:
        return {"research_scope_id": None, "plan_id": None, "window_id": None}
    if not isinstance(record, OOSTestRecord):
        raise TypeError("oos_test_record must be an OOSTestRecord or None")
    comparisons = (
        (record.backtest_run_id, result.run_id),
        (record.backtest_git_commit, result.git_commit),
        (record.backtester_version, result.backtester_version),
        (record.backtest_window_start, result.window_start),
        (record.backtest_window_end, result.window_end),
        (record.cost_policy_id, result.cost_policy_id),
        (record.brokerage_plan, result.brokerage_plan.value),
        (record.symbols, result.symbols),
        (record.strategy_versions, result.strategy_versions),
        (record.ml_model_versions, result.ml_model_versions),
    )
    if any(left != right for left, right in comparisons):
        raise MetaModelEvaluationIntegrityError("OOS provenance does not match backtest result")
    if record.result_fingerprint != fingerprint:
        raise MetaModelEvaluationIntegrityError("OOS result fingerprint does not match")
    return {
        "research_scope_id": record.research_scope_id,
        "plan_id": record.plan_id,
        "window_id": record.window_id,
    }


def _record_key(record: BacktestTradeRecord) -> tuple[object, ...]:
    return record.trade.signal.timestamp, str(record.allocation_identity), record.trade.is_shadow


def _mean(values: tuple[Decimal, ...]) -> Decimal | None:
    return sum(values, start=ZERO) / len(values) if values else None


def _required_mean(values: tuple[Decimal, ...]) -> Decimal:
    result = _mean(values)
    if result is None:
        raise RuntimeError("non-empty group unexpectedly produced no mean")
    return result


def _non_decreasing(values: tuple[Decimal, ...]) -> bool | None:
    if len(values) < 2:
        return None
    return all(left <= right for left, right in zip(values, values[1:], strict=False))
