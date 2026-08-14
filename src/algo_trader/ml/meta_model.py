"""Scope-specific cumulative Trade Meta-Model training and scoring."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import ROUND_HALF_UP, Decimal
from typing import Any
from zoneinfo import ZoneInfo

import lightgbm as lgb
import sklearn
from sklearn.linear_model import LogisticRegression

from algo_trader.domain import (
    MAX_NOTIONAL,
    MIN_NOTIONAL,
    NOTIONAL_INCREMENT,
    MLScore,
    Signal,
    SignalStatus,
)
from algo_trader.ml.features import extract_meta_features
from algo_trader.ml.models import (
    MetaFeatureSchema,
    MetaModelMetadata,
    MetaModelTrainingConfig,
    MetaTrainingSample,
    MetaTrainingSource,
    TrainingSourceProvenance,
)
from algo_trader.oos import OOSRegistry, fingerprint_backtest_result

ML_ARCHITECTURE_VERSION = "2"
SIZING_POLICY_ID = "quality-linear-half-up-v1"
MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")


class InsufficientTrainingDataError(ValueError):
    """Raised when chronological classifier calibration is not defensible."""


class DuplicateTrainingObservationError(ValueError):
    """Raised when one candidate would be weighted more than once."""


def recommended_notional_for_quality(quality_score: int | float | Decimal) -> int:
    """Map quality monotonically to one domain notional bucket using HALF_UP."""
    if isinstance(quality_score, bool) or not isinstance(quality_score, int | float | Decimal):
        raise TypeError("quality_score must be a numeric scalar")
    quality = Decimal(str(quality_score))
    if not quality.is_finite() or not Decimal("0") <= quality <= Decimal("1"):
        raise ValueError("quality_score must be finite and inside 0..1")
    bucket_count = (MAX_NOTIONAL - MIN_NOTIONAL) // NOTIONAL_INCREMENT
    index = int((quality * bucket_count).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    index = min(max(index, 0), bucket_count)
    return MIN_NOTIONAL + index * NOTIONAL_INCREMENT


class BootstrapTradeScorer:
    """Neutral untrained mode of the Trade Meta-Model system."""

    def __init__(self, model_version: str, recommended_notional: int = MIN_NOTIONAL) -> None:
        self._score = MLScore(
            model_version=model_version,
            quality_score=0.5,
            calibrated_probability=0.5,
            predicted_net_return=0.0,
            recommended_notional=recommended_notional,
        )

    def score(self, signal: Signal) -> MLScore:
        """Return a neutral advisory score without inspecting features or outcomes."""
        _validate_generated_signal(signal)
        return self._score


@dataclass(frozen=True)
class TrainedTradeMetaModel:
    """Immutable runtime scorer backed by native LightGBM boosters."""

    metadata: MetaModelMetadata
    classifier: lgb.Booster
    regressor: lgb.Booster

    def score(self, signal: Signal) -> MLScore:
        """Score one generated in-lineage signal using signal-time features only."""
        _validate_generated_signal(signal)
        if signal.strategy_id not in self.metadata.allowed_strategy_ids:
            raise ValueError("signal strategy_id is outside this model's allowed lineage")
        schema = MetaFeatureSchema(feature_names=self.metadata.feature_names)
        features = [list(extract_meta_features(signal, schema))]
        raw_score = float(self.classifier.predict(features, raw_score=True)[0])
        calibrated = _sigmoid(
            self.metadata.calibration_coefficient * raw_score
            + self.metadata.calibration_intercept
        )
        predicted_return = float(self.regressor.predict(features)[0])
        if not math.isfinite(predicted_return):
            raise RuntimeError("Trade Meta-Model produced a non-finite net-return prediction")
        return MLScore(
            model_version=self.metadata.model_version,
            quality_score=calibrated,
            calibrated_probability=calibrated,
            predicted_net_return=predicted_return,
            recommended_notional=recommended_notional_for_quality(calibrated),
        )


def extract_meta_training_samples(
    config: MetaModelTrainingConfig,
    sources: tuple[MetaTrainingSource, ...],
    oos_registry: OOSRegistry,
) -> tuple[MetaTrainingSample, ...]:
    """Guard every source first, then extract deterministic completed observations."""
    if not isinstance(config, MetaModelTrainingConfig):
        raise TypeError("config must be a MetaModelTrainingConfig")
    if not sources:
        raise ValueError("at least one MetaTrainingSource is required")
    if not isinstance(oos_registry, OOSRegistry):
        raise TypeError("oos_registry must be an OOSRegistry")

    for source in sources:
        if not isinstance(source, MetaTrainingSource):
            raise TypeError("all sources must be MetaTrainingSource instances")
        if (
            source.research_scope_id != config.research_scope_id
            or source.plan_id != config.plan_id
        ):
            raise ValueError("all training sources must match the target scope and plan")
        if not set(source.strategy_ids) <= set(config.allowed_strategy_ids):
            raise ValueError("source strategy_ids must be inside allowed_strategy_ids")
        start_date, end_date = _full_day_range(source.backtest_result)
        oos_registry.assert_training_range_allowed(
            config.research_scope_id,
            config.plan_id,
            start_date,
            end_date,
        )

    samples: list[MetaTrainingSample] = []
    identities: set[tuple[object, ...]] = set()
    for source in sources:
        records = (
            *source.backtest_result.actual_trade_records,
            *source.backtest_result.shadow_trade_records,
        )
        for record in records:
            trade = record.trade
            if trade.signal.strategy_id not in source.strategy_ids:
                continue
            identity = tuple(record.allocation_identity)
            if identity in identities:
                raise DuplicateTrainingObservationError(
                    "duplicate labeled candidate identity across training sources"
                )
            identities.add(identity)
            samples.append(
                MetaTrainingSample(
                    candidate_identity=identity,
                    signal_timestamp=trade.signal.timestamp,
                    feature_values=extract_meta_features(trade.signal, config.feature_schema),
                    profitable=int(trade.net_pnl > 0),
                    net_return=trade.net_return,
                    is_shadow=trade.is_shadow,
                )
            )
    return tuple(sorted(samples, key=_sample_sort_key))


def chronological_calibration_split(
    samples: tuple[MetaTrainingSample, ...],
    calibration_fraction: float,
) -> tuple[tuple[MetaTrainingSample, ...], tuple[MetaTrainingSample, ...]]:
    """Split at the nearest timestamp-group boundary without shuffling."""
    if not samples:
        raise InsufficientTrainingDataError("training samples must not be empty")
    ordered = tuple(sorted(samples, key=_sample_sort_key))
    desired_boundary = len(ordered) * (1.0 - calibration_fraction)
    boundaries = [
        index
        for index in range(1, len(ordered))
        if ordered[index - 1].signal_timestamp != ordered[index].signal_timestamp
    ]
    if not boundaries:
        raise InsufficientTrainingDataError(
            "fit and calibration require at least two signal timestamp groups"
        )
    boundary = min(boundaries, key=lambda value: (abs(value - desired_boundary), value))
    fit = ordered[:boundary]
    calibration = ordered[boundary:]
    if not fit or not calibration:
        raise InsufficientTrainingDataError("fit and calibration sets must both be non-empty")
    if {sample.profitable for sample in fit} != {0, 1}:
        raise InsufficientTrainingDataError("classifier fit set must contain both classes")
    if {sample.profitable for sample in calibration} != {0, 1}:
        raise InsufficientTrainingDataError("calibration set must contain both classes")
    return fit, calibration


def train_trade_meta_model(
    config: MetaModelTrainingConfig,
    sources: tuple[MetaTrainingSource, ...],
    oos_registry: OOSRegistry,
) -> TrainedTradeMetaModel:
    """Cumulatively retrain one immutable scope-specific v1 model artifact."""
    samples = extract_meta_training_samples(config, sources, oos_registry)
    fit, calibration = chronological_calibration_split(samples, config.calibration_fraction)
    base_parameters: dict[str, bool | int | float | str] = {
        "n_estimators": 40,
        "learning_rate": 0.05,
        "max_depth": 3,
        "num_leaves": 7,
        "min_child_samples": 2,
        "random_state": config.random_seed,
        "n_jobs": 1,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    classifier = lgb.LGBMClassifier(**base_parameters)
    regressor = lgb.LGBMRegressor(**base_parameters)
    classifier_parameters = classifier.get_params(deep=False)
    regressor_parameters = regressor.get_params(deep=False)
    classifier.fit(
        [list(sample.feature_values) for sample in fit],
        [sample.profitable for sample in fit],
    )
    raw_scores = classifier.predict(
        [list(sample.feature_values) for sample in calibration],
        raw_score=True,
    )
    calibrator = LogisticRegression(random_state=config.random_seed, solver="lbfgs")
    calibrator.fit(
        [[float(value)] for value in raw_scores],
        [sample.profitable for sample in calibration],
    )
    regressor.fit(
        [list(sample.feature_values) for sample in samples],
        [float(sample.net_return) for sample in samples],
    )
    source_provenance = []
    for source in sorted(
        sources,
        key=lambda item: (
            item.backtest_result.window_start,
            item.backtest_result.window_end,
            item.backtest_result.run_id,
        ),
    ):
        start_date, end_date = _full_day_range(source.backtest_result)
        source_provenance.append(
            TrainingSourceProvenance(
                run_id=source.backtest_result.run_id,
                start_date=start_date,
                end_date=end_date,
                result_fingerprint=fingerprint_backtest_result(source.backtest_result),
            )
        )
    metadata = MetaModelMetadata(
        ml_architecture_version=ML_ARCHITECTURE_VERSION,
        research_scope_id=config.research_scope_id,
        plan_id=config.plan_id,
        model_version=config.model_version,
        allowed_strategy_ids=tuple(sorted(config.allowed_strategy_ids)),
        feature_names=config.feature_schema.feature_names,
        created_at=config.created_at,
        git_commit=config.git_commit,
        random_seed=config.random_seed,
        calibration_fraction=config.calibration_fraction,
        calibration_cutoff=calibration[0].signal_timestamp,
        classifier_parameters=classifier_parameters,
        regressor_parameters=regressor_parameters,
        calibration_coefficient=float(calibrator.coef_[0][0]),
        calibration_intercept=float(calibrator.intercept_[0]),
        sizing_policy_id=SIZING_POLICY_ID,
        regressor_training_population="all eligible cumulative training samples",
        training_sample_count=len(samples),
        actual_sample_count=sum(not sample.is_shadow for sample in samples),
        shadow_sample_count=sum(sample.is_shadow for sample in samples),
        training_sources=tuple(source_provenance),
        scikit_learn_version=sklearn.__version__,
        lightgbm_version=lgb.__version__,
    )
    return TrainedTradeMetaModel(
        metadata=metadata,
        classifier=classifier.booster_,
        regressor=regressor.booster_,
    )


def _validate_generated_signal(signal: Signal) -> None:
    if not isinstance(signal, Signal):
        raise TypeError("signal must be a Signal")
    if signal.status is not SignalStatus.GENERATED:
        raise ValueError("Trade Meta-Model scoring requires a GENERATED signal")


def _full_day_range(result: Any) -> tuple[date, date]:
    if (
        result.window_start.tzinfo is None
        or result.window_start.utcoffset() is None
        or result.window_end.tzinfo is None
        or result.window_end.utcoffset() is None
    ):
        raise ValueError("Meta-Model training source boundaries must be timezone-aware")
    start = result.window_start.astimezone(MARKET_TIMEZONE)
    end = result.window_end.astimezone(MARKET_TIMEZONE)
    if start.time() != time.min or end.time() != time.min:
        raise ValueError("Meta-Model training sources require full-day Asia/Kolkata boundaries")
    if start >= end:
        raise ValueError("training source range must be non-empty")
    return start.date(), end.date()


def _canonical_identity(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Decimal):
        return format(value, "f")
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, tuple):
        return [_canonical_identity(item) for item in value]
    return value


def _sample_sort_key(sample: MetaTrainingSample) -> tuple[datetime, str, bool]:
    identity = json.dumps(
        _canonical_identity(sample.candidate_identity),
        sort_keys=True,
        separators=(",", ":"),
    )
    return sample.signal_timestamp, identity, sample.is_shadow


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)
