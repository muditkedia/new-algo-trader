"""Immutable records for the two explicit machine-learning systems."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_serializer,
    model_validator,
)

from algo_trader.backtest import BacktestRunResult

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
FiniteDecimal = Annotated[Decimal, Field(allow_inf_nan=False)]
NonNegativeDecimal = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]
StrictDate = Annotated[date, Field(strict=True)]
K = TypeVar("K")
V = TypeVar("V")


class FrozenMapping[K, V](dict[K, V]):
    """Detached, deterministic immutable mapping used by ML-owned records."""

    def __init__(self, values: Mapping[K, V]) -> None:
        dict.__init__(
            self,
            sorted(values.items(), key=lambda item: (type(item[0]).__name__, repr(item[0]))),
        )

    def _immutable(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("FrozenMapping does not support mutation")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __repr__(self) -> str:
        return f"FrozenMapping({dict(self)!r})"


def _deep_freeze(value: object) -> object:
    if isinstance(value, FrozenMapping):
        return value
    if isinstance(value, Mapping):
        return FrozenMapping({deepcopy(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, set | frozenset):
        return tuple(
            sorted(
                (_deep_freeze(item) for item in value),
                key=lambda item: (type(item).__name__, repr(item)),
            )
        )
    if isinstance(value, BaseModel):
        return value
    try:
        return deepcopy(value)
    except (TypeError, ValueError):
        return value


def _deep_thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_deep_thaw(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted((_deep_thaw(item) for item in value), key=repr)
    return value


class FrozenMLModel(BaseModel):
    """Validation policy for immutable ML-owned records."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def deeply_freeze(self) -> FrozenMLModel:
        for name in type(self).model_fields:
            object.__setattr__(self, name, _deep_freeze(getattr(self, name)))
        return self

    @model_serializer(mode="wrap")
    def serialize_frozen_values(self, handler: object) -> object:
        return _deep_thaw(handler(self))  # type: ignore[operator]


class MetaFeatureSchema(FrozenMLModel):
    """Caller-ordered top-level signal feature names for one scope model."""

    feature_names: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def validate_names(self) -> MetaFeatureSchema:
        if not self.feature_names:
            raise ValueError("feature_names must not be empty")
        if len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError("feature_names must be unique")
        return self


class MetaTrainingSource(FrozenMLModel):
    """One guarded backtest result and its strategy-lineage membership."""

    research_scope_id: NonEmptyStr
    plan_id: NonEmptyStr
    strategy_ids: tuple[NonEmptyStr, ...]
    backtest_result: BacktestRunResult

    @model_validator(mode="after")
    def validate_strategy_ids(self) -> MetaTrainingSource:
        if not self.strategy_ids:
            raise ValueError("strategy_ids must not be empty")
        if len(self.strategy_ids) != len(set(self.strategy_ids)):
            raise ValueError("strategy_ids must be unique")
        return self


class MetaModelTrainingConfig(FrozenMLModel):
    """Explicit reproducibility and scope contract for cumulative retraining."""

    research_scope_id: NonEmptyStr
    plan_id: NonEmptyStr
    model_version: NonEmptyStr
    feature_schema: MetaFeatureSchema
    allowed_strategy_ids: tuple[NonEmptyStr, ...]
    created_at: datetime
    git_commit: NonEmptyStr
    random_seed: int = Field(strict=True)
    calibration_fraction: float = Field(gt=0, lt=0.5, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_config(self) -> MetaModelTrainingConfig:
        if not self.allowed_strategy_ids:
            raise ValueError("allowed_strategy_ids must not be empty")
        if len(self.allowed_strategy_ids) != len(set(self.allowed_strategy_ids)):
            raise ValueError("allowed_strategy_ids must be unique")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return self


class MetaTrainingSample(FrozenMLModel):
    """One completed actual or shadow signal observation after OOS guarding."""

    candidate_identity: tuple[object, ...]
    signal_timestamp: datetime
    feature_values: tuple[float, ...]
    profitable: int = Field(ge=0, le=1)
    net_return: FiniteDecimal
    is_shadow: bool


class TrainingSourceProvenance(FrozenMLModel):
    """Exact guarded source range and canonical backtest fingerprint."""

    run_id: NonEmptyStr
    start_date: date
    end_date: date
    result_fingerprint: NonEmptyStr


class MetaModelMetadata(FrozenMLModel):
    """Immutable scope-specific Trade Meta-Model artifact metadata."""

    ml_architecture_version: NonEmptyStr
    research_scope_id: NonEmptyStr
    plan_id: NonEmptyStr
    model_version: NonEmptyStr
    allowed_strategy_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    created_at: datetime
    git_commit: NonEmptyStr
    random_seed: int
    calibration_fraction: float
    calibration_cutoff: datetime
    classifier_parameters: Mapping[str, bool | int | float | str | None]
    regressor_parameters: Mapping[str, bool | int | float | str | None]
    calibration_coefficient: float
    calibration_intercept: float
    sizing_policy_id: NonEmptyStr
    regressor_training_population: NonEmptyStr
    training_sample_count: int = Field(gt=0)
    actual_sample_count: int = Field(ge=0)
    shadow_sample_count: int = Field(ge=0)
    training_sources: tuple[TrainingSourceProvenance, ...]
    scikit_learn_version: NonEmptyStr
    lightgbm_version: NonEmptyStr


class QualityBucketMetrics(FrozenMLModel):
    """Fixed quality interval diagnostics over completed scored trades."""

    lower_bound: Decimal
    upper_bound: Decimal
    upper_inclusive: bool
    count: int = Field(ge=0)
    actual_count: int = Field(ge=0)
    shadow_count: int = Field(ge=0)
    average_quality_score: NonNegativeDecimal | None
    average_calibrated_probability: NonNegativeDecimal | None
    empirical_win_rate: NonNegativeDecimal | None
    average_predicted_net_return: FiniteDecimal | None
    average_realized_net_return: FiniteDecimal | None


class SizeBucketMetrics(FrozenMLModel):
    """Diagnostics for one populated recommended-notional bucket."""

    recommended_notional: int
    count: int = Field(gt=0)
    average_quality_score: NonNegativeDecimal
    empirical_win_rate: NonNegativeDecimal
    average_realized_net_return: FiniteDecimal


class MetaModelEvaluation(FrozenMLModel):
    """Read-only diagnostics constrained by one verified artifact identity.

    The fingerprint identifies the immutable artifact selected for evaluation; it
    does not claim that legacy MLScore values cryptographically carry that hash.
    """

    research_scope_id: NonEmptyStr
    plan_id: NonEmptyStr
    model_version: NonEmptyStr
    allowed_strategy_ids: tuple[NonEmptyStr, ...]
    artifact_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_backtest_fingerprint: NonEmptyStr
    labeled_count: int = Field(ge=0)
    actual_labeled_count: int = Field(ge=0)
    shadow_labeled_count: int = Field(ge=0)
    profitable_count: int = Field(ge=0)
    empirical_win_rate: NonNegativeDecimal | None
    brier_score: NonNegativeDecimal | None
    predicted_net_return_mae: NonNegativeDecimal | None
    mean_predicted_net_return: FiniteDecimal | None
    mean_realized_net_return: FiniteDecimal | None
    allocated_entry_not_filled_count: int = Field(ge=0)
    shadow_entry_not_filled_count: int = Field(ge=0)
    capital_exhausted_count: int = Field(ge=0)
    quality_buckets: tuple[QualityBucketMetrics, ...]
    size_buckets: tuple[SizeBucketMetrics, ...]
    quality_return_monotonic: bool | None
    quality_win_rate_monotonic: bool | None
    size_return_monotonic: bool | None
    window_id: str | None = None


class MetaModelArtifactIdentity(FrozenMLModel):
    """Composite identity derived only from a checksum-verified artifact."""

    research_scope_id: NonEmptyStr
    plan_id: NonEmptyStr
    model_version: NonEmptyStr
    allowed_strategy_ids: tuple[NonEmptyStr, ...]
    artifact_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_strategy_ids(self) -> MetaModelArtifactIdentity:
        if not self.allowed_strategy_ids:
            raise ValueError("allowed_strategy_ids must not be empty")
        normalized = tuple(sorted(self.allowed_strategy_ids))
        if len(normalized) != len(set(normalized)):
            raise ValueError("allowed_strategy_ids must be unique")
        object.__setattr__(self, "allowed_strategy_ids", normalized)
        return self


class FloatParameterSpec(FrozenMLModel):
    """Bounded scalar float parameter eligible for sparse optimization."""

    kind: Literal["float"] = "float"
    name: NonEmptyStr
    baseline_value: float = Field(allow_inf_nan=False)
    low: float = Field(allow_inf_nan=False)
    high: float = Field(allow_inf_nan=False)
    step: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    log: bool = False

    @model_validator(mode="after")
    def validate_domain(self) -> FloatParameterSpec:
        if self.low >= self.high:
            raise ValueError("float parameter low must be below high")
        if not self.low <= self.baseline_value <= self.high:
            raise ValueError("float baseline_value must be inside the search range")
        if self.log and (self.low <= 0 or self.step is not None):
            raise ValueError("log float parameters require positive bounds and no step")
        if self.step is not None:
            offset = Decimal(str(self.baseline_value)) - Decimal(str(self.low))
            if offset % Decimal(str(self.step)) != 0:
                raise ValueError("float baseline_value must align with the search step")
        return self


class IntParameterSpec(FrozenMLModel):
    """Bounded scalar integer parameter eligible for sparse optimization."""

    kind: Literal["int"] = "int"
    name: NonEmptyStr
    baseline_value: int = Field(strict=True)
    low: int = Field(strict=True)
    high: int = Field(strict=True)
    step: int = Field(default=1, strict=True, gt=0)
    log: bool = False

    @model_validator(mode="after")
    def validate_domain(self) -> IntParameterSpec:
        if self.low >= self.high:
            raise ValueError("int parameter low must be below high")
        if not self.low <= self.baseline_value <= self.high:
            raise ValueError("int baseline_value must be inside the search range")
        if self.log and (self.low <= 0 or self.step != 1):
            raise ValueError("log int parameters require positive bounds and unit step")
        if (self.baseline_value - self.low) % self.step != 0:
            raise ValueError("int baseline_value must align with the search step")
        return self


class CategoricalParameterSpec(FrozenMLModel):
    """Finite scalar categorical parameter eligible for sparse optimization."""

    kind: Literal["categorical"] = "categorical"
    name: NonEmptyStr
    baseline_value: int | float | str | bool
    choices: tuple[int | float | str | bool, ...]

    @model_validator(mode="after")
    def validate_domain(self) -> CategoricalParameterSpec:
        if not self.choices:
            raise ValueError("categorical choices must not be empty")
        if len(self.choices) != len(set(self.choices)):
            raise ValueError("categorical choices must be unique")
        if self.baseline_value not in self.choices:
            raise ValueError("categorical baseline_value must be one of choices")
        return self


type ParameterSpec = FloatParameterSpec | IntParameterSpec | CategoricalParameterSpec


class OptimizationEvaluationRange(FrozenMLModel):
    """One half-open training-authorized range for strategy evaluation."""

    research_scope_id: NonEmptyStr
    plan_id: NonEmptyStr
    start_date: StrictDate
    end_date: StrictDate

    @model_validator(mode="after")
    def validate_range(self) -> OptimizationEvaluationRange:
        if isinstance(self.start_date, datetime) or isinstance(self.end_date, datetime):
            raise TypeError("optimization range boundaries must be date objects")
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be earlier than end_date")
        return self


class StrategyOptimizerConfig(FrozenMLModel):
    """Explicit sparse-search and reproducibility contract for one lineage."""

    research_scope_id: NonEmptyStr
    plan_id: NonEmptyStr
    strategy_id: NonEmptyStr
    baseline_parameters: Mapping[str, int | float | str | bool]
    parameter_specs: tuple[ParameterSpec, ...]
    evaluation_ranges: tuple[OptimizationEvaluationRange, ...]
    n_trials: int = Field(gt=0)
    random_seed: int
    max_changed_parameters: int = Field(gt=0)
    low_quality_threshold: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_config(self) -> StrategyOptimizerConfig:
        if not self.parameter_specs:
            raise ValueError("parameter_specs must not be empty")
        names = [spec.name for spec in self.parameter_specs]
        if len(names) != len(set(names)):
            raise ValueError("parameter spec names must be unique")
        if self.max_changed_parameters > len(self.parameter_specs):
            raise ValueError("max_changed_parameters cannot exceed parameter spec count")
        for spec in self.parameter_specs:
            if self.baseline_parameters.get(spec.name) != spec.baseline_value:
                raise ValueError("each parameter spec must match baseline_parameters exactly")
        if not self.evaluation_ranges:
            raise ValueError("evaluation_ranges must not be empty")
        if any(
            item.research_scope_id != self.research_scope_id
            or item.plan_id != self.plan_id
            for item in self.evaluation_ranges
        ):
            raise ValueError("all evaluation ranges must match optimizer scope and plan")
        if self.evaluation_ranges != tuple(
            sorted(self.evaluation_ranges, key=lambda item: (item.start_date, item.end_date))
        ):
            raise ValueError("evaluation_ranges must be chronological")
        return self


class OptimizationTrialState(StrEnum):
    """Persisted optimizer trial states exposed by completed results."""

    COMPLETE = "COMPLETE"


class OptimizationTrialResult(FrozenMLModel):
    """One complete multi-objective parameter evaluation."""

    trial_number: int = Field(ge=0)
    state: OptimizationTrialState
    parameters: Mapping[str, int | float | str | bool]
    changed_parameter_names: tuple[str, ...]
    parameter_distance: NonNegativeDecimal
    pf_score: NonNegativeDecimal
    average_net_return: FiniteDecimal
    average_cagr: FiniteDecimal
    average_win_rate: NonNegativeDecimal
    average_trades_per_day: NonNegativeDecimal
    worst_max_drawdown_pct: NonNegativeDecimal
    instability: NonNegativeDecimal
    low_quality_trade_fraction: NonNegativeDecimal
    evaluation_window_count: int = Field(gt=0)
    source_report_ids: tuple[str, ...]
    source_run_ids: tuple[str, ...]
    source_backtest_fingerprints: tuple[str, ...]


class StrategyOptimizationResult(FrozenMLModel):
    """All complete trials and the deterministic Pareto set, with no winner."""

    optimizer_version: NonEmptyStr
    research_scope_id: NonEmptyStr
    plan_id: NonEmptyStr
    strategy_id: NonEmptyStr
    random_seed: int
    max_changed_parameters: int
    low_quality_threshold: float
    optuna_version: NonEmptyStr
    objective_directions: tuple[str, ...]
    completed_trials: tuple[OptimizationTrialResult, ...]
    pareto_trials: tuple[OptimizationTrialResult, ...]
    baseline_trial: OptimizationTrialResult
