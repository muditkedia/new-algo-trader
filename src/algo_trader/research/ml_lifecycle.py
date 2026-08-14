"""Scope-specific, authorization-aware Trade Meta-Model lifecycle wiring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from algo_trader.backtest import BacktestRunResult
from algo_trader.ml import (
    BootstrapTradeScorer,
    MetaFeatureSchema,
    MetaModelArtifactIdentity,
    MetaModelEvaluation,
    MetaModelTrainingConfig,
    MetaTrainingSource,
    TradeScorer,
    evaluate_trade_meta_model,
    inspect_trade_meta_model_artifact,
    load_trade_meta_model,
    save_trade_meta_model,
    train_trade_meta_model,
)
from algo_trader.oos import OOSRegistry
from algo_trader.research.fingerprints import canonical_fingerprint

ACTIVE_MODEL_POINTER = "active_model.json"


@dataclass(frozen=True, slots=True)
class ModelLifecycleSelection:
    scorer: TradeScorer
    scorer_kind: str
    reason: str
    artifact_identity: MetaModelArtifactIdentity | None = None
    evaluation: MetaModelEvaluation | None = None


def select_scope_trade_scorer(
    *,
    model_root: Path,
    bootstrap_model_version: str,
    research_scope_id: str,
    plan_id: str,
    strategy_id: str,
    feature_schema: MetaFeatureSchema,
    strategy_config_fingerprint: str,
    completed_results: tuple[BacktestRunResult, ...],
) -> ModelLifecycleSelection:
    """Load only an explicitly evaluated active artifact; otherwise bootstrap visibly."""
    pointer_path = Path(model_root) / ACTIVE_MODEL_POINTER
    if not pointer_path.exists():
        return ModelLifecycleSelection(
            scorer=BootstrapTradeScorer(bootstrap_model_version),
            scorer_kind="BOOTSTRAP",
            reason="no explicitly evaluated active model artifact exists",
        )
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    required = {
        "artifact_directory",
        "artifact_fingerprint",
        "strategy_config_fingerprint",
        "evaluation_run_id",
    }
    if not isinstance(pointer, dict) or not required <= set(pointer):
        raise ValueError("active model pointer is malformed")
    if pointer["strategy_config_fingerprint"] != strategy_config_fingerprint:
        raise ValueError("active model artifact is stale for the current strategy config")
    artifact_directory = Path(model_root) / str(pointer["artifact_directory"])
    identity = inspect_trade_meta_model_artifact(artifact_directory)
    if identity.artifact_fingerprint != pointer["artifact_fingerprint"]:
        raise ValueError("active model pointer artifact fingerprint does not match")
    if (
        identity.research_scope_id != research_scope_id
        or identity.plan_id != plan_id
        or identity.allowed_strategy_ids != (strategy_id,)
    ):
        raise ValueError("active model artifact lineage is incompatible")
    model = load_trade_meta_model(artifact_directory)
    if model.metadata.feature_names != feature_schema.feature_names:
        raise ValueError("active model feature schema is incompatible")
    matching = tuple(
        result for result in completed_results if result.run_id == pointer["evaluation_run_id"]
    )
    if len(matching) != 1:
        raise ValueError("active model evaluation result is unavailable or ambiguous")
    evaluation = evaluate_trade_meta_model(matching[0], identity)
    if evaluation.artifact_fingerprint != identity.artifact_fingerprint:
        raise ValueError("active model evaluation identity is inconsistent")
    return ModelLifecycleSelection(
        scorer=model,
        scorer_kind="TRAINED",
        reason="checksum-verified compatible model has explicit evaluation evidence",
        artifact_identity=identity,
        evaluation=evaluation,
    )


def eligible_training_sources(
    *,
    results: tuple[tuple[BacktestRunResult, dict[str, object]], ...],
    registry: OOSRegistry,
    research_scope_id: str,
    plan_id: str,
    strategy_id: str,
    strategy_version: str,
    strategy_config_fingerprint: str,
) -> tuple[MetaTrainingSource, ...]:
    """Select exact-config development and TRAINING_ALLOWED results only."""
    selected: list[MetaTrainingSource] = []
    for result, manifest in results:
        reproducibility = manifest.get("reproducibility")
        if not isinstance(reproducibility, dict):
            continue
        if reproducibility.get("strategy_config_fingerprint") != strategy_config_fingerprint:
            continue
        if dict(result.strategy_versions).get(strategy_id) != strategy_version:
            continue
        start = result.window_start.date()
        end = result.window_end.date()
        try:
            registry.assert_training_range_allowed(research_scope_id, plan_id, start, end)
        except PermissionError:
            continue
        selected.append(
            MetaTrainingSource(
                research_scope_id=research_scope_id,
                plan_id=plan_id,
                strategy_ids=(strategy_id,),
                backtest_result=result,
            )
        )
    return tuple(selected)


def prepare_scope_model_candidate(
    *,
    config: MetaModelTrainingConfig,
    sources: tuple[MetaTrainingSource, ...],
    registry: OOSRegistry,
    model_root: Path,
    strategy_config_fingerprint: str,
) -> tuple[Path, MetaModelArtifactIdentity]:
    """Train and persist an immutable candidate; never activate it implicitly."""
    model = train_trade_meta_model(config, sources, registry)
    source_fingerprint = canonical_fingerprint(
        [item.model_dump(mode="json") for item in model.metadata.training_sources]
    )
    directory_name = (
        f"{config.model_version}-{strategy_config_fingerprint[:12]}-"
        f"{source_fingerprint[:12]}"
    )
    directory = save_trade_meta_model(model, Path(model_root) / directory_name)
    identity = inspect_trade_meta_model_artifact(directory)
    lifecycle = {
        "status": "CANDIDATE_UNEVALUATED",
        "strategy_config_fingerprint": strategy_config_fingerprint,
        "training_data_lineage_fingerprint": source_fingerprint,
        "feature_schema": config.feature_schema.model_dump(mode="json"),
        "artifact_fingerprint": identity.artifact_fingerprint,
        "created_at": config.created_at.isoformat(),
    }
    (directory / "lifecycle.json").write_text(
        json.dumps(lifecycle, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
        newline="\n",
    )
    return directory, identity
