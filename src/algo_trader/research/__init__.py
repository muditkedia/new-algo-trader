"""Generic, integrity-checked research orchestration support."""

from algo_trader.research.artifacts import (
    REVIEW_JSON_NAME,
    REVIEW_WORKBOOK_NAME,
    CompletedResearchRun,
    ResearchArtifactIntegrityError,
    ResearchRunDiscovery,
    build_review_artifacts,
    discover_research_runs,
    finalize_staging_run_directory,
    prepare_staging_run_directory,
)
from algo_trader.research.fingerprints import (
    MarketDataFileManifest,
    build_market_data_manifest,
    canonical_fingerprint,
    environment_snapshot,
    file_sha256,
)
from algo_trader.research.ml_lifecycle import (
    ModelLifecycleSelection,
    eligible_training_sources,
    prepare_scope_model_candidate,
    select_scope_trade_scorer,
)
from algo_trader.research.models import ResearchDecisionRecord
from algo_trader.research.orchestration import (
    ResearchStrategySpec,
    score_and_build_requests,
)

__all__ = [
    "REVIEW_JSON_NAME",
    "REVIEW_WORKBOOK_NAME",
    "CompletedResearchRun",
    "MarketDataFileManifest",
    "ModelLifecycleSelection",
    "ResearchArtifactIntegrityError",
    "ResearchDecisionRecord",
    "ResearchRunDiscovery",
    "ResearchStrategySpec",
    "build_review_artifacts",
    "build_market_data_manifest",
    "canonical_fingerprint",
    "discover_research_runs",
    "eligible_training_sources",
    "finalize_staging_run_directory",
    "environment_snapshot",
    "file_sha256",
    "prepare_staging_run_directory",
    "prepare_scope_model_candidate",
    "score_and_build_requests",
    "select_scope_trade_scorer",
]
