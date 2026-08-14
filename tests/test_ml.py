from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import optuna
import pytest
from pydantic import ValidationError

from algo_trader.backtest import (
    BacktestRequestOutcome,
    BacktestRequestResult,
    BacktestRunResult,
    BacktestTradeRecord,
    BacktestTradeRequest,
)
from algo_trader.costs import BrokeragePlan, LegCostBreakdown, RoundTripCostBreakdown
from algo_trader.domain import (
    ExitReason,
    Fill,
    MLScore,
    OrderIntent,
    Side,
    Signal,
    SignalStatus,
    Trade,
)
from algo_trader.ml import (
    ARTIFACT_FILES,
    ML_ARCHITECTURE_VERSION,
    OBJECTIVE_DIRECTIONS,
    STRATEGY_OPTIMIZER_VERSION,
    BootstrapTradeScorer,
    CategoricalParameterSpec,
    DuplicateTrainingObservationError,
    FeatureIntegrityError,
    FloatParameterSpec,
    InsufficientTrainingDataError,
    IntParameterSpec,
    MetaFeatureSchema,
    MetaModelArtifactIdentity,
    MetaModelEvaluationIntegrityError,
    MetaModelTrainingConfig,
    MetaTrainingSample,
    MetaTrainingSource,
    ModelArtifactIntegrityError,
    OptimizationEvaluationRange,
    StrategyOptimizerConfig,
    TradeScorer,
    chronological_calibration_split,
    evaluate_trade_meta_model,
    extract_meta_features,
    extract_meta_training_samples,
    inspect_trade_meta_model_artifact,
    load_trade_meta_model,
    optimize_strategy_parameters,
    parameter_distance,
    recommended_notional_for_quality,
    save_trade_meta_model,
    train_trade_meta_model,
)
from algo_trader.ml.optimizer import _prepare_trial_parameters
from algo_trader.oos import (
    OOSAuditContext,
    OOSRegistry,
    OOSTestRecord,
    OOSWindowSpec,
    create_oos_plan,
    fingerprint_backtest_result,
)
from algo_trader.portfolio import (
    AllocationCandidate,
    AllocationDecision,
    AllocationOutcome,
    CapitalReservation,
    MarginRequirementQuote,
)
from algo_trader.reporting import ReportContext, build_report

MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")
ZERO = Decimal("0")


def model_identity(
    *,
    scope: str = "scope-a",
    plan: str = "plan-a",
    version: str = "meta-v1",
    strategies: tuple[str, ...] = ("strategy-a",),
) -> MetaModelArtifactIdentity:
    return MetaModelArtifactIdentity(
        research_scope_id=scope,
        plan_id=plan,
        model_version=version,
        allowed_strategy_ids=strategies,
        artifact_fingerprint="a" * 64,
    )


def at(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=MARKET_TIMEZONE)


def audit(event_id: str, offset: int = 0) -> OOSAuditContext:
    return OOSAuditContext(
        event_id=event_id,
        occurred_at=at(2026, 8, 14, 10) + timedelta(minutes=offset),
        git_commit=f"audit-{event_id}",
    )


@pytest.fixture
def registry(tmp_path: Path) -> OOSRegistry:
    selected = OOSRegistry(tmp_path / "oos.duckdb")
    selected.create_plan(
        create_oos_plan(
            research_scope_id="scope-a",
            plan_id="plan-a",
            strategy_ids=("strategy-a",),
            data_start_date=date(2024, 1, 1),
            data_end_exclusive=date(2027, 1, 1),
            oos_windows=(
                OOSWindowSpec(
                    window_id="w1",
                    start_date=date(2025, 1, 1),
                    end_date=date(2025, 7, 1),
                ),
                OOSWindowSpec(
                    window_id="w2",
                    start_date=date(2025, 7, 1),
                    end_date=date(2026, 1, 1),
                ),
            ),
            audit_context=audit("create"),
        )
    )
    yield selected
    selected.close()


def zero_leg(turnover: Decimal) -> LegCostBreakdown:
    return LegCostBreakdown(
        turnover=turnover,
        brokerage=ZERO,
        exchange_transaction_charge=ZERO,
        sebi_turnover_fee=ZERO,
        ipft=ZERO,
        stt=ZERO,
        stamp_duty=ZERO,
        gst=ZERO,
    )


def make_record(
    index: int,
    pnl: Decimal,
    timestamp: datetime,
    *,
    strategy_id: str = "strategy-a",
    strategy_version: str = "1",
    shadow: bool = False,
    quality: float = 0.5,
    predicted_return: float = 0.0,
    model_version: str = "bootstrap-v1",
    feature_snapshot: dict[str, object] | None = None,
) -> BacktestTradeRecord:
    generated = Signal(
        strategy_id=strategy_id,
        strategy_version=strategy_version,
        symbol=f"SYM{index % 3}",
        timestamp=timestamp,
        side=Side.LONG if index % 2 == 0 else Side.SHORT,
        feature_snapshot=feature_snapshot or {
            "momentum": Decimal(str(index + 1)),
            "volatility": float((index % 5) + 1),
            "audit": {"source": "synthetic"},
        },
    )
    score = MLScore(
        model_version=model_version,
        quality_score=quality,
        calibrated_probability=quality,
        predicted_net_return=predicted_return,
        recommended_notional=recommended_notional_for_quality(quality),
    )
    order = OrderIntent(
        signal=generated,
        timestamp=timestamp + timedelta(minutes=1),
        quantity=500,
        requested_notional=score.recommended_notional,
    )
    candidate = AllocationCandidate(order_intent=order, ml_score=score)
    quote = MarginRequirementQuote(provider_id="margin-v1", required_margin=Decimal("10000"))
    if shadow:
        decision = AllocationDecision(
            candidate=candidate,
            outcome=AllocationOutcome.CAPACITY_REJECTED,
            margin_quote=quote,
            signal=generated.model_copy(update={"status": SignalStatus.CAPACITY_REJECTED}),
        )
        completed_signal = decision.signal
    else:
        decision = AllocationDecision(
            candidate=candidate,
            outcome=AllocationOutcome.ALLOCATED,
            margin_quote=quote,
            signal=generated,
            reservation=CapitalReservation(candidate=candidate, margin_quote=quote),
        )
        completed_signal = generated.model_copy(update={"status": SignalStatus.EXECUTED})
    breakdown = RoundTripCostBreakdown(
        schedule_id="cost-policy",
        entry=zero_leg(Decimal("50000")),
        exit=zero_leg(Decimal("50000") + pnl),
    )
    trade = Trade(
        signal=completed_signal,
        ml_score=score,
        target_notional=score.recommended_notional,
        entry_fill=Fill(
            timestamp=timestamp + timedelta(minutes=2),
            price=Decimal("100"),
            quantity=500,
            is_simulated=True,
        ),
        exit_fill=Fill(
            timestamp=timestamp + timedelta(minutes=30),
            price=Decimal("101"),
            quantity=500,
            is_simulated=True,
        ),
        gross_pnl=pnl,
        total_costs=ZERO,
        net_pnl=pnl,
        mfe_return=Decimal("0.02"),
        mae_return=Decimal("-0.01"),
        exit_reason=ExitReason.TIME_EXIT,
        is_shadow=shadow,
    )
    return BacktestTradeRecord(
        trade=trade,
        round_trip_cost_breakdown=breakdown,
        cost_policy_id="cost-policy",
        allocation_identity=candidate.identity,
        allocation_decision=decision,
    )


def make_result(
    start: datetime,
    end: datetime,
    *,
    records: tuple[BacktestTradeRecord, ...] = (),
    shadows: tuple[BacktestTradeRecord, ...] = (),
    run_id: str = "run-1",
) -> BacktestRunResult:
    actual_pnl = sum((record.trade.net_pnl for record in records), start=ZERO)
    strategy_versions = tuple(
        sorted(
            {
                (record.trade.signal.strategy_id, record.trade.signal.strategy_version)
                for record in (*records, *shadows)
            }
        )
    )
    model_versions = tuple(
        sorted({record.trade.ml_score.model_version for record in (*records, *shadows)})
    )
    return BacktestRunResult(
        run_id=run_id,
        git_commit="deadbeef",
        backtester_version="1",
        window_start=start,
        window_end=end,
        cost_policy_id="cost-policy",
        cost_policy_source_as_of_date=date(2026, 8, 14),
        brokerage_plan=BrokeragePlan.PLUS,
        starting_capital=Decimal("100000"),
        ending_capital=Decimal("100000") + actual_pnl,
        capital_exhausted=False,
        actual_trade_records=records,
        shadow_trade_records=shadows,
        request_results=(),
        ending_portfolio_state=None,
        symbols=tuple(sorted({record.trade.signal.symbol for record in (*records, *shadows)})),
        strategy_versions=strategy_versions,
        ml_model_versions=model_versions,
    )


def training_result() -> BacktestRunResult:
    records = tuple(
        make_record(
            index,
            Decimal("500") if index % 2 else Decimal("-250"),
            at(2024, 2, 1) + timedelta(minutes=5 * index),
            strategy_version="1" if index < 8 else "2",
            shadow=index in {5, 11},
        )
        for index in range(16)
    )
    return make_result(
        at(2024, 2, 1),
        at(2024, 2, 2),
        records=tuple(record for record in records if not record.trade.is_shadow),
        shadows=tuple(record for record in records if record.trade.is_shadow),
    )


def training_config() -> MetaModelTrainingConfig:
    return MetaModelTrainingConfig(
        research_scope_id="scope-a",
        plan_id="plan-a",
        model_version="meta-v1",
        feature_schema=MetaFeatureSchema(feature_names=("momentum", "volatility")),
        allowed_strategy_ids=("strategy-a",),
        created_at=at(2026, 8, 14, 12),
        git_commit="train-commit",
        random_seed=17,
        calibration_fraction=0.25,
    )


def test_feature_schema_and_extraction_are_ordered_strict_and_read_only() -> None:
    snapshot = {"first": Decimal("1.5"), "second": 2, "third": 3.25, "audit": {"x": 1}}
    signal = Signal(
        strategy_id="strategy-a",
        strategy_version="1",
        symbol="AAA",
        timestamp=at(2024, 2, 1, 9, 15),
        side=Side.LONG,
        feature_snapshot=snapshot,
    )
    schema = MetaFeatureSchema(feature_names=("second", "first", "third"))
    before = signal.model_dump()
    assert extract_meta_features(signal, schema) == (2.0, 1.5, 3.25)
    assert signal.model_dump() == before

    with pytest.raises(ValidationError, match="unique"):
        MetaFeatureSchema(feature_names=("first", "first"))


@pytest.mark.parametrize(
    "value",
    [True, None, "1", datetime(2024, 1, 1), [1], (1,), {"x": 1}, float("nan"), float("inf")],
)
def test_invalid_selected_features_are_rejected(value: object) -> None:
    signal = Signal(
        strategy_id="strategy-a",
        strategy_version="1",
        symbol="AAA",
        timestamp=at(2024, 2, 1, 9, 15),
        side=Side.LONG,
        feature_snapshot={"selected": value, "extra": Decimal("2")},
    )
    with pytest.raises(FeatureIntegrityError):
        extract_meta_features(signal, MetaFeatureSchema(feature_names=("selected",)))


def test_missing_feature_rejected_and_unselected_invalid_context_ignored() -> None:
    signal = Signal(
        strategy_id="strategy-a",
        strategy_version="1",
        symbol="AAA",
        timestamp=at(2024, 2, 1, 9, 15),
        side=Side.LONG,
        feature_snapshot={"selected": 1, "ignored": "audit context"},
    )
    assert extract_meta_features(signal, MetaFeatureSchema(feature_names=("selected",))) == (1.0,)
    with pytest.raises(FeatureIntegrityError, match="missing"):
        extract_meta_features(signal, MetaFeatureSchema(feature_names=("absent",)))


def test_oos_guard_precedes_extraction_and_scope_filtering(registry: OOSRegistry) -> None:
    allowed = training_result()
    source = MetaTrainingSource(
        research_scope_id="scope-a",
        plan_id="plan-a",
        strategy_ids=("strategy-a",),
        backtest_result=allowed,
    )
    samples = extract_meta_training_samples(training_config(), (source,), registry)
    assert len(samples) == 16
    assert sum(sample.is_shadow for sample in samples) == 2
    assert {sample.profitable for sample in samples} == {0, 1}

    invalid = make_record(
        99,
        Decimal("100"),
        at(2025, 2, 1, 9),
        feature_snapshot={"momentum": "outcome must not be inspected", "volatility": 1},
    )
    forbidden = make_result(
        at(2025, 2, 1),
        at(2025, 2, 2),
        records=(invalid,),
        run_id="forbidden",
    )
    with pytest.raises(PermissionError, match="not authorized"):
        extract_meta_training_samples(
            training_config(),
            (
                MetaTrainingSource(
                    research_scope_id="scope-a",
                    plan_id="plan-a",
                    strategy_ids=("strategy-a",),
                    backtest_result=forbidden,
                ),
            ),
            registry,
        )


def test_oos_training_lifecycle_allows_only_development_or_training_allowed(
    registry: OOSRegistry,
) -> None:
    window_result = make_result(
        at(2025, 1, 1),
        at(2025, 7, 1),
        records=(make_record(80, Decimal("500"), at(2025, 2, 1, 10)),),
        run_id="window-run",
    )
    source = MetaTrainingSource(
        research_scope_id="scope-a",
        plan_id="plan-a",
        strategy_ids=("strategy-a",),
        backtest_result=window_result,
    )
    config = training_config()
    with pytest.raises(PermissionError):
        extract_meta_training_samples(config, (source,), registry)

    registry.register_test_result(
        "scope-a",
        "plan-a",
        "w1",
        window_result,
        audit("test", 1),
        window_result.strategy_versions,
    )
    with pytest.raises(PermissionError):
        extract_meta_training_samples(config, (source,), registry)

    registry.mark_consumed("scope-a", "plan-a", "w1", audit("consume", 2))
    with pytest.raises(PermissionError):
        extract_meta_training_samples(config, (source,), registry)

    registry.authorize_training("scope-a", "plan-a", "w1", audit("authorize", 3))
    before = registry.transition_history("scope-a", "plan-a")
    assert len(extract_meta_training_samples(config, (source,), registry)) == 1
    assert registry.transition_history("scope-a", "plan-a") == before

    sealed = source.model_copy(
        update={
            "backtest_result": make_result(
                at(2026, 2, 1),
                at(2026, 2, 2),
                records=(make_record(81, Decimal("500"), at(2026, 2, 1, 10)),),
                run_id="sealed",
            )
        }
    )
    with pytest.raises(PermissionError):
        extract_meta_training_samples(config, (sealed,), registry)


def test_training_source_scope_and_full_day_boundaries_are_enforced(
    registry: OOSRegistry,
) -> None:
    result = training_result()
    source = MetaTrainingSource(
        research_scope_id="wrong-scope",
        plan_id="plan-a",
        strategy_ids=("strategy-a",),
        backtest_result=result,
    )
    with pytest.raises(ValueError, match="scope and plan"):
        extract_meta_training_samples(training_config(), (source,), registry)

    intraday = result.model_copy(update={"window_start": at(2024, 2, 1, 9, 15)})
    with pytest.raises(ValueError, match="full-day"):
        intraday_source = source.model_copy(
            update={"research_scope_id": "scope-a", "backtest_result": intraday}
        )
        extract_meta_training_samples(
            training_config(),
            (intraday_source,),
            registry,
        )


def test_unrelated_strategy_excluded_and_duplicate_observation_rejected(
    registry: OOSRegistry,
) -> None:
    result = training_result()
    unrelated = make_record(50, Decimal("999"), at(2024, 2, 1, 12), strategy_id="other")
    mixed = result.model_copy(
        update={"actual_trade_records": (*result.actual_trade_records, unrelated)}
    )
    source = MetaTrainingSource(
        research_scope_id="scope-a",
        plan_id="plan-a",
        strategy_ids=("strategy-a",),
        backtest_result=mixed,
    )
    samples = extract_meta_training_samples(training_config(), (source,), registry)
    assert len(samples) == 16
    assert Decimal("0.01998") not in {sample.net_return for sample in samples}
    with pytest.raises(DuplicateTrainingObservationError):
        extract_meta_training_samples(training_config(), (source, source), registry)


def test_chronological_split_keeps_timestamp_groups_and_requires_both_classes() -> None:
    samples = tuple(
        MetaTrainingSample(
            candidate_identity=(str(index),),
            signal_timestamp=at(2024, 2, 1, 9) + timedelta(minutes=5 * (index // 2)),
            feature_values=(float(index),),
            profitable=index % 2,
            net_return=Decimal("0.01") if index % 2 else Decimal("-0.01"),
            is_shadow=False,
        )
        for index in range(12)
    )
    fit, calibration = chronological_calibration_split(samples, 0.25)
    assert fit[-1].signal_timestamp < calibration[0].signal_timestamp
    assert {sample.profitable for sample in fit} == {0, 1}
    assert {sample.profitable for sample in calibration} == {0, 1}

    one_class = tuple(sample.model_copy(update={"profitable": 1}) for sample in samples)
    with pytest.raises(InsufficientTrainingDataError, match="both classes"):
        chronological_calibration_split(one_class, 0.25)


def test_training_scoring_determinism_and_scope_metadata(registry: OOSRegistry) -> None:
    source = MetaTrainingSource(
        research_scope_id="scope-a",
        plan_id="plan-a",
        strategy_ids=("strategy-a",),
        backtest_result=training_result(),
    )
    first = train_trade_meta_model(training_config(), (source,), registry)
    second = train_trade_meta_model(training_config(), (source,), registry)
    signal = Signal(
        strategy_id="strategy-a",
        strategy_version="3",
        symbol="NEW",
        timestamp=at(2024, 3, 1, 9, 15),
        side=Side.LONG,
        feature_snapshot={"momentum": Decimal("7"), "volatility": 2.0},
    )
    first_score = first.score(signal)
    assert isinstance(first, TradeScorer)
    assert first_score == second.score(signal)
    assert first_score.quality_score == first_score.calibrated_probability
    assert 0 <= first_score.quality_score <= 1
    assert first_score.recommended_notional in range(50_000, 100_001, 5_000)
    assert first.metadata.actual_sample_count == 14
    assert first.metadata.shadow_sample_count == 2
    assert first.metadata.feature_names == ("momentum", "volatility")
    assert first.metadata.training_sources[0].result_fingerprint
    assert first.metadata.scikit_learn_version == "1.9.0"
    assert first.metadata.lightgbm_version == "4.7.0"
    assert "random_state" in first.metadata.classifier_parameters

    with pytest.raises(ValueError, match="allowed lineage"):
        first.score(signal.model_copy(update={"strategy_id": "unrelated"}))
    with pytest.raises(FeatureIntegrityError):
        first.score(signal.model_copy(update={"feature_snapshot": {"momentum": 1}}))


@pytest.mark.parametrize(
    ("quality", "expected"),
    [
        (Decimal("0"), 50_000),
        (Decimal("0.05"), 55_000),
        (Decimal("0.5"), 75_000),
        (Decimal("1"), 100_000),
    ],
)
def test_quality_sizing_uses_monotonic_decimal_half_up(
    quality: Decimal,
    expected: int,
) -> None:
    assert recommended_notional_for_quality(quality) == expected
    values = [recommended_notional_for_quality(Decimal(index) / 100) for index in range(101)]
    assert values == sorted(values)


def test_bootstrap_scorer_is_neutral_configurable_and_feature_independent() -> None:
    scorer = BootstrapTradeScorer("bootstrap-v1", 60_000)
    signal = Signal(
        strategy_id="any",
        strategy_version="1",
        symbol="AAA",
        timestamp=at(2024, 1, 1, 9, 15),
        side=Side.LONG,
        feature_snapshot={},
    )
    score = scorer.score(signal)
    assert (score.quality_score, score.calibrated_probability, score.predicted_net_return) == (
        0.5,
        0.5,
        0.0,
    )
    assert score.recommended_notional == 60_000


def test_native_artifact_round_trip_checksum_and_no_overwrite(
    registry: OOSRegistry,
    tmp_path: Path,
) -> None:
    source = MetaTrainingSource(
        research_scope_id="scope-a",
        plan_id="plan-a",
        strategy_ids=("strategy-a",),
        backtest_result=training_result(),
    )
    model = train_trade_meta_model(training_config(), (source,), registry)
    assert ML_ARCHITECTURE_VERSION == "2"
    assert model.metadata.ml_architecture_version == "2"
    directory = save_trade_meta_model(model, tmp_path / "meta-v1")
    assert {path.name for path in directory.iterdir()} == {*ARTIFACT_FILES, "manifest.json"}
    first_record = source.backtest_result.actual_trade_records[0]
    signal = first_record.allocation_decision.candidate.order_intent.signal
    assert load_trade_meta_model(directory).score(signal) == model.score(signal)
    identity = inspect_trade_meta_model_artifact(directory)
    assert identity.research_scope_id == "scope-a"
    assert identity.plan_id == "plan-a"
    assert identity.model_version == "meta-v1"
    assert identity.allowed_strategy_ids == ("strategy-a",)
    assert len(identity.artifact_fingerprint) == 64
    assert inspect_trade_meta_model_artifact(directory) == identity
    with pytest.raises(FileExistsError):
        save_trade_meta_model(model, directory)

    classifier = directory / "classifier.txt"
    classifier.write_text(classifier.read_text(encoding="utf-8") + "tampered", encoding="utf-8")
    with pytest.raises(ModelArtifactIntegrityError, match="checksum"):
        load_trade_meta_model(directory)
    with pytest.raises(ModelArtifactIntegrityError, match="checksum"):
        inspect_trade_meta_model_artifact(directory)


def test_meta_evaluation_uses_actual_and_shadow_and_fixed_bins() -> None:
    records = (
        make_record(
            1,
            Decimal("-500"),
            at(2024, 2, 1, 9),
            quality=0.1,
            predicted_return=-0.008,
            model_version="meta-v1",
        ),
        make_record(
            2,
            Decimal("500"),
            at(2024, 2, 1, 10),
            quality=0.5,
            predicted_return=0.008,
            model_version="meta-v1",
        ),
        make_record(
            3,
            Decimal("1000"),
            at(2024, 2, 1, 11),
            quality=1.0,
            predicted_return=0.015,
            model_version="meta-v1",
        ),
    )
    shadow = make_record(
        4,
        Decimal("750"),
        at(2024, 2, 1, 12),
        quality=0.8,
        predicted_return=0.014,
        model_version="meta-v1",
        shadow=True,
    )
    result = make_result(at(2024, 2, 1), at(2024, 2, 2), records=records, shadows=(shadow,))
    evaluation = evaluate_trade_meta_model(result, model_identity())
    assert (evaluation.actual_labeled_count, evaluation.shadow_labeled_count) == (3, 1)
    assert evaluation.labeled_count == 4
    assert evaluation.brier_score == Decimal("0.075")
    assert evaluation.predicted_net_return_mae == Decimal("0.0025")
    assert len(evaluation.quality_buckets) == 5
    assert evaluation.quality_buckets[-1].count == 2
    assert evaluation.quality_return_monotonic is True
    assert evaluation.quality_win_rate_monotonic is True
    assert evaluation.size_return_monotonic is True

    one = make_result(at(2024, 2, 1), at(2024, 2, 2), records=(records[0],))
    assert evaluate_trade_meta_model(one, model_identity()).quality_return_monotonic is None


def test_meta_evaluation_verifies_optional_oos_provenance() -> None:
    record = make_record(
        1,
        Decimal("500"),
        at(2024, 2, 1, 9),
        model_version="meta-v1",
    )
    result = make_result(at(2024, 2, 1), at(2024, 2, 2), records=(record,))
    oos = OOSTestRecord(
        research_scope_id="scope-a",
        plan_id="plan-a",
        window_id="w1",
        backtest_run_id=result.run_id,
        backtest_git_commit=result.git_commit,
        backtester_version=result.backtester_version,
        backtest_window_start=result.window_start,
        backtest_window_end=result.window_end,
        cost_policy_id=result.cost_policy_id,
        brokerage_plan=result.brokerage_plan.value,
        symbols=result.symbols,
        strategy_versions=result.strategy_versions,
        ml_model_versions=result.ml_model_versions,
        result_fingerprint=fingerprint_backtest_result(result),
        scope_strategy_ids=("strategy-a",),
        tested_strategy_versions=result.strategy_versions,
        registration_audit=audit("evaluation"),
    )
    before = oos.model_dump()
    evaluation = evaluate_trade_meta_model(result, model_identity(), oos)
    assert evaluation.research_scope_id == "scope-a"
    assert oos.model_dump() == before
    with pytest.raises(ValueError, match="fingerprint"):
        evaluate_trade_meta_model(
            result,
            model_identity(),
            oos.model_copy(update={"result_fingerprint": "tampered"}),
        )
    with pytest.raises(MetaModelEvaluationIntegrityError, match="scope"):
        evaluate_trade_meta_model(
            result,
            model_identity(scope="another-scope"),
            oos,
        )
    with pytest.raises(MetaModelEvaluationIntegrityError, match="tested strategy"):
        evaluate_trade_meta_model(
            result,
            model_identity(),
            oos.model_copy(
                update={"tested_strategy_versions": (("strategy-a", "2"),)}
            ),
        )
    with pytest.raises(MetaModelEvaluationIntegrityError, match="outside model"):
        evaluate_trade_meta_model(
            result,
            model_identity(),
            oos.model_copy(
                update={"tested_strategy_versions": (("foreign-strategy", "1"),)}
            ),
        )


def test_meta_evaluation_requires_composite_identity_and_rejects_version_collision() -> None:
    allowed = make_record(1, Decimal("500"), at(2024, 2, 1, 9), model_version="shared")
    collision = make_record(
        2,
        Decimal("500"),
        at(2024, 2, 1, 10),
        strategy_id="other-strategy",
        model_version="shared",
    )
    result = make_result(
        at(2024, 2, 1), at(2024, 2, 2), records=(allowed, collision)
    )
    with pytest.raises(TypeError, match="MetaModelArtifactIdentity"):
        evaluate_trade_meta_model(result, "shared")  # type: ignore[arg-type]
    with pytest.raises(MetaModelEvaluationIntegrityError, match="outside"):
        evaluate_trade_meta_model(result, model_identity(version="shared"))


def request_result_for(
    record: BacktestTradeRecord,
    outcome: BacktestRequestOutcome,
) -> BacktestRequestResult:
    request = BacktestTradeRequest(candidate=record.allocation_decision.candidate)
    completed = outcome is BacktestRequestOutcome.COMPLETED_ACTUAL
    return BacktestRequestResult(
        request=request,
        outcome=outcome,
        terminal_at=(
            record.trade.exit_fill.timestamp
            if completed
            else request.candidate.order_intent.timestamp
        ),
        allocation_decision=record.allocation_decision,
        trade_record=record if completed else None,
    )


def test_meta_evaluation_uses_real_request_results_and_ignores_other_versions() -> None:
    actual = make_record(
        20, Decimal("500"), at(2024, 2, 1, 9), model_version="meta-v1"
    )
    allowed_no_fill = make_record(
        21, Decimal("500"), at(2024, 2, 1, 10), model_version="meta-v1"
    )
    unrelated = make_record(
        22,
        Decimal("500"),
        at(2024, 2, 1, 11),
        strategy_id="other-strategy",
        model_version="other-model",
    )
    result = make_result(
        at(2024, 2, 1),
        at(2024, 2, 2),
        records=(actual,),
    ).model_copy(
        update={
            "request_results": (
                request_result_for(actual, BacktestRequestOutcome.COMPLETED_ACTUAL),
                request_result_for(
                    allowed_no_fill,
                    BacktestRequestOutcome.ALLOCATED_ENTRY_NOT_FILLED,
                ),
                request_result_for(
                    unrelated,
                    BacktestRequestOutcome.ALLOCATED_ENTRY_NOT_FILLED,
                ),
            ),
            "strategy_versions": (("other-strategy", "1"), ("strategy-a", "1")),
            "ml_model_versions": ("meta-v1", "other-model"),
        }
    )
    evaluation = evaluate_trade_meta_model(result, model_identity())
    assert evaluation.actual_labeled_count == 1
    assert evaluation.allocated_entry_not_filled_count == 1


def test_request_only_same_version_foreign_strategy_is_lineage_collision() -> None:
    actual = make_record(
        23, Decimal("500"), at(2024, 2, 1, 9), model_version="meta-v1"
    )
    foreign = make_record(
        24,
        Decimal("500"),
        at(2024, 2, 1, 10),
        strategy_id="foreign-strategy",
        model_version="meta-v1",
    )
    result = make_result(
        at(2024, 2, 1), at(2024, 2, 2), records=(actual,)
    ).model_copy(
        update={
            "request_results": (
                request_result_for(actual, BacktestRequestOutcome.COMPLETED_ACTUAL),
                request_result_for(
                    foreign,
                    BacktestRequestOutcome.ALLOCATED_ENTRY_NOT_FILLED,
                ),
            ),
            "strategy_versions": (("foreign-strategy", "1"), ("strategy-a", "1")),
        }
    )
    with pytest.raises(MetaModelEvaluationIntegrityError, match="outside"):
        evaluate_trade_meta_model(result, model_identity())


def oos_record_for_result(
    result: BacktestRunResult,
    *,
    tested_strategy_versions: tuple[tuple[str, str], ...] = (("strategy-a", "1"),),
) -> OOSTestRecord:
    return OOSTestRecord(
        research_scope_id="scope-a",
        plan_id="plan-a",
        window_id="w1",
        backtest_run_id=result.run_id,
        backtest_git_commit=result.git_commit,
        backtester_version=result.backtester_version,
        backtest_window_start=result.window_start,
        backtest_window_end=result.window_end,
        cost_policy_id=result.cost_policy_id,
        brokerage_plan=result.brokerage_plan.value,
        symbols=result.symbols,
        scope_strategy_ids=("strategy-a",),
        tested_strategy_versions=tested_strategy_versions,
        strategy_versions=result.strategy_versions,
        ml_model_versions=result.ml_model_versions,
        result_fingerprint=fingerprint_backtest_result(result),
        registration_audit=audit("zero-signal-evaluation"),
    )


def test_zero_signal_oos_attestation_evaluates_without_fabricated_records() -> None:
    result = make_result(at(2024, 2, 1), at(2024, 2, 2))
    evaluation = evaluate_trade_meta_model(
        result,
        model_identity(),
        oos_record_for_result(result),
    )
    assert evaluation.labeled_count == 0
    assert evaluation.actual_labeled_count == 0
    assert evaluation.shadow_labeled_count == 0


def test_empty_strategy_provenance_with_request_result_fails_oos_verification() -> None:
    no_fill_record = make_record(
        25, Decimal("500"), at(2024, 2, 1, 10), model_version="meta-v1"
    )
    result = make_result(at(2024, 2, 1), at(2024, 2, 2)).model_copy(
        update={
            "request_results": (
                request_result_for(
                    no_fill_record,
                    BacktestRequestOutcome.ALLOCATED_ENTRY_NOT_FILLED,
                ),
            ),
            "symbols": (no_fill_record.trade.signal.symbol,),
            "ml_model_versions": ("meta-v1",),
        }
    )
    with pytest.raises(MetaModelEvaluationIntegrityError, match="zero-signal"):
        evaluate_trade_meta_model(
            result,
            model_identity(),
            oos_record_for_result(result),
        )


def test_ml_owned_nested_values_are_detached_immutable_and_serializable() -> None:
    caller = {"nested": [{"value": 1}], "unordered": {3, 1, 2}}
    sample = MetaTrainingSample(
        candidate_identity=(caller,),
        signal_timestamp=at(2024, 2, 1, 9),
        feature_values=(1.0,),
        profitable=1,
        net_return=Decimal("0.01"),
        is_shadow=False,
    )
    caller["nested"][0]["value"] = 99
    caller["unordered"].add(4)
    stored = sample.candidate_identity[0]
    assert stored["nested"][0]["value"] == 1
    assert stored["unordered"] == (1, 2, 3)
    with pytest.raises(TypeError):
        stored["nested"][0]["value"] = 2
    assert sample.model_dump(mode="python") == sample.model_dump(mode="python")
    assert sample.model_dump(mode="json")["candidate_identity"][0]["nested"] == [
        {"value": 1}
    ]

    baseline = {"threshold": 1.0, "mode": "base", "fixed": True}
    values = optimizer_config(1).model_dump(mode="python")
    values["baseline_parameters"] = baseline
    config = StrategyOptimizerConfig(**values)
    baseline["threshold"] = 9.0
    assert config.baseline_parameters["threshold"] == 1.0
    with pytest.raises(TypeError):
        config.baseline_parameters["threshold"] = 2.0


class ControlledTrial:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = values

    def suggest_categorical(self, name, choices):
        del choices
        return self.values[name]

    def suggest_int(self, name, low, high, *, step, log):
        del low, high, step, log
        return self.values[name]

    def suggest_float(self, name, low, high, *, step, log):
        del low, high, step, log
        return self.values[name]


def test_optimizer_change_budget_uses_actual_values_after_sampling() -> None:
    config = StrategyOptimizerConfig(
        research_scope_id="scope-a",
        plan_id="plan-a",
        strategy_id="strategy-a",
        baseline_parameters={"category": "base", "integer": 2, "floating": 1.0},
        parameter_specs=(
            CategoricalParameterSpec(
                name="category", baseline_value="base", choices=("base", "other")
            ),
            IntParameterSpec(name="integer", baseline_value=2, low=1, high=3),
            FloatParameterSpec(
                name="floating", baseline_value=1.0, low=0.5, high=1.5, step=0.5
            ),
        ),
        evaluation_ranges=optimizer_config(1).evaluation_ranges,
        n_trials=1,
        random_seed=1,
        max_changed_parameters=1,
        low_quality_threshold=0.5,
    )
    baseline_values = {
        "change::category": True,
        "value::category": "base",
        "change::integer": True,
        "value::integer": 2,
        "change::floating": True,
        "value::floating": 1.0,
    }
    parameters, changed, distance = _prepare_trial_parameters(
        ControlledTrial(baseline_values), config  # type: ignore[arg-type]
    )
    assert parameters == config.baseline_parameters
    assert changed == ()
    assert distance == 0

    two_changes = baseline_values | {
        "value::category": "other",
        "value::integer": 3,
    }
    with pytest.raises(optuna.TrialPruned, match="max_changed_parameters"):
        _prepare_trial_parameters(ControlledTrial(two_changes), config)  # type: ignore[arg-type]


def report_for_range(
    selected: OptimizationEvaluationRange,
    parameters: dict[str, int | float | str | bool],
    index: int,
):
    multiplier = Decimal(str(parameters["threshold"]))
    records = (
        make_record(
            index * 10 + 1,
            Decimal("500") * multiplier + Decimal("100") * index,
            datetime.combine(selected.start_date, datetime.min.time(), MARKET_TIMEZONE)
            + timedelta(hours=10),
            quality=0.2,
            model_version="meta-v1",
        ),
        make_record(
            index * 10 + 2,
            Decimal("-250"),
            datetime.combine(selected.start_date, datetime.min.time(), MARKET_TIMEZONE)
            + timedelta(hours=11),
            quality=0.8,
            model_version="meta-v1",
        ),
    )
    result = make_result(
        datetime.combine(selected.start_date, datetime.min.time(), MARKET_TIMEZONE),
        datetime.combine(selected.end_date, datetime.min.time(), MARKET_TIMEZONE),
        records=records,
        shadows=(
            make_record(
                index * 10 + 3,
                Decimal("50000"),
                datetime.combine(selected.start_date, datetime.min.time(), MARKET_TIMEZONE)
                + timedelta(hours=12),
                quality=0.0,
                model_version="meta-v1",
                shadow=True,
            ),
        ),
        run_id=f"optimizer-{index}-{parameters['threshold']}-{parameters['mode']}",
    )
    return build_report(
        result,
        ReportContext(
            report_id=f"report-{result.run_id}",
            generated_at=at(2026, 8, 14, 12),
            trading_dates=(selected.start_date,),
        ),
    )


class SyntheticEvaluator:
    def __init__(self) -> None:
        self.calls: list[dict[str, int | float | str | bool]] = []

    def evaluate(self, parameters, evaluation_ranges):
        copied = dict(parameters)
        self.calls.append(copied)
        return tuple(
            report_for_range(selected, copied, index)
            for index, selected in enumerate(evaluation_ranges)
        )


class MissingReportEvaluator:
    def evaluate(self, parameters, evaluation_ranges):
        return ()


class ReversedReportEvaluator:
    def evaluate(self, parameters, evaluation_ranges):
        copied = dict(parameters)
        return tuple(
            report_for_range(selected, copied, index)
            for index, selected in enumerate(reversed(evaluation_ranges))
        )


class UndefinedProfitFactorEvaluator:
    def evaluate(self, parameters, evaluation_ranges):
        reports = []
        for index, selected in enumerate(evaluation_ranges):
            timestamp = datetime.combine(
                selected.start_date, datetime.min.time(), MARKET_TIMEZONE
            ) + timedelta(hours=10)
            result = make_result(
                datetime.combine(selected.start_date, datetime.min.time(), MARKET_TIMEZONE),
                datetime.combine(selected.end_date, datetime.min.time(), MARKET_TIMEZONE),
                records=(make_record(index, ZERO, timestamp, model_version="meta-v1"),),
                run_id=f"undefined-{index}",
            )
            reports.append(
                build_report(
                    result,
                    ReportContext(
                        report_id=f"undefined-report-{index}",
                        generated_at=at(2026, 8, 14, 12),
                        trading_dates=(selected.start_date,),
                    ),
                )
            )
        return tuple(reports)


def optimizer_config(n_trials: int = 12) -> StrategyOptimizerConfig:
    return StrategyOptimizerConfig(
        research_scope_id="scope-a",
        plan_id="plan-a",
        strategy_id="strategy-a",
        baseline_parameters={"threshold": 1.0, "mode": "base", "fixed": True},
        parameter_specs=(
            FloatParameterSpec(
                name="threshold",
                baseline_value=1.0,
                low=0.5,
                high=1.5,
                step=0.25,
            ),
            CategoricalParameterSpec(
                name="mode",
                baseline_value="base",
                choices=("base", "alternate"),
            ),
        ),
        evaluation_ranges=(
            OptimizationEvaluationRange(
                research_scope_id="scope-a",
                plan_id="plan-a",
                start_date=date(2024, 2, 1),
                end_date=date(2024, 2, 2),
            ),
            OptimizationEvaluationRange(
                research_scope_id="scope-a",
                plan_id="plan-a",
                start_date=date(2024, 3, 1),
                end_date=date(2024, 3, 2),
            ),
        ),
        n_trials=n_trials,
        random_seed=7,
        max_changed_parameters=1,
        low_quality_threshold=0.5,
    )


def test_parameter_specs_distance_and_sparse_config_validation() -> None:
    with pytest.raises(ValidationError, match="inside"):
        IntParameterSpec(name="period", baseline_value=20, low=1, high=10)
    with pytest.raises(ValidationError, match="choices"):
        CategoricalParameterSpec(name="mode", baseline_value="x", choices=("a", "b"))
    specs = optimizer_config(1).parameter_specs
    assert parameter_distance({"threshold": 1.0, "mode": "base"}, specs) == 0
    assert parameter_distance({"threshold": 1.5, "mode": "alternate"}, specs) == Decimal("0.75")
    with pytest.raises(ValidationError, match="cannot exceed"):
        values = optimizer_config(1).model_dump() | {"max_changed_parameters": 3}
        StrategyOptimizerConfig(**values)


def test_optimizer_guards_all_ranges_before_evaluation(registry: OOSRegistry) -> None:
    evaluator = SyntheticEvaluator()
    config = optimizer_config(1)
    forbidden = config.model_copy(
        update={
            "evaluation_ranges": (
                OptimizationEvaluationRange(
                    research_scope_id="scope-a",
                    plan_id="plan-a",
                    start_date=date(2025, 2, 1),
                    end_date=date(2025, 2, 2),
                ),
            )
        }
    )
    with pytest.raises(PermissionError, match="not authorized"):
        optimize_strategy_parameters(forbidden, evaluator, registry)
    assert evaluator.calls == []


def test_optimizer_is_seeded_sparse_multiobjective_and_returns_pareto(
    registry: OOSRegistry,
) -> None:
    evaluator = SyntheticEvaluator()
    result = optimize_strategy_parameters(optimizer_config(), evaluator, registry)
    assert STRATEGY_OPTIMIZER_VERSION == "2"
    assert result.optimizer_version == "2"
    assert result.objective_directions == OBJECTIVE_DIRECTIONS
    assert result.baseline_trial.trial_number == 0
    assert result.baseline_trial.parameters == {
        "threshold": 1.0,
        "mode": "base",
        "fixed": True,
    }
    assert result.baseline_trial.parameter_distance == 0
    expected_pf_score = (
        (Decimal("2") / Decimal("3")) + (Decimal("2.4") / Decimal("3.4"))
    ) / 2
    assert float(result.baseline_trial.pf_score) == pytest.approx(float(expected_pf_score))
    assert result.baseline_trial.average_net_return == Decimal("0.003")
    assert result.baseline_trial.average_win_rate == Decimal("0.5")
    assert result.baseline_trial.average_trades_per_day == Decimal("2")
    assert result.baseline_trial.worst_max_drawdown_pct == Decimal("250") / Decimal("100500")
    assert result.baseline_trial.instability == Decimal("0.0005")
    assert result.baseline_trial.low_quality_trade_fraction == Decimal("0.5")
    assert result.baseline_trial.evaluation_window_count == 2
    assert result.pareto_trials
    assert all(len(trial.changed_parameter_names) <= 1 for trial in result.completed_trials)
    assert all(
        trial.changed_parameter_names
        == tuple(
            spec.name
            for spec in optimizer_config().parameter_specs
            if trial.parameters[spec.name] != spec.baseline_value
        )
        for trial in result.completed_trials
    )
    assert len(result.completed_trials) < optimizer_config().n_trials
    assert len(evaluator.calls) == len(result.completed_trials)
    assert all(call["fixed"] is True for call in evaluator.calls)
    assert not hasattr(result, "best_strategy")
    assert not hasattr(result, "strategy_approved")


def test_optimizer_same_seed_and_evaluator_are_deterministic(registry: OOSRegistry) -> None:
    first = optimize_strategy_parameters(optimizer_config(6), SyntheticEvaluator(), registry)
    second = optimize_strategy_parameters(optimizer_config(6), SyntheticEvaluator(), registry)
    assert [trial.parameters for trial in first.completed_trials] == [
        trial.parameters for trial in second.completed_trials
    ]
    assert [trial.model_dump() for trial in first.pareto_trials] == [
        trial.model_dump() for trial in second.pareto_trials
    ]


@pytest.mark.parametrize("evaluator", [MissingReportEvaluator(), ReversedReportEvaluator()])
def test_optimizer_rejects_missing_extra_or_misordered_report_contract(
    registry: OOSRegistry,
    evaluator: object,
) -> None:
    with pytest.raises(ValueError, match="report"):
        optimize_strategy_parameters(optimizer_config(1), evaluator, registry)


def test_optimizer_prunes_undefined_profit_factor(registry: OOSRegistry) -> None:
    with pytest.raises(ValueError, match="baseline trial"):
        optimize_strategy_parameters(
            optimizer_config(1), UndefinedProfitFactorEvaluator(), registry
        )
