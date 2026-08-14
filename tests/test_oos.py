from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from algo_trader.backtest import BacktestRunResult
from algo_trader.costs import BrokeragePlan
from algo_trader.oos import (
    DEFAULT_OOS_PROTOCOL_VERSION,
    SEALED_HOLDOUT_WINDOW_ID,
    OOSAuditContext,
    OOSPlan,
    OOSRegistry,
    OOSTestRecord,
    OOSWindowSpec,
    OOSWindowState,
    create_oos_plan,
    fingerprint_backtest_result,
    shift_calendar_months,
)

MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")
AUDIT_TIME = datetime(2026, 8, 14, 12, 0, tzinfo=MARKET_TIMEZONE)


def audit(event_id: str, offset: int = 0) -> OOSAuditContext:
    return OOSAuditContext(
        event_id=event_id,
        occurred_at=AUDIT_TIME + timedelta(minutes=offset),
        git_commit=f"commit-{event_id}",
    )


def specs() -> tuple[OOSWindowSpec, ...]:
    return (
        OOSWindowSpec(
            window_id="w1",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 7, 1),
        ),
        OOSWindowSpec(
            window_id="w2",
            start_date=date(2024, 7, 1),
            end_date=date(2025, 1, 1),
        ),
    )


def make_plan(
    scope: str = "scope-a",
    plan_id: str = "plan-a",
    *,
    event_id: str | None = None,
    data_start: date = date(2023, 1, 1),
    data_end: date = date(2026, 1, 1),
    windows: tuple[OOSWindowSpec, ...] | None = None,
    strategy_ids: tuple[str, ...] = ("strategy",),
) -> OOSPlan:
    return create_oos_plan(
        research_scope_id=scope,
        plan_id=plan_id,
        strategy_ids=strategy_ids,
        data_start_date=data_start,
        data_end_exclusive=data_end,
        oos_windows=windows if windows is not None else specs(),
        audit_context=audit(event_id or f"create-{scope}-{plan_id}"),
    )


def make_result(
    start_date: date,
    end_date: date,
    *,
    run_id: str = "run-1",
    ending_capital: Decimal = Decimal("101000"),
) -> BacktestRunResult:
    start = datetime.combine(start_date, datetime.min.time(), MARKET_TIMEZONE)
    end = datetime.combine(end_date, datetime.min.time(), MARKET_TIMEZONE)
    return BacktestRunResult(
        run_id=run_id,
        git_commit="backtest-commit",
        backtester_version="1",
        window_start=start,
        window_end=end,
        cost_policy_id="angel-one-nse-intraday-backtest-2026-08-14-plus",
        cost_policy_source_as_of_date=date(2026, 8, 14),
        brokerage_plan=BrokeragePlan.PLUS,
        starting_capital=Decimal("100000"),
        ending_capital=ending_capital,
        capital_exhausted=False,
        actual_trade_records=(),
        shadow_trade_records=(),
        request_results=(),
        ending_portfolio_state=None,
        symbols=("AAA", "BBB"),
        strategy_versions=(("strategy", "1"),),
        ml_model_versions=("model-1",),
    )


def register_first(registry: OOSRegistry, scope: str, plan_id: str, event: str = "test"):
    window = registry.next_testable_window(scope, plan_id)
    assert window is not None
    return registry.register_test_result(
        scope,
        plan_id,
        window.window_id,
        make_result(window.start_date, window.end_date),
        audit(event, 1),
        (("strategy", "1"),),
    )


@pytest.mark.parametrize(
    ("source", "months", "expected"),
    [
        (date(2026, 8, 12), -12, date(2025, 8, 12)),
        (date(2024, 2, 29), -12, date(2023, 2, 28)),
        (date(2024, 3, 31), -1, date(2024, 2, 29)),
    ],
)
def test_calendar_month_shift_clamps_deterministically(
    source: date,
    months: int,
    expected: date,
) -> None:
    assert shift_calendar_months(source, months) == expected


def test_plan_has_exact_final_twelve_calendar_month_holdout() -> None:
    plan = make_plan()

    assert DEFAULT_OOS_PROTOCOL_VERSION == "2"
    assert plan.protocol_version == "2"
    assert plan.sealed_holdout_start_date == date(2025, 1, 1)
    assert plan.sealed_holdout_end_exclusive == date(2026, 1, 1)
    assert plan.sealed_holdout.state is OOSWindowState.SEALED_HOLDOUT
    assert plan.sealed_holdout.window_id == SEALED_HOLDOUT_WINDOW_ID
    assert plan.development_start_date == date(2023, 1, 1)
    assert plan.development_end_exclusive == date(2024, 1, 1)


def test_twelve_month_holdout_is_not_a_365_day_delta() -> None:
    data_end = date(2024, 3, 1)
    windows = (
        OOSWindowSpec(
            window_id="w1",
            start_date=date(2022, 9, 1),
            end_date=date(2023, 3, 1),
        ),
    )
    plan = make_plan(data_start=date(2022, 1, 1), data_end=data_end, windows=windows)

    assert plan.sealed_holdout_start_date == date(2023, 3, 1)
    assert data_end - plan.sealed_holdout_start_date == timedelta(days=366)


def test_insufficient_pre_holdout_history_is_rejected() -> None:
    with pytest.raises(ValidationError, match="non-empty pre-holdout"):
        make_plan(
            data_start=date(2025, 1, 1),
            data_end=date(2026, 1, 1),
            windows=(
                OOSWindowSpec(
                    window_id="w",
                    start_date=date(2025, 1, 1),
                    end_date=date(2025, 6, 1),
                ),
            ),
        )


@pytest.mark.parametrize(
    ("windows", "message"),
    [
        (
            (
                OOSWindowSpec(
                    window_id="same",
                    start_date=date(2024, 1, 1),
                    end_date=date(2024, 7, 1),
                ),
                OOSWindowSpec(
                    window_id="same",
                    start_date=date(2024, 7, 1),
                    end_date=date(2025, 1, 1),
                ),
            ),
            "unique",
        ),
        (
            (
                OOSWindowSpec(
                    window_id="w1",
                    start_date=date(2024, 1, 1),
                    end_date=date(2024, 8, 1),
                ),
                OOSWindowSpec(
                    window_id="w2",
                    start_date=date(2024, 7, 1),
                    end_date=date(2025, 1, 1),
                ),
            ),
            "contiguous",
        ),
        (
            (
                OOSWindowSpec(
                    window_id="w1",
                    start_date=date(2024, 1, 1),
                    end_date=date(2024, 6, 1),
                ),
                OOSWindowSpec(
                    window_id="w2",
                    start_date=date(2024, 7, 1),
                    end_date=date(2025, 1, 1),
                ),
            ),
            "contiguous",
        ),
        (
            (
                OOSWindowSpec(
                    window_id="w1",
                    start_date=date(2023, 1, 1),
                    end_date=date(2025, 1, 1),
                ),
            ),
            "non-empty development",
        ),
        (
            (
                OOSWindowSpec(
                    window_id="w1",
                    start_date=date(2024, 1, 1),
                    end_date=date(2024, 12, 1),
                ),
            ),
            "end exactly",
        ),
        (
            (
                OOSWindowSpec(
                    window_id="w1",
                    start_date=date(2024, 1, 1),
                    end_date=date(2025, 2, 1),
                ),
            ),
            "pre-holdout",
        ),
    ],
)
def test_invalid_ordinary_partitions_are_rejected(
    windows: tuple[OOSWindowSpec, ...],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        make_plan(windows=windows)


def test_datetime_is_not_accepted_as_partition_date() -> None:
    with pytest.raises(ValidationError):
        OOSWindowSpec(
            window_id="w",
            start_date=datetime(2024, 1, 1),
            end_date=date(2024, 2, 1),
        )


def test_caller_cannot_shorten_or_move_holdout() -> None:
    plan = make_plan()
    values = plan.model_dump()
    values["sealed_holdout_start_date"] = date(2025, 2, 1)

    with pytest.raises(ValidationError, match="exactly 12 calendar months"):
        OOSPlan(**values)


def test_models_are_immutable() -> None:
    plan = make_plan()

    with pytest.raises(ValidationError):
        plan.plan_id = "changed"


def test_strategy_binding_is_normalized_persisted_and_scope_stable(tmp_path: Path) -> None:
    plan = make_plan(strategy_ids=[" beta ", "alpha"])
    assert plan.strategy_ids == ("alpha", "beta")
    with pytest.raises(ValidationError, match="duplicates"):
        make_plan(strategy_ids=("alpha", " alpha "))
    with OOSRegistry(tmp_path / "oos.duckdb") as registry:
        registry.create_plan(plan)
        assert registry.get_plan("scope-a", "plan-a").strategy_ids == (
            "alpha",
            "beta",
        )
        with pytest.raises(ValueError, match="binding"):
            registry.create_plan(
                make_plan(
                    plan_id="drift",
                    data_start=date(2026, 1, 1),
                    data_end=date(2029, 1, 1),
                    windows=(
                        OOSWindowSpec(
                            window_id="later",
                            start_date=date(2027, 1, 1),
                            end_date=date(2028, 1, 1),
                        ),
                    ),
                    strategy_ids=("alpha",),
                )
            )


@pytest.mark.parametrize("strategy_ids", ["strategy-a", " ", None])
def test_create_oos_plan_rejects_non_container_strategy_ids(
    strategy_ids: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="strategy_ids"):
        make_plan(strategy_ids=strategy_ids)  # type: ignore[arg-type]


def test_create_oos_plan_normalizes_supported_strategy_iterables() -> None:
    assert make_plan(strategy_ids=("strategy-a",)).strategy_ids == ("strategy-a",)
    assert make_plan(strategy_ids=[" beta ", "alpha"]).strategy_ids == (
        "alpha",
        "beta",
    )
    with pytest.raises(ValidationError, match="duplicates"):
        make_plan(strategy_ids=("alpha", " alpha "))


def direct_test_record(**updates: object) -> OOSTestRecord:
    result = make_result(date(2024, 1, 1), date(2024, 7, 1))
    values: dict[str, object] = {
        "research_scope_id": "scope-a",
        "plan_id": "plan-a",
        "window_id": "w1",
        "backtest_run_id": result.run_id,
        "backtest_git_commit": result.git_commit,
        "backtester_version": result.backtester_version,
        "backtest_window_start": result.window_start,
        "backtest_window_end": result.window_end,
        "cost_policy_id": result.cost_policy_id,
        "brokerage_plan": result.brokerage_plan.value,
        "symbols": result.symbols,
        "scope_strategy_ids": ("strategy",),
        "tested_strategy_versions": (("strategy", "1"),),
        "strategy_versions": (("strategy", "1"),),
        "ml_model_versions": result.ml_model_versions,
        "result_fingerprint": fingerprint_backtest_result(result),
        "registration_audit": audit("direct-record"),
    }
    return OOSTestRecord(**(values | updates))


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"scope_strategy_ids": ()}, "scope_strategy_ids"),
        ({"tested_strategy_versions": ()}, "tested_strategy_versions"),
        (
            {
                "tested_strategy_versions": (("foreign", "1"),),
                "strategy_versions": (("foreign", "1"),),
            },
            "belong",
        ),
        ({"tested_strategy_versions": (("strategy", "2"),)}, "exactly equal"),
    ],
)
def test_oos_test_record_rejects_inconsistent_v2_lineage(
    updates: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        direct_test_record(**updates)


def test_oos_test_record_normalizes_lineage_and_accepts_zero_signal_attestation() -> None:
    record = direct_test_record(
        scope_strategy_ids=[" beta ", "alpha"],
        tested_strategy_versions=[(" beta ", "2"), ("alpha", "1")],
        strategy_versions=(),
    )
    assert record.scope_strategy_ids == ("alpha", "beta")
    assert record.tested_strategy_versions == (("alpha", "1"), ("beta", "2"))
    assert record.strategy_versions == ()


def test_legacy_unbound_scope_fails_closed_without_inference(tmp_path: Path) -> None:
    with OOSRegistry(tmp_path / "oos.duckdb") as registry:
        registry.create_plan(make_plan())
        registry._connection.execute(
            "DELETE FROM oos_research_scope_strategies WHERE research_scope_id = ?",
            ["scope-a"],
        )
        with pytest.raises(RuntimeError, match="legacy.*binding"):
            registry.get_plan("scope-a", "plan-a")


@pytest.mark.parametrize(
    ("attestation", "message"),
    [
        ((), "non-empty"),
        ((("outside", "1"),), "outside"),
        ((("strategy", "2"),), "exactly"),
    ],
)
def test_tested_strategy_attestation_rejects_invalid_input_before_mutation(
    tmp_path: Path,
    attestation: tuple[tuple[str, str], ...],
    message: str,
) -> None:
    with OOSRegistry(tmp_path / "oos.duckdb") as registry:
        registry.create_plan(make_plan())
        result = make_result(date(2024, 1, 1), date(2024, 7, 1))
        with pytest.raises(ValueError, match=message):
            registry.register_test_result(
                "scope-a", "plan-a", "w1", result, audit("invalid"), attestation
            )
        assert registry.next_testable_window("scope-a", "plan-a").window_id == "w1"
        with pytest.raises(LookupError):
            registry.get_test_record("scope-a", "plan-a", "w1")


def test_empty_result_requires_and_persists_explicit_strategy_attestation(
    tmp_path: Path,
) -> None:
    result = make_result(date(2024, 1, 1), date(2024, 7, 1)).model_copy(
        update={"strategy_versions": (), "ml_model_versions": ()}
    )
    with OOSRegistry(tmp_path / "oos.duckdb") as registry:
        registry.create_plan(make_plan())
        record = registry.register_test_result(
            "scope-a",
            "plan-a",
            "w1",
            result,
            audit("empty"),
            (("strategy", "1"),),
        )
        assert record.scope_strategy_ids == ("strategy",)
        assert record.tested_strategy_versions == (("strategy", "1"),)
        assert registry.get_test_record("scope-a", "plan-a", "w1") == record


def test_different_scopes_can_persist_identical_horizons_and_window_ids(
    tmp_path: Path,
) -> None:
    with OOSRegistry(tmp_path / "oos.duckdb") as registry:
        registry.create_plan(make_plan("scope-a", "plan", event_id="create-a"))
        registry.create_plan(make_plan("scope-b", "plan", event_id="create-b"))

        assert registry.next_testable_window("scope-a", "plan").window_id == "w1"
        assert registry.next_testable_window("scope-b", "plan").window_id == "w1"


def test_same_scope_overlapping_plan_is_rejected_even_with_new_plan_id(
    tmp_path: Path,
) -> None:
    with OOSRegistry(tmp_path / "oos.duckdb") as registry:
        registry.create_plan(make_plan("scope-a", "original", event_id="create-original"))
        with pytest.raises(ValueError, match="overlapping data horizon"):
            registry.create_plan(make_plan("scope-a", "reset", event_id="create-reset"))


def test_same_scope_cannot_reset_tested_dates_via_overlapping_plan(
    tmp_path: Path,
) -> None:
    with OOSRegistry(tmp_path / "oos.duckdb") as registry:
        registry.create_plan(make_plan(event_id="create"))
        register_first(registry, "scope-a", "plan-a", "tested")

        with pytest.raises(ValueError, match="overlapping data horizon"):
            registry.create_plan(make_plan(plan_id="new-plan", event_id="reset"))


def test_next_window_requires_training_allowed_predecessor(tmp_path: Path) -> None:
    with OOSRegistry(tmp_path / "oos.duckdb") as registry:
        registry.create_plan(make_plan(event_id="create"))
        assert registry.next_testable_window("scope-a", "plan-a").window_id == "w1"

        register_first(registry, "scope-a", "plan-a", "tested")
        assert registry.next_testable_window("scope-a", "plan-a") is None
        registry.mark_consumed("scope-a", "plan-a", "w1", audit("consumed", 2))
        assert registry.next_testable_window("scope-a", "plan-a") is None
        registry.authorize_training(
            "scope-a", "plan-a", "w1", audit("training", 3)
        )
        assert registry.next_testable_window("scope-a", "plan-a").window_id == "w2"


def test_future_and_sealed_windows_are_never_testable(tmp_path: Path) -> None:
    with OOSRegistry(tmp_path / "oos.duckdb") as registry:
        registry.create_plan(make_plan(event_id="create"))
        with pytest.raises(PermissionError):
            registry.assert_oos_test_range_allowed(
                "scope-a", "plan-a", date(2024, 7, 1), date(2025, 1, 1)
            )
        with pytest.raises(PermissionError):
            registry.assert_oos_test_range_allowed(
                "scope-a", "plan-a", date(2025, 1, 1), date(2026, 1, 1)
            )


def test_exact_backtest_result_registration_and_provenance(tmp_path: Path) -> None:
    result = make_result(date(2024, 1, 1), date(2024, 7, 1))
    before = result.model_dump()
    with OOSRegistry(tmp_path / "oos.duckdb") as registry:
        registry.create_plan(make_plan(event_id="create"))
        record = registry.register_test_result(
            "scope-a",
            "plan-a",
            "w1",
            result,
            audit("tested", 1),
            result.strategy_versions,
        )
        persisted = registry.get_test_record("scope-a", "plan-a", "w1")

        assert record == persisted
        assert record.backtest_run_id == result.run_id
        assert record.backtest_git_commit == result.git_commit
        assert record.backtester_version == result.backtester_version
        assert record.cost_policy_id == result.cost_policy_id
        assert record.brokerage_plan == result.brokerage_plan.value
        assert record.symbols == result.symbols
        assert record.strategy_versions == result.strategy_versions
        assert record.ml_model_versions == result.ml_model_versions
        assert registry.get_plan("scope-a", "plan-a").oos_windows[0].state is (
            OOSWindowState.TESTED
        )
    assert result.model_dump() == before


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (date(2024, 1, 2), date(2024, 7, 1)),
        (date(2024, 1, 1), date(2024, 6, 30)),
        (date(2024, 1, 1), date(2025, 1, 1)),
    ],
)
def test_non_exact_backtest_window_is_rejected(
    tmp_path: Path,
    start: date,
    end: date,
) -> None:
    with OOSRegistry(tmp_path / "oos.duckdb") as registry:
        registry.create_plan(make_plan(event_id="create"))
        with pytest.raises(ValueError, match="must exactly match"):
            registry.register_test_result(
                "scope-a",
                "plan-a",
                "w1",
                make_result(start, end),
                audit("test", 1),
                (("strategy", "1"),),
            )


def test_duplicate_run_and_window_retesting_are_rejected(tmp_path: Path) -> None:
    with OOSRegistry(tmp_path / "oos.duckdb") as registry:
        registry.create_plan(make_plan(event_id="create"))
        register_first(registry, "scope-a", "plan-a", "tested")

        with pytest.raises(ValueError, match="current testable"):
            registry.register_test_result(
                "scope-a",
                "plan-a",
                "w1",
                make_result(date(2024, 1, 1), date(2024, 7, 1), run_id="new-run"),
                audit("retest", 2),
                (("strategy", "1"),),
            )


def test_run_ids_are_plan_scoped_not_global_across_research_scopes(
    tmp_path: Path,
) -> None:
    with OOSRegistry(tmp_path / "oos.duckdb") as registry:
        registry.create_plan(make_plan("scope-a", "plan", event_id="create-a"))
        registry.create_plan(make_plan("scope-b", "plan", event_id="create-b"))
        register_first(registry, "scope-a", "plan", "test-a")
        register_first(registry, "scope-b", "plan", "test-b")

        assert registry.get_test_record("scope-a", "plan", "w1").backtest_run_id == "run-1"
        assert registry.get_test_record("scope-b", "plan", "w1").backtest_run_id == "run-1"


def test_identical_cross_scope_window_states_evolve_independently(
    tmp_path: Path,
) -> None:
    with OOSRegistry(tmp_path / "oos.duckdb") as registry:
        registry.create_plan(make_plan("scope-a", "plan", event_id="create-a"))
        registry.create_plan(make_plan("scope-b", "plan", event_id="create-b"))
        register_first(registry, "scope-a", "plan", "test-a")
        registry.mark_consumed("scope-a", "plan", "w1", audit("consume-a", 2))
        registry.authorize_training("scope-a", "plan", "w1", audit("train-a", 3))

        scope_a = registry.get_plan("scope-a", "plan").oos_windows[0]
        scope_b = registry.get_plan("scope-b", "plan").oos_windows[0]
        assert scope_a.start_date == scope_b.start_date
        assert scope_a.end_date == scope_b.end_date
        assert scope_a.state is OOSWindowState.TRAINING_ALLOWED
        assert scope_b.state is OOSWindowState.AVAILABLE


def test_duplicate_run_id_is_rejected_within_plan_across_windows(
    tmp_path: Path,
) -> None:
    with OOSRegistry(tmp_path / "oos.duckdb") as registry:
        registry.create_plan(make_plan(event_id="create"))
        register_first(registry, "scope-a", "plan-a", "test-w1")
        registry.mark_consumed("scope-a", "plan-a", "w1", audit("consume", 2))
        registry.authorize_training("scope-a", "plan-a", "w1", audit("train", 3))

        with pytest.raises(ValueError, match="run_id"):
            registry.register_test_result(
                "scope-a",
                "plan-a",
                "w2",
                make_result(date(2024, 7, 1), date(2025, 1, 1), run_id="run-1"),
                audit("test-w2", 4),
                (("strategy", "1"),),
            )
        assert registry.get_plan("scope-a", "plan-a").oos_windows[1].state is (
            OOSWindowState.AVAILABLE
        )


def test_fingerprint_is_deterministic_and_materially_sensitive() -> None:
    first = make_result(date(2024, 1, 1), date(2024, 7, 1))
    reconstructed = make_result(date(2024, 1, 1), date(2024, 7, 1))
    changed = make_result(
        date(2024, 1, 1),
        date(2024, 7, 1),
        ending_capital=Decimal("99999"),
    )
    equivalent_decimal_representation = first.model_copy(
        update={"ending_capital": Decimal("101000.00")}
    )

    assert fingerprint_backtest_result(first) == fingerprint_backtest_result(reconstructed)
    assert fingerprint_backtest_result(first) == fingerprint_backtest_result(
        equivalent_decimal_representation
    )
    assert fingerprint_backtest_result(first) != fingerprint_backtest_result(changed)
    assert len(fingerprint_backtest_result(first)) == 64


def test_state_transitions_are_monotonic_and_failed_transition_is_atomic(
    tmp_path: Path,
) -> None:
    with OOSRegistry(tmp_path / "oos.duckdb") as registry:
        registry.create_plan(make_plan(event_id="create"))
        with pytest.raises(ValueError, match="requires CONSUMED"):
            registry.authorize_training(
                "scope-a", "plan-a", "w1", audit("skip", 1)
            )
        assert registry.get_plan("scope-a", "plan-a").oos_windows[0].state is (
            OOSWindowState.AVAILABLE
        )

        register_first(registry, "scope-a", "plan-a", "tested")
        registry.mark_consumed("scope-a", "plan-a", "w1", audit("consumed", 2))
        training = registry.authorize_training(
            "scope-a", "plan-a", "w1", audit("training", 3)
        )
        assert training.state is OOSWindowState.TRAINING_ALLOWED
        with pytest.raises(ValueError, match="requires TESTED"):
            registry.mark_consumed(
                "scope-a", "plan-a", "w1", audit("backward", 4)
            )


def test_duplicate_event_id_rejected_registry_globally_and_state_unchanged(
    tmp_path: Path,
) -> None:
    with OOSRegistry(tmp_path / "oos.duckdb") as registry:
        registry.create_plan(make_plan(event_id="create"))
        register_first(registry, "scope-a", "plan-a", "tested")
        with pytest.raises(ValueError, match="event_id"):
            registry.mark_consumed(
                "scope-a", "plan-a", "w1", audit("tested", 2)
            )
        assert registry.get_plan("scope-a", "plan-a").oos_windows[0].state is (
            OOSWindowState.TESTED
        )


@pytest.mark.parametrize(
    ("state", "allowed"),
    [
        (OOSWindowState.AVAILABLE, False),
        (OOSWindowState.TESTED, False),
        (OOSWindowState.CONSUMED, False),
        (OOSWindowState.TRAINING_ALLOWED, True),
    ],
)
def test_training_guard_follows_scope_window_state(
    tmp_path: Path,
    state: OOSWindowState,
    allowed: bool,
) -> None:
    with OOSRegistry(tmp_path / "oos.duckdb") as registry:
        registry.create_plan(make_plan(event_id="create"))
        if state is not OOSWindowState.AVAILABLE:
            register_first(registry, "scope-a", "plan-a", "tested")
        if state in {OOSWindowState.CONSUMED, OOSWindowState.TRAINING_ALLOWED}:
            registry.mark_consumed("scope-a", "plan-a", "w1", audit("consumed", 2))
        if state is OOSWindowState.TRAINING_ALLOWED:
            registry.authorize_training(
                "scope-a", "plan-a", "w1", audit("training", 3)
            )

        def operation() -> None:
            registry.assert_training_range_allowed(
                "scope-a", "plan-a", date(2024, 1, 1), date(2024, 7, 1)
            )

        if allowed:
            operation()
        else:
            with pytest.raises(PermissionError):
                operation()


def test_training_guard_allows_development_and_merges_adjacent_authorized_oos(
    tmp_path: Path,
) -> None:
    with OOSRegistry(tmp_path / "oos.duckdb") as registry:
        registry.create_plan(make_plan(event_id="create"))
        registry.assert_training_range_allowed(
            "scope-a", "plan-a", date(2023, 1, 1), date(2024, 1, 1)
        )
        register_first(registry, "scope-a", "plan-a", "tested")
        registry.mark_consumed("scope-a", "plan-a", "w1", audit("consumed", 2))
        registry.authorize_training(
            "scope-a", "plan-a", "w1", audit("training", 3)
        )

        ranges = registry.training_allowed_ranges("scope-a", "plan-a")
        assert [(item.start_date, item.end_date) for item in ranges] == [
            (date(2023, 1, 1), date(2024, 7, 1))
        ]


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (date(2024, 7, 1), date(2025, 1, 1)),
        (date(2025, 1, 1), date(2026, 1, 1)),
        (date(2023, 12, 1), date(2024, 2, 1)),
    ],
)
def test_training_guard_rejects_forbidden_or_mixed_ranges(
    tmp_path: Path,
    start: date,
    end: date,
) -> None:
    with OOSRegistry(tmp_path / "oos.duckdb") as registry:
        registry.create_plan(make_plan(event_id="create"))
        with pytest.raises(PermissionError):
            registry.assert_training_range_allowed(
                "scope-a", "plan-a", start, end
            )


def test_scope_a_holdout_does_not_block_scope_b_development(tmp_path: Path) -> None:
    scope_b_windows = (
        OOSWindowSpec(
            window_id="b-w1",
            start_date=date(2025, 7, 1),
            end_date=date(2026, 1, 1),
        ),
    )
    with OOSRegistry(tmp_path / "oos.duckdb") as registry:
        registry.create_plan(make_plan("scope-a", "plan", event_id="create-a"))
        registry.create_plan(
            make_plan(
                "scope-b",
                "plan",
                event_id="create-b",
                data_start=date(2025, 1, 1),
                data_end=date(2027, 1, 1),
                windows=scope_b_windows,
            )
        )

        with pytest.raises(PermissionError):
            registry.assert_training_range_allowed(
                "scope-a", "plan", date(2025, 1, 1), date(2025, 7, 1)
            )
        registry.assert_training_range_allowed(
            "scope-b", "plan", date(2025, 1, 1), date(2025, 7, 1)
        )


def test_holdout_rejects_all_ordinary_operations_and_no_release_api_exists(
    tmp_path: Path,
) -> None:
    with OOSRegistry(tmp_path / "oos.duckdb") as registry:
        registry.create_plan(make_plan(event_id="create"))
        with pytest.raises(ValueError, match="SEALED_HOLDOUT"):
            registry.mark_consumed(
                "scope-a",
                "plan-a",
                SEALED_HOLDOUT_WINDOW_ID,
                audit("consume-holdout", 1),
            )
        with pytest.raises(ValueError, match="SEALED_HOLDOUT"):
            registry.authorize_training(
                "scope-a",
                "plan-a",
                SEALED_HOLDOUT_WINDOW_ID,
                audit("train-holdout", 2),
            )
        with pytest.raises(ValueError, match="SEALED_HOLDOUT"):
            registry.register_test_result(
                "scope-a",
                "plan-a",
                SEALED_HOLDOUT_WINDOW_ID,
                make_result(date(2025, 1, 1), date(2026, 1, 1)),
                audit("test-holdout", 3),
                (("strategy", "1"),),
            )

        assert not hasattr(registry, "unseal")
        assert not hasattr(registry, "release_holdout")
        assert not hasattr(registry, "delete_plan")
        assert not hasattr(registry, "reset_window")


def test_persistence_preserves_plans_records_and_transition_history(
    tmp_path: Path,
) -> None:
    database = tmp_path / "oos.duckdb"
    with OOSRegistry(database) as registry:
        registry.create_plan(make_plan(event_id="create"))
        register_first(registry, "scope-a", "plan-a", "tested")
        registry.mark_consumed("scope-a", "plan-a", "w1", audit("consumed", 2))

    with OOSRegistry(database) as reopened:
        plan = reopened.get_plan("scope-a", "plan-a")
        record = reopened.get_test_record("scope-a", "plan-a", "w1")
        history = reopened.transition_history("scope-a", "plan-a")

        assert plan.oos_windows[0].state is OOSWindowState.CONSUMED
        assert record.backtest_run_id == "run-1"
        assert [item.event_id for item in history] == ["create", "tested", "consumed"]
        assert [item.event_type for item in history] == [
            "PLAN_CREATED",
            "TEST_RESULT_REGISTERED",
            "RESULT_CONSUMED",
        ]
