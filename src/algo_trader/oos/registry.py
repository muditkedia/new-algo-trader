"""Transactional DuckDB registry for research-scope OOS governance."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb

from algo_trader.backtest import BacktestRunResult
from algo_trader.oos.fingerprint import fingerprint_backtest_result
from algo_trader.oos.models import (
    OOSAuditContext,
    OOSDateRange,
    OOSPlan,
    OOSTestRecord,
    OOSTransitionRecord,
    OOSWindow,
    OOSWindowState,
    normalize_strategy_versions,
)

MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")


class OOSRegistry:
    """Persistent OOS state keyed by composite research-scope/plan identity.

    Plan IDs, window IDs, and backtest run IDs are unique only within their
    composite plan. Audit event IDs are registry-global to make duplicate
    state-change submission unambiguous.
    """

    def __init__(self, database_path: str | Path) -> None:
        if not isinstance(database_path, str | Path):
            raise TypeError("database_path must be a string or Path")
        self._connection = duckdb.connect(str(database_path))
        self._initialize_schema()

    def close(self) -> None:
        """Close the registry connection without changing governance state."""
        self._connection.close()

    def __enter__(self) -> OOSRegistry:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def create_plan(self, plan: OOSPlan) -> OOSPlan:
        """Persist one plan, rejecting overlapping horizons in the same scope."""
        if not isinstance(plan, OOSPlan):
            raise TypeError("plan must be an OOSPlan")
        if any(
            window.state is not OOSWindowState.AVAILABLE
            for window in plan.oos_windows
        ):
            raise ValueError("new plans require all ordinary OOS windows to be AVAILABLE")
        with self._transaction():
            self._assert_event_id_available(plan.creation_audit.event_id)
            existing_scope_plans = self._connection.execute(
                "SELECT COUNT(*) FROM oos_plans WHERE research_scope_id = ?",
                [plan.research_scope_id],
            ).fetchone()[0]
            binding_rows = self._connection.execute(
                """SELECT strategy_id FROM oos_research_scope_strategies
                   WHERE research_scope_id = ? ORDER BY ordinal""",
                [plan.research_scope_id],
            ).fetchall()
            existing_binding = tuple(row[0] for row in binding_rows)
            if existing_scope_plans and not existing_binding:
                raise RuntimeError(
                    "legacy OOS research scope has no explicit strategy binding; "
                    "recreate the v1 registry with explicit v2 lineage"
                )
            if existing_binding and existing_binding != plan.strategy_ids:
                raise ValueError("research scope strategy binding cannot drift")
            overlap = self._connection.execute(
                """
                SELECT plan_id
                FROM oos_plans
                WHERE research_scope_id = ?
                  AND NOT (data_end_exclusive <= ? OR ? <= data_start_date)
                LIMIT 1
                """,
                [
                    plan.research_scope_id,
                    plan.data_start_date,
                    plan.data_end_exclusive,
                ],
            ).fetchone()
            if overlap is not None:
                raise ValueError(
                    "research scope already has a plan with an overlapping data horizon"
                )

            audit = plan.creation_audit
            self._connection.execute(
                """
                INSERT INTO oos_plans VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    plan.research_scope_id,
                    plan.plan_id,
                    plan.protocol_version,
                    plan.data_start_date,
                    plan.data_end_exclusive,
                    plan.development_start_date,
                    plan.development_end_exclusive,
                    plan.sealed_holdout_start_date,
                    plan.sealed_holdout_end_exclusive,
                    audit.event_id,
                    audit.occurred_at.isoformat(),
                    audit.git_commit,
                ],
            )
            if not existing_binding:
                for ordinal, strategy_id in enumerate(plan.strategy_ids):
                    self._connection.execute(
                        "INSERT INTO oos_research_scope_strategies VALUES (?, ?, ?)",
                        [plan.research_scope_id, strategy_id, ordinal],
                    )
            for ordinal, window in enumerate(
                (*plan.oos_windows, plan.sealed_holdout)
            ):
                self._connection.execute(
                    """
                    INSERT INTO oos_windows VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        plan.research_scope_id,
                        plan.plan_id,
                        window.window_id,
                        window.start_date,
                        window.end_date,
                        window.state.value,
                        window is plan.sealed_holdout,
                        ordinal,
                    ],
                )
            self._insert_transition(
                audit=audit,
                research_scope_id=plan.research_scope_id,
                plan_id=plan.plan_id,
                window_id=None,
                from_state=None,
                to_state=None,
                event_type="PLAN_CREATED",
            )
        return self.get_plan(plan.research_scope_id, plan.plan_id)

    def get_plan(self, research_scope_id: str, plan_id: str) -> OOSPlan:
        """Reconstruct one immutable plan and its current window states."""
        row = self._plan_row(research_scope_id, plan_id)
        window_rows = self._connection.execute(
            """
            SELECT window_id, start_date, end_date, state, is_holdout
            FROM oos_windows
            WHERE research_scope_id = ? AND plan_id = ?
            ORDER BY ordinal
            """,
            [research_scope_id, plan_id],
        ).fetchall()
        windows = tuple(
            OOSWindow(
                window_id=window_id,
                start_date=start_date,
                end_date=end_date,
                state=OOSWindowState(state),
            )
            for window_id, start_date, end_date, state, _ in window_rows
        )
        ordinary = tuple(
            window
            for window, database_row in zip(windows, window_rows, strict=True)
            if not database_row[4]
        )
        holdouts = tuple(
            window
            for window, database_row in zip(windows, window_rows, strict=True)
            if database_row[4]
        )
        if len(holdouts) != 1:
            raise RuntimeError("persisted OOS plan must contain exactly one holdout")
        binding_rows = self._connection.execute(
            """SELECT strategy_id FROM oos_research_scope_strategies
               WHERE research_scope_id = ? ORDER BY ordinal""",
            [research_scope_id],
        ).fetchall()
        if not binding_rows:
            raise RuntimeError(
                "legacy OOS plan has no explicit strategy binding; "
                "recreate the v1 registry with explicit v2 lineage"
            )
        return OOSPlan(
            research_scope_id=row[0],
            plan_id=row[1],
            protocol_version=row[2],
            strategy_ids=tuple(item[0] for item in binding_rows),
            data_start_date=row[3],
            data_end_exclusive=row[4],
            development_start_date=row[5],
            development_end_exclusive=row[6],
            sealed_holdout_start_date=row[7],
            sealed_holdout_end_exclusive=row[8],
            oos_windows=ordinary,
            sealed_holdout=holdouts[0],
            creation_audit=OOSAuditContext(
                event_id=row[9],
                occurred_at=datetime.fromisoformat(row[10]),
                git_commit=row[11],
            ),
        )

    def next_testable_window(
        self,
        research_scope_id: str,
        plan_id: str,
    ) -> OOSWindow | None:
        """Return only the earliest AVAILABLE window with trained predecessors."""
        plan = self.get_plan(research_scope_id, plan_id)
        for window in plan.oos_windows:
            if window.state is OOSWindowState.AVAILABLE:
                return window
            if window.state is not OOSWindowState.TRAINING_ALLOWED:
                return None
        return None

    def register_test_result(
        self,
        research_scope_id: str,
        plan_id: str,
        window_id: str,
        result: BacktestRunResult,
        audit_context: OOSAuditContext,
        tested_strategy_versions: object,
    ) -> OOSTestRecord:
        """Register exactly one complete result for the current testable window."""
        if not isinstance(result, BacktestRunResult):
            raise TypeError("result must be a BacktestRunResult")
        if not isinstance(audit_context, OOSAuditContext):
            raise TypeError("audit_context must be an OOSAuditContext")
        attestation = normalize_strategy_versions(tested_strategy_versions)
        plan = self.get_plan(research_scope_id, plan_id)
        outside_scope = sorted(
            {strategy_id for strategy_id, _ in attestation} - set(plan.strategy_ids)
        )
        if outside_scope:
            raise ValueError(
                "tested strategy IDs are outside the research scope binding: "
                + ",".join(outside_scope)
            )
        result_versions = tuple(sorted(result.strategy_versions))
        if result_versions:
            if result_versions != attestation:
                raise ValueError(
                    "Backtest strategy_versions must exactly equal tested_strategy_versions"
                )
        elif any(
            (
                result.request_results,
                result.actual_trade_records,
                result.shadow_trade_records,
            )
        ):
            raise ValueError(
                "empty Backtest strategy_versions require a completely empty result"
            )
        window = self._ordinary_window(research_scope_id, plan_id, window_id)
        current = self.next_testable_window(research_scope_id, plan_id)
        if current is None or current.window_id != window.window_id:
            raise ValueError("window is not the current testable OOS window")
        self.assert_oos_test_range_allowed(
            research_scope_id,
            plan_id,
            window.start_date,
            window.end_date,
        )
        expected_start = datetime.combine(window.start_date, time.min, MARKET_TIMEZONE)
        expected_end = datetime.combine(window.end_date, time.min, MARKET_TIMEZONE)
        if (
            result.window_start != expected_start
            or result.window_start.tzinfo != MARKET_TIMEZONE
        ):
            raise ValueError("backtest window_start must exactly match the OOS window")
        if result.window_end != expected_end or result.window_end.tzinfo != MARKET_TIMEZONE:
            raise ValueError("backtest window_end must exactly match the OOS window")

        record = OOSTestRecord(
            research_scope_id=research_scope_id,
            plan_id=plan_id,
            window_id=window_id,
            backtest_run_id=result.run_id,
            backtest_git_commit=result.git_commit,
            backtester_version=result.backtester_version,
            backtest_window_start=result.window_start,
            backtest_window_end=result.window_end,
            cost_policy_id=result.cost_policy_id,
            brokerage_plan=result.brokerage_plan.value,
            symbols=result.symbols,
            scope_strategy_ids=plan.strategy_ids,
            tested_strategy_versions=attestation,
            strategy_versions=result.strategy_versions,
            ml_model_versions=result.ml_model_versions,
            result_fingerprint=fingerprint_backtest_result(result),
            registration_audit=audit_context,
        )
        with self._transaction():
            self._assert_event_id_available(audit_context.event_id)
            duplicate_run = self._connection.execute(
                """
                SELECT 1 FROM oos_test_records
                WHERE research_scope_id = ? AND plan_id = ? AND backtest_run_id = ?
                """,
                [research_scope_id, plan_id, result.run_id],
            ).fetchone()
            if duplicate_run is not None:
                raise ValueError("backtest run_id is already registered in this plan")
            persisted_state = self._window_state(
                research_scope_id,
                plan_id,
                window_id,
            )
            if persisted_state is not OOSWindowState.AVAILABLE:
                raise ValueError("only an AVAILABLE window may be tested")
            duplicate_window = self._connection.execute(
                """
                SELECT 1 FROM oos_test_records
                WHERE research_scope_id = ? AND plan_id = ? AND window_id = ?
                """,
                [research_scope_id, plan_id, window_id],
            ).fetchone()
            if duplicate_window is not None:
                raise ValueError("OOS window has already been tested")
            self._connection.execute(
                """
                INSERT INTO oos_test_records (
                    research_scope_id, plan_id, window_id, backtest_run_id,
                    backtest_git_commit, backtester_version, backtest_window_start,
                    backtest_window_end, cost_policy_id, brokerage_plan, symbols_json,
                    strategy_versions_json, ml_model_versions_json, result_fingerprint,
                    registration_event_id, registration_occurred_at,
                    registration_git_commit, scope_strategy_ids_json,
                    tested_strategy_versions_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    record.research_scope_id,
                    record.plan_id,
                    record.window_id,
                    record.backtest_run_id,
                    record.backtest_git_commit,
                    record.backtester_version,
                    record.backtest_window_start.isoformat(),
                    record.backtest_window_end.isoformat(),
                    record.cost_policy_id,
                    record.brokerage_plan,
                    _canonical_json(record.symbols),
                    _canonical_json(record.strategy_versions),
                    _canonical_json(record.ml_model_versions),
                    record.result_fingerprint,
                    audit_context.event_id,
                    audit_context.occurred_at.isoformat(),
                    audit_context.git_commit,
                    _canonical_json(record.scope_strategy_ids),
                    _canonical_json(record.tested_strategy_versions),
                ],
            )
            self._set_window_state(
                research_scope_id,
                plan_id,
                window_id,
                OOSWindowState.TESTED,
            )
            self._insert_transition(
                audit=audit_context,
                research_scope_id=research_scope_id,
                plan_id=plan_id,
                window_id=window_id,
                from_state=OOSWindowState.AVAILABLE,
                to_state=OOSWindowState.TESTED,
                event_type="TEST_RESULT_REGISTERED",
            )
        return record

    def mark_consumed(
        self,
        research_scope_id: str,
        plan_id: str,
        window_id: str,
        audit_context: OOSAuditContext,
    ) -> OOSWindow:
        """Apply the only valid TESTED to CONSUMED transition."""
        return self._transition_window(
            research_scope_id,
            plan_id,
            window_id,
            expected=OOSWindowState.TESTED,
            target=OOSWindowState.CONSUMED,
            audit_context=audit_context,
            event_type="RESULT_CONSUMED",
        )

    def authorize_training(
        self,
        research_scope_id: str,
        plan_id: str,
        window_id: str,
        audit_context: OOSAuditContext,
    ) -> OOSWindow:
        """Apply the only valid CONSUMED to TRAINING_ALLOWED transition."""
        return self._transition_window(
            research_scope_id,
            plan_id,
            window_id,
            expected=OOSWindowState.CONSUMED,
            target=OOSWindowState.TRAINING_ALLOWED,
            audit_context=audit_context,
            event_type="TRAINING_AUTHORIZED",
        )

    def training_allowed_ranges(
        self,
        research_scope_id: str,
        plan_id: str,
    ) -> tuple[OOSDateRange, ...]:
        """Return deterministic merged development/training-approved ranges."""
        plan = self.get_plan(research_scope_id, plan_id)
        ranges = [
            OOSDateRange(
                start_date=plan.development_start_date,
                end_date=plan.development_end_exclusive,
            )
        ]
        ranges.extend(
            OOSDateRange(start_date=window.start_date, end_date=window.end_date)
            for window in plan.oos_windows
            if window.state is OOSWindowState.TRAINING_ALLOWED
        )
        merged: list[OOSDateRange] = []
        for selected in sorted(ranges, key=lambda item: item.start_date):
            if merged and merged[-1].end_date == selected.start_date:
                previous = merged.pop()
                merged.append(
                    OOSDateRange(
                        start_date=previous.start_date,
                        end_date=selected.end_date,
                    )
                )
            else:
                merged.append(selected)
        return tuple(merged)

    def assert_training_range_allowed(
        self,
        research_scope_id: str,
        plan_id: str,
        start_date: date,
        end_date: date,
    ) -> None:
        """Reject unless the entire range is development or training-approved."""
        _validate_date_range(start_date, end_date)
        if not any(
            allowed.start_date <= start_date and end_date <= allowed.end_date
            for allowed in self.training_allowed_ranges(research_scope_id, plan_id)
        ):
            raise PermissionError(
                "requested range includes OOS data not authorized for training"
            )

    def assert_oos_test_range_allowed(
        self,
        research_scope_id: str,
        plan_id: str,
        start_date: date,
        end_date: date,
    ) -> None:
        """Require an exact match to the current testable ordinary OOS window."""
        _validate_date_range(start_date, end_date)
        current = self.next_testable_window(research_scope_id, plan_id)
        if (
            current is None
            or current.start_date != start_date
            or current.end_date != end_date
        ):
            raise PermissionError(
                "requested range must exactly match the current testable OOS window"
            )

    def get_test_record(
        self,
        research_scope_id: str,
        plan_id: str,
        window_id: str,
    ) -> OOSTestRecord:
        """Return compact registered result provenance for one window."""
        row = self._connection.execute(
            """
            SELECT * FROM oos_test_records
            WHERE research_scope_id = ? AND plan_id = ? AND window_id = ?
            """,
            [research_scope_id, plan_id, window_id],
        ).fetchone()
        if row is None:
            raise LookupError("no OOS test record for the requested window")
        if row[17] is None or row[18] is None:
            raise RuntimeError(
                "legacy OOS test record lacks explicit v2 strategy lineage"
            )
        return OOSTestRecord(
            research_scope_id=row[0],
            plan_id=row[1],
            window_id=row[2],
            backtest_run_id=row[3],
            backtest_git_commit=row[4],
            backtester_version=row[5],
            backtest_window_start=datetime.fromisoformat(row[6]),
            backtest_window_end=datetime.fromisoformat(row[7]),
            cost_policy_id=row[8],
            brokerage_plan=row[9],
            symbols=tuple(json.loads(row[10])),
            strategy_versions=tuple(tuple(item) for item in json.loads(row[11])),
            ml_model_versions=tuple(json.loads(row[12])),
            result_fingerprint=row[13],
            registration_audit=OOSAuditContext(
                event_id=row[14],
                occurred_at=datetime.fromisoformat(row[15]),
                git_commit=row[16],
            ),
            scope_strategy_ids=tuple(json.loads(row[17])),
            tested_strategy_versions=tuple(
                tuple(item) for item in json.loads(row[18])
            ),
        )

    def transition_history(
        self,
        research_scope_id: str,
        plan_id: str,
    ) -> tuple[OOSTransitionRecord, ...]:
        """Return deterministic caller-timestamp/event-ID ordered audit history."""
        rows = self._connection.execute(
            """
            SELECT event_id, occurred_at, git_commit, research_scope_id, plan_id,
                   window_id, from_state, to_state, event_type
            FROM oos_transitions
            WHERE research_scope_id = ? AND plan_id = ?
            """,
            [research_scope_id, plan_id],
        ).fetchall()
        records = tuple(
            OOSTransitionRecord(
                event_id=row[0],
                occurred_at=datetime.fromisoformat(row[1]),
                git_commit=row[2],
                research_scope_id=row[3],
                plan_id=row[4],
                window_id=row[5],
                from_state=OOSWindowState(row[6]) if row[6] is not None else None,
                to_state=OOSWindowState(row[7]) if row[7] is not None else None,
                event_type=row[8],
            )
            for row in rows
        )
        return tuple(sorted(records, key=lambda item: (item.occurred_at, item.event_id)))

    def _transition_window(
        self,
        research_scope_id: str,
        plan_id: str,
        window_id: str,
        *,
        expected: OOSWindowState,
        target: OOSWindowState,
        audit_context: OOSAuditContext,
        event_type: str,
    ) -> OOSWindow:
        if not isinstance(audit_context, OOSAuditContext):
            raise TypeError("audit_context must be an OOSAuditContext")
        with self._transaction():
            self._assert_event_id_available(audit_context.event_id)
            self._ordinary_window(research_scope_id, plan_id, window_id)
            current = self._window_state(research_scope_id, plan_id, window_id)
            if current is not expected:
                raise ValueError(
                    f"window transition requires {expected.value}, found {current.value}"
                )
            self._set_window_state(research_scope_id, plan_id, window_id, target)
            self._insert_transition(
                audit=audit_context,
                research_scope_id=research_scope_id,
                plan_id=plan_id,
                window_id=window_id,
                from_state=expected,
                to_state=target,
                event_type=event_type,
            )
        return self._ordinary_window(research_scope_id, plan_id, window_id)

    def _ordinary_window(
        self,
        research_scope_id: str,
        plan_id: str,
        window_id: str,
    ) -> OOSWindow:
        row = self._connection.execute(
            """
            SELECT start_date, end_date, state, is_holdout
            FROM oos_windows
            WHERE research_scope_id = ? AND plan_id = ? AND window_id = ?
            """,
            [research_scope_id, plan_id, window_id],
        ).fetchone()
        if row is None:
            raise LookupError("unknown OOS window")
        if row[3]:
            raise ValueError("SEALED_HOLDOUT cannot use ordinary OOS operations")
        return OOSWindow(
            window_id=window_id,
            start_date=row[0],
            end_date=row[1],
            state=OOSWindowState(row[2]),
        )

    def _plan_row(self, research_scope_id: str, plan_id: str) -> tuple:
        row = self._connection.execute(
            """
            SELECT * FROM oos_plans
            WHERE research_scope_id = ? AND plan_id = ?
            """,
            [research_scope_id, plan_id],
        ).fetchone()
        if row is None:
            raise LookupError("unknown research scope / OOS plan")
        return row

    def _window_state(
        self,
        research_scope_id: str,
        plan_id: str,
        window_id: str,
    ) -> OOSWindowState:
        row = self._connection.execute(
            """
            SELECT state FROM oos_windows
            WHERE research_scope_id = ? AND plan_id = ? AND window_id = ?
            """,
            [research_scope_id, plan_id, window_id],
        ).fetchone()
        if row is None:
            raise LookupError("unknown OOS window")
        return OOSWindowState(row[0])

    def _set_window_state(
        self,
        research_scope_id: str,
        plan_id: str,
        window_id: str,
        state: OOSWindowState,
    ) -> None:
        self._connection.execute(
            """
            UPDATE oos_windows SET state = ?
            WHERE research_scope_id = ? AND plan_id = ? AND window_id = ?
            """,
            [state.value, research_scope_id, plan_id, window_id],
        )

    def _assert_event_id_available(self, event_id: str) -> None:
        exists = self._connection.execute(
            "SELECT 1 FROM oos_transitions WHERE event_id = ?",
            [event_id],
        ).fetchone()
        if exists is not None:
            raise ValueError("audit event_id is already registered")

    def _insert_transition(
        self,
        *,
        audit: OOSAuditContext,
        research_scope_id: str,
        plan_id: str,
        window_id: str | None,
        from_state: OOSWindowState | None,
        to_state: OOSWindowState | None,
        event_type: str,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO oos_transitions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                audit.event_id,
                audit.occurred_at.isoformat(),
                audit.git_commit,
                research_scope_id,
                plan_id,
                window_id,
                from_state.value if from_state is not None else None,
                to_state.value if to_state is not None else None,
                event_type,
            ],
        )

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self._connection.execute("BEGIN TRANSACTION")
        try:
            yield
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")

    def _initialize_schema(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS oos_plans (
                research_scope_id VARCHAR NOT NULL,
                plan_id VARCHAR NOT NULL,
                protocol_version VARCHAR NOT NULL,
                data_start_date DATE NOT NULL,
                data_end_exclusive DATE NOT NULL,
                development_start_date DATE NOT NULL,
                development_end_exclusive DATE NOT NULL,
                sealed_holdout_start_date DATE NOT NULL,
                sealed_holdout_end_exclusive DATE NOT NULL,
                creation_event_id VARCHAR NOT NULL,
                creation_occurred_at VARCHAR NOT NULL,
                creation_git_commit VARCHAR NOT NULL,
                PRIMARY KEY (research_scope_id, plan_id)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS oos_research_scope_strategies (
                research_scope_id VARCHAR NOT NULL,
                strategy_id VARCHAR NOT NULL,
                ordinal INTEGER NOT NULL,
                PRIMARY KEY (research_scope_id, strategy_id),
                UNIQUE (research_scope_id, ordinal)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS oos_windows (
                research_scope_id VARCHAR NOT NULL,
                plan_id VARCHAR NOT NULL,
                window_id VARCHAR NOT NULL,
                start_date DATE NOT NULL,
                end_date DATE NOT NULL,
                state VARCHAR NOT NULL,
                is_holdout BOOLEAN NOT NULL,
                ordinal INTEGER NOT NULL,
                PRIMARY KEY (research_scope_id, plan_id, window_id)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS oos_test_records (
                research_scope_id VARCHAR NOT NULL,
                plan_id VARCHAR NOT NULL,
                window_id VARCHAR NOT NULL,
                backtest_run_id VARCHAR NOT NULL,
                backtest_git_commit VARCHAR NOT NULL,
                backtester_version VARCHAR NOT NULL,
                backtest_window_start VARCHAR NOT NULL,
                backtest_window_end VARCHAR NOT NULL,
                cost_policy_id VARCHAR NOT NULL,
                brokerage_plan VARCHAR NOT NULL,
                symbols_json VARCHAR NOT NULL,
                strategy_versions_json VARCHAR NOT NULL,
                ml_model_versions_json VARCHAR NOT NULL,
                result_fingerprint VARCHAR NOT NULL,
                registration_event_id VARCHAR NOT NULL,
                registration_occurred_at VARCHAR NOT NULL,
                registration_git_commit VARCHAR NOT NULL,
                PRIMARY KEY (research_scope_id, plan_id, window_id),
                UNIQUE (research_scope_id, plan_id, backtest_run_id)
            )
            """
        )
        self._connection.execute(
            "ALTER TABLE oos_test_records ADD COLUMN IF NOT EXISTS "
            "scope_strategy_ids_json VARCHAR"
        )
        self._connection.execute(
            "ALTER TABLE oos_test_records ADD COLUMN IF NOT EXISTS "
            "tested_strategy_versions_json VARCHAR"
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS oos_transitions (
                event_id VARCHAR PRIMARY KEY,
                occurred_at VARCHAR NOT NULL,
                git_commit VARCHAR NOT NULL,
                research_scope_id VARCHAR NOT NULL,
                plan_id VARCHAR NOT NULL,
                window_id VARCHAR,
                from_state VARCHAR,
                to_state VARCHAR,
                event_type VARCHAR NOT NULL
            )
            """
        )


def _validate_date_range(start_date: date, end_date: date) -> None:
    if isinstance(start_date, datetime) or not isinstance(start_date, date):
        raise TypeError("start_date must be a date")
    if isinstance(end_date, datetime) or not isinstance(end_date, date):
        raise TypeError("end_date must be a date")
    if start_date >= end_date:
        raise ValueError("start_date must be earlier than end_date")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
