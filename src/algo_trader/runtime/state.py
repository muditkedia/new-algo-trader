"""Compact DuckDB persistence for restart-safe Runtime orchestration."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from threading import RLock

import duckdb
from pydantic import BaseModel

from algo_trader.broker import BrokerOrderSnapshot, BrokerTradeFill
from algo_trader.portfolio import AllocationDecision, CapitalReservation
from algo_trader.runtime.models import (
    RuntimeEvent,
    RuntimeOrderLifecycle,
    RuntimeOrderRecord,
    RuntimePositionRecord,
    RuntimeSessionRecord,
    RuntimeTradePlan,
    RuntimeTradeRecord,
)


class RuntimeStateStore:
    """Caller-owned durable ledger; no credentials or tokens are accepted."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = duckdb.connect(str(self.path))
        self._lock = RLock()
        self._closed = False
        self._initialize()

    def _initialize(self) -> None:
        statements = (
            """CREATE TABLE IF NOT EXISTS runtime_sessions (
                runtime_session_id VARCHAR PRIMARY KEY,
                trading_date DATE NOT NULL,
                mode VARCHAR NOT NULL,
                phase VARCHAR NOT NULL,
                ended_at TIMESTAMPTZ,
                record_json VARCHAR NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS runtime_events (
                runtime_session_id VARCHAR NOT NULL,
                sequence BIGINT NOT NULL,
                occurred_at TIMESTAMPTZ NOT NULL,
                event_type VARCHAR NOT NULL,
                record_json VARCHAR NOT NULL,
                PRIMARY KEY (runtime_session_id, sequence)
            )""",
            """CREATE TABLE IF NOT EXISTS runtime_orders (
                runtime_session_id VARCHAR NOT NULL,
                client_order_id VARCHAR PRIMARY KEY,
                candidate_fingerprint VARCHAR NOT NULL,
                leg VARCHAR NOT NULL,
                lifecycle VARCHAR NOT NULL,
                broker_order_tag VARCHAR,
                broker_order_id VARCHAR,
                unique_order_id VARCHAR,
                record_json VARCHAR NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS runtime_allocations (
                runtime_session_id VARCHAR NOT NULL,
                candidate_fingerprint VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                plan_json VARCHAR NOT NULL,
                decision_json VARCHAR NOT NULL,
                PRIMARY KEY (runtime_session_id, candidate_fingerprint)
            )""",
            """CREATE TABLE IF NOT EXISTS broker_order_snapshots (
                runtime_session_id VARCHAR NOT NULL,
                client_order_id VARCHAR NOT NULL,
                observed_at TIMESTAMPTZ NOT NULL,
                record_json VARCHAR NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS broker_trade_fills (
                runtime_session_id VARCHAR NOT NULL,
                client_order_id VARCHAR NOT NULL,
                fill_key VARCHAR PRIMARY KEY,
                record_json VARCHAR NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS active_reservations (
                runtime_session_id VARCHAR NOT NULL,
                candidate_fingerprint VARCHAR NOT NULL,
                record_json VARCHAR NOT NULL,
                PRIMARY KEY (runtime_session_id, candidate_fingerprint)
            )""",
            """CREATE TABLE IF NOT EXISTS runtime_positions (
                runtime_session_id VARCHAR NOT NULL,
                candidate_fingerprint VARCHAR NOT NULL,
                active BOOLEAN NOT NULL,
                record_json VARCHAR NOT NULL,
                PRIMARY KEY (runtime_session_id, candidate_fingerprint)
            )""",
            """CREATE TABLE IF NOT EXISTS runtime_trades (
                runtime_session_id VARCHAR NOT NULL,
                candidate_fingerprint VARCHAR NOT NULL,
                record_json VARCHAR NOT NULL,
                PRIMARY KEY (runtime_session_id, candidate_fingerprint)
            )""",
        )
        with self._lock:
            for statement in statements:
                self._connection.execute(statement)

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        with self._lock:
            self._require_open()
            self._connection.execute("BEGIN TRANSACTION")
            try:
                yield
            except Exception:
                self._connection.execute("ROLLBACK")
                raise
            else:
                self._connection.execute("COMMIT")

    def create_session(self, session: RuntimeSessionRecord) -> None:
        """Create one caller-identified active session and its first event atomically."""
        _expect(session, RuntimeSessionRecord, "session")
        with self._transaction():
            existing = self._connection.execute(
                "SELECT ended_at FROM runtime_sessions WHERE runtime_session_id = ?",
                [session.runtime_session_id],
            ).fetchone()
            if existing is not None:
                qualifier = "active " if existing[0] is None else ""
                raise ValueError(f"duplicate {qualifier}runtime_session_id")
            self._write_session(session, insert=True)
            self._append_event_unlocked(
                session.runtime_session_id,
                session.started_at,
                "SESSION_STARTED",
                f"mode={session.mode.value}",
            )

    def get_session(self, runtime_session_id: str) -> RuntimeSessionRecord:
        row = self._fetchone(
            "SELECT record_json FROM runtime_sessions WHERE runtime_session_id = ?",
            [runtime_session_id],
        )
        if row is None:
            raise LookupError(f"unknown runtime session: {runtime_session_id}")
        return RuntimeSessionRecord.model_validate_json(row[0])

    def list_unfinished_sessions(
        self, trading_date: object
    ) -> tuple[RuntimeSessionRecord, ...]:
        """Return every unfinished same-day session; callers must resolve ambiguity."""
        rows = self._fetchall(
            """SELECT record_json FROM runtime_sessions
               WHERE trading_date = ? AND ended_at IS NULL ORDER BY runtime_session_id""",
            [trading_date],
        )
        return tuple(RuntimeSessionRecord.model_validate_json(row[0]) for row in rows)

    def find_unfinished_session(self, trading_date: object) -> RuntimeSessionRecord | None:
        sessions = self.list_unfinished_sessions(trading_date)
        if len(sessions) > 1:
            raise RuntimeError("multiple unfinished runtime sessions exist for trading date")
        return sessions[0] if sessions else None

    def update_session(
        self,
        session: RuntimeSessionRecord,
        *,
        occurred_at: datetime,
        event_type: str,
        description: str = "",
    ) -> RuntimeEvent:
        _expect(session, RuntimeSessionRecord, "session")
        with self._transaction():
            self._write_session(session, insert=False)
            return self._append_event_unlocked(
                session.runtime_session_id, occurred_at, event_type, description
            )

    def append_event(
        self,
        runtime_session_id: str,
        occurred_at: datetime,
        event_type: str,
        description: str = "",
    ) -> RuntimeEvent:
        with self._transaction():
            return self._append_event_unlocked(
                runtime_session_id, occurred_at, event_type, description
            )

    def list_events(self, runtime_session_id: str) -> tuple[RuntimeEvent, ...]:
        rows = self._fetchall(
            """SELECT record_json FROM runtime_events
               WHERE runtime_session_id = ? ORDER BY sequence""",
            [runtime_session_id],
        )
        return tuple(RuntimeEvent.model_validate_json(row[0]) for row in rows)

    def record_order_intent(self, order: RuntimeOrderRecord) -> None:
        """Durably persist INTENT_RECORDED and its event before any broker call."""
        _expect(order, RuntimeOrderRecord, "order")
        if order.lifecycle is not RuntimeOrderLifecycle.INTENT_RECORDED:
            raise ValueError("new runtime order must start as INTENT_RECORDED")
        with self._transaction():
            self._connection.execute(
                """INSERT INTO runtime_orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    order.runtime_session_id,
                    order.client_order_id,
                    order.candidate_fingerprint,
                    order.leg.value,
                    order.lifecycle.value,
                    order.broker_order_tag,
                    order.broker_order_id,
                    order.unique_order_id,
                    _json(order),
                ],
            )
            self._append_event_unlocked(
                order.runtime_session_id,
                order.intended_at,
                "ORDER_INTENT_RECORDED",
                f"client_order_id={order.client_order_id};leg={order.leg.value}",
            )

    def save_allocation(
        self,
        runtime_session_id: str,
        candidate_fingerprint: str,
        plan: RuntimeTradePlan,
        decision: AllocationDecision,
        *,
        status: str = "PENDING",
    ) -> None:
        _expect(plan, RuntimeTradePlan, "plan")
        _expect(decision, AllocationDecision, "decision")
        with self._transaction():
            self._connection.execute(
                "INSERT INTO runtime_allocations VALUES (?, ?, ?, ?, ?)",
                [
                    runtime_session_id,
                    candidate_fingerprint,
                    status,
                    _json(plan),
                    _json(decision),
                ],
            )

    def save_allocation_batch(
        self,
        runtime_session_id: str,
        rows: Sequence[tuple[str, RuntimeTradePlan, AllocationDecision]],
        occurred_at: datetime,
    ) -> None:
        """Persist an allocator's complete result and reservations atomically."""
        if not rows:
            raise ValueError("allocation batch must not be empty")
        with self._transaction():
            for fingerprint, plan, decision in rows:
                _expect(plan, RuntimeTradePlan, "plan")
                _expect(decision, AllocationDecision, "decision")
                self._connection.execute(
                    "INSERT INTO runtime_allocations VALUES (?, ?, 'PENDING', ?, ?)",
                    [runtime_session_id, fingerprint, _json(plan), _json(decision)],
                )
                if decision.reservation is not None:
                    self._connection.execute(
                        "INSERT INTO active_reservations VALUES (?, ?, ?)",
                        [runtime_session_id, fingerprint, _json(decision.reservation)],
                    )
                event_type = (
                    "ALLOCATION_APPROVED"
                    if decision.reservation is not None
                    else "CAPACITY_REJECTED"
                )
                description = f"candidate={fingerprint}"
                if decision.reservation is not None:
                    description += f";provider={decision.margin_quote.provider_id}"
                self._append_event_unlocked(
                    runtime_session_id, occurred_at, event_type, description
                )

    def update_allocation_status(
        self,
        runtime_session_id: str,
        candidate_fingerprint: str,
        status: str,
    ) -> None:
        with self._transaction():
            self._connection.execute(
                """UPDATE runtime_allocations SET status = ?
                   WHERE runtime_session_id = ? AND candidate_fingerprint = ?""",
                [status, runtime_session_id, candidate_fingerprint],
            )

    def load_allocations(
        self,
        runtime_session_id: str,
        *,
        status: str | None = None,
    ) -> tuple[tuple[str, RuntimeTradePlan, AllocationDecision, str], ...]:
        if status is None:
            rows = self._fetchall(
                """SELECT candidate_fingerprint, plan_json, decision_json, status
                   FROM runtime_allocations WHERE runtime_session_id = ?
                   ORDER BY candidate_fingerprint""",
                [runtime_session_id],
            )
        else:
            rows = self._fetchall(
                """SELECT candidate_fingerprint, plan_json, decision_json, status
                   FROM runtime_allocations WHERE runtime_session_id = ? AND status = ?
                   ORDER BY candidate_fingerprint""",
                [runtime_session_id, status],
            )
        return tuple(
            (
                str(row[0]),
                RuntimeTradePlan.model_validate_json(row[1]),
                AllocationDecision.model_validate_json(row[2]),
                str(row[3]),
            )
            for row in rows
        )

    def update_order(
        self,
        order: RuntimeOrderRecord,
        *,
        occurred_at: datetime,
        event_type: str,
    ) -> None:
        _expect(order, RuntimeOrderRecord, "order")
        with self._transaction():
            result = self._connection.execute(
                """UPDATE runtime_orders SET lifecycle = ?, broker_order_tag = ?,
                   broker_order_id = ?, unique_order_id = ?, record_json = ?
                   WHERE client_order_id = ? AND runtime_session_id = ?""",
                [
                    order.lifecycle.value,
                    order.broker_order_tag,
                    order.broker_order_id,
                    order.unique_order_id,
                    _json(order),
                    order.client_order_id,
                    order.runtime_session_id,
                ],
            )
            if result.rowcount == 0:
                raise LookupError("runtime order intent is not persisted")
            self._append_event_unlocked(
                order.runtime_session_id,
                occurred_at,
                event_type,
                f"client_order_id={order.client_order_id};state={order.lifecycle.value}",
            )

    def get_order(self, client_order_id: str) -> RuntimeOrderRecord:
        row = self._fetchone(
            "SELECT record_json FROM runtime_orders WHERE client_order_id = ?",
            [client_order_id],
        )
        if row is None:
            raise LookupError(f"unknown runtime order: {client_order_id}")
        return RuntimeOrderRecord.model_validate_json(row[0])

    def list_orders(self, runtime_session_id: str) -> tuple[RuntimeOrderRecord, ...]:
        rows = self._fetchall(
            """SELECT record_json FROM runtime_orders
               WHERE runtime_session_id = ? ORDER BY client_order_id""",
            [runtime_session_id],
        )
        return tuple(RuntimeOrderRecord.model_validate_json(row[0]) for row in rows)

    def record_order_snapshot(
        self,
        runtime_session_id: str,
        client_order_id: str,
        snapshot: BrokerOrderSnapshot,
        observed_at: datetime,
    ) -> None:
        _expect(snapshot, BrokerOrderSnapshot, "snapshot")
        with self._transaction():
            self._connection.execute(
                "INSERT INTO broker_order_snapshots VALUES (?, ?, ?, ?)",
                [runtime_session_id, client_order_id, observed_at, _json(snapshot)],
            )

    def record_broker_fill(
        self,
        runtime_session_id: str,
        client_order_id: str,
        fill: BrokerTradeFill,
    ) -> bool:
        """Persist one normalized fill idempotently without collapsing evidence."""
        _expect(fill, BrokerTradeFill, "fill")
        fill_key = hashlib.sha256(
            f"{fill.broker_order_id}\0{fill.fill_id}".encode()
        ).hexdigest()
        with self._transaction():
            existing = self._connection.execute(
                "SELECT 1 FROM broker_trade_fills WHERE fill_key = ?", [fill_key]
            ).fetchone()
            if existing is not None:
                return False
            self._connection.execute(
                "INSERT INTO broker_trade_fills VALUES (?, ?, ?, ?)",
                [runtime_session_id, client_order_id, fill_key, _json(fill)],
            )
            return True

    def list_broker_fills(self, client_order_id: str) -> tuple[BrokerTradeFill, ...]:
        rows = self._fetchall(
            """SELECT record_json FROM broker_trade_fills
               WHERE client_order_id = ? ORDER BY fill_key""",
            [client_order_id],
        )
        return tuple(BrokerTradeFill.model_validate_json(row[0]) for row in rows)

    def save_reservation(
        self,
        runtime_session_id: str,
        candidate_fingerprint: str,
        reservation: CapitalReservation,
    ) -> None:
        _expect(reservation, CapitalReservation, "reservation")
        with self._transaction():
            self._connection.execute(
                "INSERT INTO active_reservations VALUES (?, ?, ?)",
                [runtime_session_id, candidate_fingerprint, _json(reservation)],
            )

    def delete_reservation(self, runtime_session_id: str, candidate_fingerprint: str) -> None:
        with self._transaction():
            self._connection.execute(
                """DELETE FROM active_reservations
                   WHERE runtime_session_id = ? AND candidate_fingerprint = ?""",
                [runtime_session_id, candidate_fingerprint],
            )

    def load_reservations(self, runtime_session_id: str) -> tuple[CapitalReservation, ...]:
        rows = self._fetchall(
            """SELECT record_json FROM active_reservations
               WHERE runtime_session_id = ? ORDER BY candidate_fingerprint""",
            [runtime_session_id],
        )
        return tuple(CapitalReservation.model_validate_json(row[0]) for row in rows)

    def save_position(self, position: RuntimePositionRecord) -> None:
        _expect(position, RuntimePositionRecord, "position")
        with self._transaction():
            self._connection.execute(
                """INSERT OR REPLACE INTO runtime_positions VALUES (?, ?, TRUE, ?)""",
                [
                    position.runtime_session_id,
                    position.candidate_fingerprint,
                    _json(position),
                ],
            )

    def open_position(self, position: RuntimePositionRecord) -> None:
        """Persist a newly filled position and allocation lifecycle atomically."""
        _expect(position, RuntimePositionRecord, "position")
        with self._transaction():
            self._connection.execute(
                """INSERT OR REPLACE INTO runtime_positions VALUES (?, ?, TRUE, ?)""",
                [
                    position.runtime_session_id,
                    position.candidate_fingerprint,
                    _json(position),
                ],
            )
            self._connection.execute(
                """UPDATE runtime_allocations SET status = 'OPEN'
                   WHERE runtime_session_id = ? AND candidate_fingerprint = ?""",
                [position.runtime_session_id, position.candidate_fingerprint],
            )

    def close_position(self, position: RuntimePositionRecord) -> None:
        _expect(position, RuntimePositionRecord, "position")
        with self._transaction():
            self._connection.execute(
                """UPDATE runtime_positions SET active = FALSE, record_json = ?
                   WHERE runtime_session_id = ? AND candidate_fingerprint = ?""",
                [
                    _json(position),
                    position.runtime_session_id,
                    position.candidate_fingerprint,
                ],
            )

    def load_positions(self, runtime_session_id: str) -> tuple[RuntimePositionRecord, ...]:
        rows = self._fetchall(
            """SELECT record_json FROM runtime_positions
               WHERE runtime_session_id = ? AND active ORDER BY candidate_fingerprint""",
            [runtime_session_id],
        )
        return tuple(RuntimePositionRecord.model_validate_json(row[0]) for row in rows)

    def save_trade(self, trade: RuntimeTradeRecord) -> None:
        _expect(trade, RuntimeTradeRecord, "trade")
        with self._transaction():
            self._connection.execute(
                "INSERT INTO runtime_trades VALUES (?, ?, ?)",
                [trade.runtime_session_id, trade.candidate_fingerprint, _json(trade)],
            )

    def load_trades(self, runtime_session_id: str) -> tuple[RuntimeTradeRecord, ...]:
        rows = self._fetchall(
            """SELECT record_json FROM runtime_trades
               WHERE runtime_session_id = ? ORDER BY candidate_fingerprint""",
            [runtime_session_id],
        )
        return tuple(RuntimeTradeRecord.model_validate_json(row[0]) for row in rows)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def _write_session(self, session: RuntimeSessionRecord, *, insert: bool) -> None:
        values = [
            session.runtime_session_id,
            session.trading_date,
            session.mode.value,
            session.phase.value,
            session.ended_at,
            _json(session),
        ]
        if insert:
            self._connection.execute(
                "INSERT INTO runtime_sessions VALUES (?, ?, ?, ?, ?, ?)", values
            )
        else:
            result = self._connection.execute(
                """UPDATE runtime_sessions SET trading_date = ?, mode = ?, phase = ?,
                   ended_at = ?, record_json = ? WHERE runtime_session_id = ?""",
                values[1:] + values[:1],
            )
            if result.rowcount == 0:
                raise LookupError("runtime session is not persisted")

    def _append_event_unlocked(
        self,
        runtime_session_id: str,
        occurred_at: datetime,
        event_type: str,
        description: str,
    ) -> RuntimeEvent:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) FROM runtime_events WHERE runtime_session_id = ?",
            [runtime_session_id],
        ).fetchone()
        sequence = int(row[0]) + 1
        event = RuntimeEvent(
            runtime_session_id=runtime_session_id,
            sequence=sequence,
            occurred_at=occurred_at,
            event_type=event_type,
            description=description,
        )
        self._connection.execute(
            "INSERT INTO runtime_events VALUES (?, ?, ?, ?, ?)",
            [runtime_session_id, sequence, occurred_at, event_type, _json(event)],
        )
        return event

    def _fetchone(self, query: str, parameters: list[object]) -> tuple[object, ...] | None:
        with self._lock:
            self._require_open()
            return self._connection.execute(query, parameters).fetchone()

    def _fetchall(self, query: str, parameters: list[object]) -> list[tuple[object, ...]]:
        with self._lock:
            self._require_open()
            return self._connection.execute(query, parameters).fetchall()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("runtime state store is closed")


def _json(model: BaseModel) -> str:
    return model.model_dump_json(exclude_computed_fields=True)


def _expect(value: object, expected: type[object], name: str) -> None:
    if not isinstance(value, expected):
        raise TypeError(f"{name} must be a {expected.__name__}")
