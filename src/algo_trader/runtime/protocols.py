"""Small structural boundaries supplied to Runtime by future strategy composition."""

from collections.abc import Mapping
from datetime import datetime
from typing import Protocol, runtime_checkable

import polars as pl

from algo_trader.broker import BrokerMarketTick
from algo_trader.portfolio import AllocationDecision
from algo_trader.runtime.models import RuntimeSubmissionRecord, RuntimeTradePlan


@runtime_checkable
class RuntimePlanProvider(Protocol):
    """Future boundary for already-generated and already-scored trade plans."""

    def plans_for_cycle(
        self,
        completed_candles: Mapping[str, pl.DataFrame],
        decision_at: datetime,
    ) -> tuple[RuntimeTradePlan, ...]: ...


@runtime_checkable
class RuntimeExecutionGateway(Protocol):
    """Broker-neutral entry gateway seam used by RuntimeService."""

    def submit_allocated_entry(
        self,
        plan: RuntimeTradePlan,
        decision: AllocationDecision,
        intended_at: datetime,
    ) -> RuntimeSubmissionRecord | None: ...

    def on_market_tick(self, tick: BrokerMarketTick) -> None: ...

    def request_strategy_exit(self, subject: object, requested_at: datetime) -> object: ...

    def force_square_off(self, *args: object, **kwargs: object) -> object: ...

    def reconcile(self, reconciled_at: datetime) -> object: ...
