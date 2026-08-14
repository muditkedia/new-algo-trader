"""Production Strategy -> ML -> RuntimeTradePlan composition boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime

import polars as pl

from algo_trader.domain import MLScore, Signal
from algo_trader.ml import TradeScorer
from algo_trader.runtime.models import RuntimeTradePlan
from algo_trader.strategies import Strategy


class ScoredStrategyPlanProvider:
    """Generate decision-time signals, score first, then build executable plans."""

    def __init__(
        self,
        strategy: Strategy,
        scorer: TradeScorer,
        plan_builder: Callable[[Signal, MLScore], RuntimeTradePlan | None],
    ) -> None:
        self._strategy = strategy
        self._scorer = scorer
        self._plan_builder = plan_builder

    def plans_for_cycle(
        self,
        completed_candles: Mapping[str, pl.DataFrame],
        decision_at: datetime,
    ) -> tuple[RuntimeTradePlan, ...]:
        if decision_at.tzinfo is None or decision_at.utcoffset() is None:
            raise ValueError("decision_at must be timezone-aware")
        output: list[RuntimeTradePlan] = []
        for symbol in sorted(completed_candles):
            signals = self._strategy.generate_signals(completed_candles[symbol])
            for signal in signals:
                if signal.timestamp != decision_at:
                    continue
                score = self._scorer.score(signal)
                plan = self._plan_builder(signal, score)
                if plan is not None:
                    output.append(plan)
        return tuple(
            sorted(
                output,
                key=lambda plan: (
                    -plan.candidate.ml_score.quality_score,
                    plan.candidate.identity,
                ),
            )
        )
