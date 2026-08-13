"""Small structural contracts at the two ML system boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from algo_trader.domain import MLScore, Signal
from algo_trader.ml.models import OptimizationEvaluationRange
from algo_trader.reporting import ReportBundle


@runtime_checkable
class TradeScorer(Protocol):
    """Advisory Trade Meta-Model scoring contract; it never rejects."""

    def score(self, signal: Signal) -> MLScore:
        """Return exactly one advisory score for a generated signal."""


@runtime_checkable
class StrategyParameterEvaluator(Protocol):
    """Caller-supplied deterministic strategy/report evaluation contract."""

    def evaluate(
        self,
        parameters: Mapping[str, int | float | str | bool],
        evaluation_ranges: tuple[OptimizationEvaluationRange, ...],
    ) -> tuple[ReportBundle, ...]:
        """Return exactly one report for every requested allowed range."""
