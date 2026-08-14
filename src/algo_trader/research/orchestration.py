"""Small reusable research adapter and score-before-sizing orchestration."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from algo_trader.domain import MLScore, Signal
from algo_trader.ml import TradeScorer
from algo_trader.strategies import Strategy


@dataclass(frozen=True, slots=True)
class ResearchStrategySpec[RequestT]:
    """Only strategy-specific inputs required by generic research orchestration."""

    research_scope_id: str
    plan_id: str
    output_slug: str
    strategy_factory: Callable[[], Strategy]
    request_builder: Callable[[Signal, MLScore], RequestT | None]

    def create_strategy(self) -> Strategy:
        strategy = self.strategy_factory()
        if not isinstance(strategy, Strategy):
            raise TypeError("strategy_factory must return a Strategy-compatible object")
        return strategy


def score_and_build_requests[RequestT](
    signals: Iterable[Signal],
    scorer: TradeScorer,
    request_builder: Callable[[Signal, MLScore], RequestT | None],
) -> tuple[RequestT, ...]:
    """Score each signal before any builder selects quantity or order notional."""
    output: list[RequestT] = []
    for signal in signals:
        score = scorer.score(signal)
        request = request_builder(signal, score)
        if request is not None:
            output.append(request)
    return tuple(output)
