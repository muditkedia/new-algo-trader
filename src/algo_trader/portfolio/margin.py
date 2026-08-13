"""Portfolio-owned abstraction for explicit margin requirements."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from algo_trader.portfolio.models import (
    AllocationCandidate,
    MarginRequirementQuote,
    PortfolioState,
)


@runtime_checkable
class MarginRequirementProvider(Protocol):
    """Structural contract for candidate-specific required-margin quotes."""

    def quote(
        self,
        candidate: AllocationCandidate,
        state: PortfolioState,
    ) -> MarginRequirementQuote:
        """Return the explicit margin required under the supplied state."""
        ...
