"""Structural contract implemented by trading strategies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import polars as pl

from algo_trader.domain import Signal


@runtime_checkable
class Strategy(Protocol):
    """Minimal mode-independent strategy contract.

    Runtime protocol checks confirm only that named members exist. Candle and
    signal semantics remain the responsibility of input/domain validation.
    """

    strategy_id: str
    strategy_version: str
    parameters: Mapping[str, Any]
    warmup_bars: int

    def generate_signals(self, candles: pl.DataFrame) -> list[Signal]:
        """Generate signals for one chronologically ordered symbol."""
        ...

