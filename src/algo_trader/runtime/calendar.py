"""Explicit caller-owned trading-day boundary; no inferred holiday policy."""

from collections.abc import Iterable
from datetime import date
from typing import Protocol, runtime_checkable


@runtime_checkable
class TradingDayProvider(Protocol):
    """Answer only whether a caller-defined date is a trading day."""

    def is_trading_day(self, day: date) -> bool: ...


class ExplicitTradingDayCalendar:
    """Immutable explicit set of trading dates for orchestration."""

    def __init__(self, trading_dates: Iterable[date]) -> None:
        selected = frozenset(trading_dates)
        if any(not isinstance(day, date) for day in selected):
            raise TypeError("all trading dates must be date instances")
        self._trading_dates = selected

    @property
    def trading_dates(self) -> frozenset[date]:
        return self._trading_dates

    def is_trading_day(self, day: date) -> bool:
        if not isinstance(day, date):
            raise TypeError("day must be a date")
        return day in self._trading_dates
