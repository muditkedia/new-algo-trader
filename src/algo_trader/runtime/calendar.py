"""Explicit caller-owned trading-day boundary; no inferred holiday policy."""

from collections.abc import Iterable
from datetime import date
from pathlib import Path
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


def load_trading_day_calendar(path: Path) -> ExplicitTradingDayCalendar:
    """Load one explicit ISO-date-per-line calendar; missing input fails closed."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"explicit trading calendar is required: {source}")
    dates: list[date] = []
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        try:
            dates.append(date.fromisoformat(value))
        except ValueError as error:
            raise ValueError(
                f"invalid trading calendar date on line {line_number}: {value!r}"
            ) from error
    if not dates:
        raise ValueError("explicit trading calendar must contain at least one date")
    if len(dates) != len(set(dates)):
        raise ValueError("explicit trading calendar must not contain duplicate dates")
    return ExplicitTradingDayCalendar(dates)
