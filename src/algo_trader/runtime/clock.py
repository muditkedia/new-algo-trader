"""Explicit Runtime clock boundary."""

from datetime import datetime
from typing import Protocol, runtime_checkable
from zoneinfo import ZoneInfo

MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")


@runtime_checkable
class Clock(Protocol):
    """Caller-injectable source of aware current time."""

    def now(self) -> datetime: ...


class SystemClock:
    """The only Runtime core implementation that reads the system clock."""

    def now(self) -> datetime:
        return datetime.now(MARKET_TIMEZONE)
