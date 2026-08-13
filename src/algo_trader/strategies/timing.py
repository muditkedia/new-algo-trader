"""Helpers for candle-information availability timing."""

from datetime import datetime, timedelta


def bar_available_at(bar_start: datetime, timeframe_minutes: int = 5) -> datetime:
    """Return when a start-stamped bar's completed values become available."""
    if bar_start.tzinfo is None or bar_start.utcoffset() is None:
        raise ValueError("bar_start must be timezone-aware")
    if isinstance(timeframe_minutes, bool) or not isinstance(timeframe_minutes, int):
        raise TypeError("timeframe_minutes must be an integer")
    if timeframe_minutes <= 0:
        raise ValueError("timeframe_minutes must be positive")
    return bar_start + timedelta(minutes=timeframe_minutes)

