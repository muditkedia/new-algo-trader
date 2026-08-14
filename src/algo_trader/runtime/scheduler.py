"""Explicit per-trading-date APScheduler infrastructure."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger

from algo_trader.runtime.calendar import TradingDayProvider
from algo_trader.runtime.models import RuntimeSessionTimes

MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")


class RuntimeScheduler:
    """Route explicit date jobs to RuntimeService without duplicating business rules."""

    def __init__(
        self,
        service: object,
        trading_calendar: TradingDayProvider,
        *,
        misfire_grace_seconds: int = 60,
        scheduler_factory: Callable[..., object] = BackgroundScheduler,
    ) -> None:
        if not isinstance(trading_calendar, TradingDayProvider):
            raise TypeError("trading_calendar must implement TradingDayProvider")
        if isinstance(misfire_grace_seconds, bool) or misfire_grace_seconds <= 0:
            raise ValueError("misfire_grace_seconds must be a positive integer")
        self.service = service
        self.trading_calendar = trading_calendar
        self._scheduler = scheduler_factory(timezone=MARKET_TIMEZONE)
        self._misfire_grace_seconds = misfire_grace_seconds

    @property
    def scheduler(self) -> object:
        return self._scheduler

    def configure_date(
        self,
        trading_date: date,
        session_times: RuntimeSessionTimes,
        runtime_session_id: str,
    ) -> tuple[str, ...]:
        """Create exactly four deterministic non-overlapping jobs for a trading date."""
        if not self.trading_calendar.is_trading_day(trading_date):
            return ()
        callbacks = (
            ("market-open", session_times.market_open_time, self.service.market_open),
            ("entry-cutoff", session_times.entry_cutoff_time, self.service.close_entries),
            ("square-off", session_times.square_off_time, self.service.force_square_off),
            ("shutdown", session_times.shutdown_time, self.service.shutdown),
        )
        job_ids = []
        for suffix, wall_time, callback in callbacks:
            job_id = f"runtime:{runtime_session_id}:{trading_date.isoformat()}:{suffix}"
            run_date = _at(trading_date, wall_time)
            self._scheduler.add_job(
                callback,
                trigger=DateTrigger(run_date=run_date, timezone=MARKET_TIMEZONE),
                id=job_id,
                replace_existing=True,
                max_instances=1,
                coalesce=True,
                misfire_grace_time=self._misfire_grace_seconds,
            )
            job_ids.append(job_id)
        return tuple(job_ids)

    def start(self) -> None:
        self._scheduler.start()

    def shutdown(self, *, wait: bool = True) -> None:
        self._scheduler.shutdown(wait=wait)


def _at(day: date, wall_time: time) -> datetime:
    return datetime.combine(day, wall_time, tzinfo=MARKET_TIMEZONE)
