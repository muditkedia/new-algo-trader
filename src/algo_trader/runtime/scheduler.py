"""Explicit per-trading-date APScheduler infrastructure."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, time
from threading import Condition
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

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
        self._shutdown_condition = Condition()
        self._shutdown_started = False
        self._shutdown_complete = False
        self._shutdown_result: object = None
        self._shutdown_error: BaseException | None = None

    @property
    def scheduler(self) -> object:
        return self._scheduler

    def configure_date(
        self,
        trading_date: date,
        session_times: RuntimeSessionTimes,
        runtime_session_id: str,
        strategy_cycle: Callable[[], object] | None = None,
    ) -> tuple[str, ...]:
        """Create deterministic lifecycle jobs and an optional five-minute cycle."""
        if not self.trading_calendar.is_trading_day(trading_date):
            return ()
        callbacks = (
            ("market-open", session_times.market_open_time, self.service.market_open),
            ("entry-cutoff", session_times.entry_cutoff_time, self.service.close_entries),
            ("square-off", session_times.square_off_time, self.service.force_square_off),
            (
                "market-close-check",
                session_times.market_close_time,
                self.service.market_close_check,
            ),
            ("shutdown", session_times.shutdown_time, self._scheduled_shutdown),
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
        if strategy_cycle is not None:
            job_id = f"runtime:{runtime_session_id}:{trading_date.isoformat()}:strategy-cycle"
            self._scheduler.add_job(
                strategy_cycle,
                trigger=IntervalTrigger(
                    minutes=5,
                    start_date=_at(trading_date, time(9, 20)),
                    end_date=_at(trading_date, session_times.entry_cutoff_time),
                    timezone=MARKET_TIMEZONE,
                ),
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

    def shutdown(self, *, wait: bool = True) -> object:
        """Shut down the Runtime service and APScheduler exactly once, in order."""
        return self._coordinate_shutdown(wait=wait, scheduled=False)

    def _scheduled_shutdown(self) -> object:
        return self._coordinate_shutdown(wait=False, scheduled=True)

    def _coordinate_shutdown(self, *, wait: bool, scheduled: bool) -> object:
        with self._shutdown_condition:
            if self._shutdown_complete:
                if scheduled:
                    return self._shutdown_result
                if self._shutdown_error is not None:
                    raise self._shutdown_error
                return self._shutdown_result
            if self._shutdown_started:
                if scheduled:
                    return None
                while not self._shutdown_complete:
                    self._shutdown_condition.wait()
                if self._shutdown_error is not None:
                    raise self._shutdown_error
                return self._shutdown_result
            self._shutdown_started = True

        result: object = None
        service_error: BaseException | None = None
        scheduler_error: BaseException | None = None
        try:
            result = self.service.shutdown()
        except BaseException as error:
            service_error = error
        try:
            self._scheduler.shutdown(wait=wait)
        except BaseException as error:
            scheduler_error = error

        with self._shutdown_condition:
            self._shutdown_result = result
            self._shutdown_error = service_error or scheduler_error
            self._shutdown_complete = True
            self._shutdown_condition.notify_all()
            if self._shutdown_error is not None:
                raise self._shutdown_error
            return self._shutdown_result


def _at(day: date, wall_time: time) -> datetime:
    return datetime.combine(day, wall_time, tzinfo=MARKET_TIMEZONE)
