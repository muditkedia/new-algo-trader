"""Production composition root for one explicit PAPER or LIVE runtime session."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

import polars as pl

from algo_trader.broker import (
    AngelOneCandleClient,
    AngelOneInstrumentMaster,
    BrokerInstrument,
)
from algo_trader.runtime.clock import Clock
from algo_trader.runtime.connectivity import run_smartapi_connectivity_check
from algo_trader.runtime.market_data import get_completed_five_minute_candles
from algo_trader.runtime.models import RuntimeConfig, RuntimeMode, RuntimePhase
from algo_trader.runtime.protocols import RuntimePlanProvider
from algo_trader.runtime.scheduler import RuntimeScheduler
from algo_trader.runtime.service import RuntimeService


class FiveMinuteStrategyCycle:
    """Fetch completed candles, enforce health, then process scored plans."""

    def __init__(
        self,
        *,
        service: RuntimeService,
        clock: Clock,
        candle_client: AngelOneCandleClient,
        instruments: Sequence[BrokerInstrument],
        history_start: Mapping[str, datetime],
        plan_provider: RuntimePlanProvider,
    ) -> None:
        self._service = service
        self._clock = clock
        self._candle_client = candle_client
        self._instruments = tuple(instruments)
        self._history_start = dict(history_start)
        self._plan_provider = plan_provider
        symbols = {instrument.symbol for instrument in self._instruments}
        if symbols != set(self._history_start):
            raise ValueError("history_start must exactly cover subscribed instruments")

    def __call__(self) -> tuple[object, ...]:
        decision_at = self._clock.now()
        completed: dict[str, pl.DataFrame] = {}
        for instrument in self._instruments:
            completed[instrument.symbol] = get_completed_five_minute_candles(
                self._candle_client,
                instrument,
                self._history_start[instrument.symbol],
                decision_at,
                decision_at,
            )
        required_symbols = tuple(sorted(completed))
        self._service.advance_dynamic_exits(completed, decision_at)
        if not self._service.check_market_data_health(
            decision_at,
            required_symbols=required_symbols,
        ):
            return ()
        plans = self._plan_provider.plans_for_cycle(completed, decision_at)
        if not plans:
            return ()
        result = self._service.process_plans(plans, decision_at=decision_at)
        return tuple(result.decisions)


@dataclass(frozen=True, slots=True)
class RuntimeApplication:
    """Wired runtime with one explicit start operation and no implicit LIVE mode."""

    config: RuntimeConfig
    service: RuntimeService
    scheduler: RuntimeScheduler
    strategy_cycle: FiveMinuteStrategyCycle
    instruments: tuple[BrokerInstrument, ...]
    live_connectivity_preflight: Callable[[], object] | None = None

    def start(self) -> RuntimePhase:
        if self.config.mode is RuntimeMode.LIVE:
            if self.live_connectivity_preflight is None:
                raise RuntimeError("LIVE startup requires an explicit connectivity preflight")
            self.live_connectivity_preflight()
        phase = self.service.start()
        if phase is RuntimePhase.HALTED:
            return phase
        self.service.connect_stream(self.instruments)
        self.scheduler.configure_date(
            self.service.trading_date,
            self.config.session_times,
            self.service.runtime_session_id,
            strategy_cycle=self.strategy_cycle,
        )
        self.scheduler.start()
        return phase


def compose_runtime_application(
    *,
    config: RuntimeConfig,
    service: RuntimeService,
    clock: Clock,
    candle_client: AngelOneCandleClient,
    instruments: Sequence[BrokerInstrument],
    history_start: Mapping[str, datetime],
    plan_provider: RuntimePlanProvider,
    scheduler: RuntimeScheduler,
    instrument_master: AngelOneInstrumentMaster | None = None,
    live_connectivity_preflight: Callable[[], object] | None = None,
) -> RuntimeApplication:
    """Bind existing runtime components without relocating business logic."""
    if not isinstance(config, RuntimeConfig):
        raise TypeError("config must be a RuntimeConfig")
    if config.mode is RuntimeMode.LIVE and live_connectivity_preflight is None:
        if instrument_master is None:
            raise ValueError(
                "LIVE composition requires an instrument master for connectivity preflight"
            )
        quote_symbol = instruments[0].symbol if instruments else None

        def live_connectivity_preflight() -> object:
            return run_smartapi_connectivity_check(
                config=config,
                instrument_master=instrument_master,
                checked_at=clock.now(),
                quote_symbol=quote_symbol,
            )
    cycle = FiveMinuteStrategyCycle(
        service=service,
        clock=clock,
        candle_client=candle_client,
        instruments=instruments,
        history_start=history_start,
        plan_provider=plan_provider,
    )
    return RuntimeApplication(
        config=config,
        service=service,
        scheduler=scheduler,
        strategy_cycle=cycle,
        instruments=tuple(instruments),
        live_connectivity_preflight=live_connectivity_preflight,
    )
