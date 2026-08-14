"""Deterministic registered exit-policy contracts and reusable resolvers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from decimal import Decimal
from types import MappingProxyType
from typing import Protocol, runtime_checkable
from zoneinfo import ZoneInfo

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from algo_trader.backtest.models import BacktestIntegrityError, DynamicExitPolicySpec
from algo_trader.data import bar_available_at
from algo_trader.domain import ExitReason, Fill, ProtectiveExitSpec, Side
from algo_trader.execution import ExitResult, HistoricalExecutionSimulator

MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")


@runtime_checkable
class BacktestExitPolicyResolver(Protocol):
    """Structural contract for one deterministic historical exit policy."""

    policy_id: str

    def resolve(
        self,
        spec: DynamicExitPolicySpec,
        *,
        side: Side,
        symbol: str,
        quantity: int,
        entry_fill: Fill,
        candles: pl.DataFrame,
        execution_simulator: HistoricalExecutionSimulator,
        strategy_exit_at: datetime | None,
        forced_cutoff: datetime,
    ) -> ExitResult:
        """Resolve the completed position's deterministic exit outcome."""
        ...


class _RMultipleTrailingParameters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    initial_stop_price: Decimal = Field(gt=0, allow_inf_nan=False)
    hard_target_r: Decimal = Field(gt=0, allow_inf_nan=False)
    breakeven_trigger_r: Decimal = Field(gt=0, allow_inf_nan=False)
    breakeven_stop_r: Decimal = Field(ge=0, allow_inf_nan=False)
    profit_lock_trigger_r: Decimal = Field(gt=0, allow_inf_nan=False)
    profit_lock_stop_r: Decimal = Field(ge=0, allow_inf_nan=False)
    trailing_distance_r: Decimal = Field(gt=0, allow_inf_nan=False)
    maximum_hold_minutes: int = Field(strict=True, gt=0)
    latest_exit_time: time

    @field_validator(
        "initial_stop_price",
        "hard_target_r",
        "breakeven_trigger_r",
        "breakeven_stop_r",
        "profit_lock_trigger_r",
        "profit_lock_stop_r",
        "trailing_distance_r",
        mode="before",
    )
    @classmethod
    def reject_boolean_decimals(cls, value: object) -> object:
        if isinstance(value, bool):
            raise TypeError("R-multiple numeric parameters cannot be booleans")
        return value

    @field_validator("latest_exit_time", mode="before")
    @classmethod
    def require_local_clock_time(cls, value: object) -> object:
        if not isinstance(value, time):
            raise TypeError("latest_exit_time must be a time")
        if value.tzinfo is not None:
            raise ValueError(
                "latest_exit_time must be a timezone-naive Asia/Kolkata local clock time"
            )
        return value

    @model_validator(mode="after")
    def validate_geometry(self) -> _RMultipleTrailingParameters:
        if self.profit_lock_trigger_r <= self.breakeven_trigger_r:
            raise ValueError(
                "profit_lock_trigger_r must exceed breakeven_trigger_r"
            )
        if self.hard_target_r <= self.profit_lock_trigger_r:
            raise ValueError("hard_target_r must exceed profit_lock_trigger_r")
        if self.profit_lock_stop_r < self.breakeven_stop_r:
            raise ValueError(
                "profit_lock_stop_r must be at least breakeven_stop_r"
            )
        if self.profit_lock_stop_r >= self.profit_lock_trigger_r:
            raise ValueError(
                "profit_lock_stop_r must be below profit_lock_trigger_r"
            )
        return self


@dataclass(frozen=True, slots=True)
class RMultipleTrailingExitPolicyResolver:
    """Causal next-bar R-multiple trailing policy with a fixed hard target."""

    policy_id = "R_MULTIPLE_TRAILING_V1"

    def resolve(
        self,
        spec: DynamicExitPolicySpec,
        *,
        side: Side,
        symbol: str,
        quantity: int,
        entry_fill: Fill,
        candles: pl.DataFrame,
        execution_simulator: HistoricalExecutionSimulator,
        strategy_exit_at: datetime | None,
        forced_cutoff: datetime,
    ) -> ExitResult:
        if not isinstance(spec, DynamicExitPolicySpec):
            raise TypeError("spec must be a DynamicExitPolicySpec")
        if spec.policy_id != self.policy_id:
            raise ValueError(
                f"resolver {self.policy_id!r} cannot resolve policy {spec.policy_id!r}"
            )
        if not isinstance(entry_fill, Fill):
            raise TypeError("entry_fill must be a Fill")
        if not isinstance(side, Side):
            raise TypeError("side must be a Side")
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        if quantity != entry_fill.quantity:
            raise ValueError("exit policy quantity must equal entry_fill.quantity")
        if not isinstance(candles, pl.DataFrame):
            raise TypeError("candles must be a Polars DataFrame")
        if not isinstance(execution_simulator, HistoricalExecutionSimulator):
            raise TypeError(
                "execution_simulator must be a HistoricalExecutionSimulator"
            )
        _require_aware(forced_cutoff, "forced_cutoff")
        if strategy_exit_at is not None:
            _require_aware(strategy_exit_at, "strategy_exit_at")
        parameters = _RMultipleTrailingParameters.model_validate(dict(spec.parameters))
        entry_price = entry_fill.price
        initial_stop = parameters.initial_stop_price
        risk = (
            entry_price - initial_stop
            if side is Side.LONG
            else initial_stop - entry_price
        )
        if not risk.is_finite() or risk <= 0:
            raise BacktestIntegrityError(
                "R_MULTIPLE_TRAILING_V1 requires strictly positive initial risk "
                f"from actual entry fill for {symbol!r}"
            )
        target = (
            entry_price + parameters.hard_target_r * risk
            if side is Side.LONG
            else entry_price - parameters.hard_target_r * risk
        )
        if not target.is_finite() or target <= 0:
            raise BacktestIntegrityError(
                "R_MULTIPLE_TRAILING_V1 produced a nonpositive hard target"
            )

        latest_exit = datetime.combine(
            entry_fill.timestamp.astimezone(MARKET_TIMEZONE).date(),
            parameters.latest_exit_time,
            MARKET_TIMEZONE,
        )
        maximum_hold = entry_fill.timestamp + timedelta(
            minutes=parameters.maximum_hold_minutes
        )
        deadline = min(
            value
            for value in (maximum_hold, latest_exit, forced_cutoff, strategy_exit_at)
            if value is not None
        )
        if deadline < entry_fill.timestamp:
            raise BacktestIntegrityError(
                "R_MULTIPLE_TRAILING_V1 effective deadline precedes actual entry fill"
            )
        deadline_reason = (
            ExitReason.STRATEGY_EXIT
            if strategy_exit_at is not None and strategy_exit_at == deadline
            else ExitReason.TIME_EXIT
        )
        return self._resolve_bars(
            side=side,
            symbol=symbol,
            quantity=quantity,
            entry_fill=entry_fill,
            candles=candles,
            execution_simulator=execution_simulator,
            parameters=parameters,
            initial_stop=initial_stop,
            target=target,
            risk=risk,
            deadline=deadline,
            deadline_reason=deadline_reason,
        )

    @staticmethod
    def _resolve_bars(
        *,
        side: Side,
        symbol: str,
        quantity: int,
        entry_fill: Fill,
        candles: pl.DataFrame,
        execution_simulator: HistoricalExecutionSimulator,
        parameters: _RMultipleTrailingParameters,
        initial_stop: Decimal,
        target: Decimal,
        risk: Decimal,
        deadline: datetime,
        deadline_reason: ExitReason,
    ) -> ExitResult:
        eligible = candles.filter(pl.col("timestamp") >= entry_fill.timestamp)
        current_stop = initial_stop
        best_favorable = entry_fill.price

        for index, row in enumerate(eligible.iter_rows(named=True)):
            bar_start = row["timestamp"]
            if not isinstance(bar_start, datetime):
                raise TypeError("candle timestamp must materialize as a datetime")
            if bar_start > deadline:
                break
            bar_frame = eligible.slice(index, 1)
            if bar_start == deadline:
                result = execution_simulator.fill_market_exit(
                    side=side,
                    symbol=symbol,
                    quantity=quantity,
                    requested_at=deadline,
                    exit_reason=deadline_reason,
                    candles=bar_frame,
                )
                if result is not None and result.fill.timestamp == deadline:
                    return result
                break

            bar_end = bar_available_at(
                bar_start, execution_simulator.timeframe_minutes
            )
            if bar_end > deadline:
                break
            protective = execution_simulator.fill_active_protective_exit(
                side=side,
                symbol=symbol,
                quantity=quantity,
                active_from=bar_start,
                protective_exit=ProtectiveExitSpec(
                    stop_price=current_stop,
                    target_price=target,
                ),
                candles=bar_frame,
            )
            if protective is not None:
                return protective

            favorable = Decimal(str(row["high"] if side is Side.LONG else row["low"]))
            if not favorable.is_finite() or favorable <= 0:
                raise BacktestIntegrityError(
                    "R_MULTIPLE_TRAILING_V1 requires finite positive candle prices"
                )
            best_favorable = (
                max(best_favorable, favorable)
                if side is Side.LONG
                else min(best_favorable, favorable)
            )
            mfe_r = (
                (best_favorable - entry_fill.price) / risk
                if side is Side.LONG
                else (entry_fill.price - best_favorable) / risk
            )
            current_stop = _next_stop(
                side=side,
                entry_price=entry_fill.price,
                current_stop=current_stop,
                best_favorable=best_favorable,
                mfe_r=mfe_r,
                risk=risk,
                parameters=parameters,
            )

        raise BacktestIntegrityError(
            "R_MULTIPLE_TRAILING_V1 could not exit exactly at its mandatory "
            f"deadline {deadline.isoformat()} for {symbol!r}"
        )


def _next_stop(
    *,
    side: Side,
    entry_price: Decimal,
    current_stop: Decimal,
    best_favorable: Decimal,
    mfe_r: Decimal,
    risk: Decimal,
    parameters: _RMultipleTrailingParameters,
) -> Decimal:
    if mfe_r < parameters.breakeven_trigger_r:
        return current_stop
    if side is Side.LONG:
        breakeven = entry_price + parameters.breakeven_stop_r * risk
        profit_lock = entry_price + parameters.profit_lock_stop_r * risk
        if mfe_r < parameters.profit_lock_trigger_r:
            return max(current_stop, breakeven)
        if mfe_r == parameters.profit_lock_trigger_r:
            return max(current_stop, profit_lock)
        trailing = best_favorable - parameters.trailing_distance_r * risk
        return max(current_stop, profit_lock, trailing)

    breakeven = entry_price - parameters.breakeven_stop_r * risk
    profit_lock = entry_price - parameters.profit_lock_stop_r * risk
    if mfe_r < parameters.profit_lock_trigger_r:
        return min(current_stop, breakeven)
    if mfe_r == parameters.profit_lock_trigger_r:
        return min(current_stop, profit_lock)
    trailing = best_favorable + parameters.trailing_distance_r * risk
    return min(current_stop, profit_lock, trailing)


def _require_aware(value: object, name: str) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def validate_exit_policy_resolver(resolver: object) -> BacktestExitPolicyResolver:
    """Perform explicit lightweight registry validation without deep ``isinstance`` use."""
    policy_id = getattr(resolver, "policy_id", None)
    resolve = getattr(resolver, "resolve", None)
    if not isinstance(policy_id, str) or not policy_id.strip():
        raise ValueError("exit policy resolver policy_id must be a non-empty string")
    if not callable(resolve):
        raise TypeError("exit policy resolver must provide a callable resolve method")
    return resolver  # type: ignore[return-value]


def freeze_exit_policy_registry(
    resolvers: tuple[object, ...],
) -> Mapping[str, BacktestExitPolicyResolver]:
    """Build an immutable per-backtester registry with deterministic duplicate checks."""
    registered: dict[str, BacktestExitPolicyResolver] = {}
    for item in resolvers:
        resolver = validate_exit_policy_resolver(item)
        if resolver.policy_id in registered:
            raise ValueError(
                f"duplicate exit policy resolver policy_id: {resolver.policy_id!r}"
            )
        registered[resolver.policy_id] = resolver
    return MappingProxyType(registered)
