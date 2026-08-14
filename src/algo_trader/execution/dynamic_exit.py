"""Pure R-multiple trailing state shared by historical and runtime execution."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from algo_trader.domain import ExitReasonDetail, Side


@dataclass(frozen=True, slots=True)
class RMultipleTrailingCoreParameters:
    breakeven_trigger_r: Decimal
    breakeven_stop_r: Decimal
    profit_lock_trigger_r: Decimal
    profit_lock_stop_r: Decimal
    trailing_distance_r: Decimal


@dataclass(frozen=True, slots=True)
class RMultipleTrailingState:
    entry_price: Decimal
    initial_stop: Decimal
    risk: Decimal
    current_stop: Decimal
    best_favorable: Decimal


def initialize_r_multiple_state(
    side: Side,
    entry_price: Decimal,
    initial_stop: Decimal,
) -> RMultipleTrailingState:
    risk = entry_price - initial_stop if side is Side.LONG else initial_stop - entry_price
    if not risk.is_finite() or risk <= 0:
        raise ValueError("R-multiple trailing requires strictly positive actual-fill risk")
    return RMultipleTrailingState(
        entry_price=entry_price,
        initial_stop=initial_stop,
        risk=risk,
        current_stop=initial_stop,
        best_favorable=entry_price,
    )


def advance_r_multiple_state(
    state: RMultipleTrailingState,
    side: Side,
    favorable_price: Decimal,
    parameters: RMultipleTrailingCoreParameters,
) -> RMultipleTrailingState:
    """Earn a monotonic stop update after one completed favorable bar."""
    best = (
        max(state.best_favorable, favorable_price)
        if side is Side.LONG
        else min(state.best_favorable, favorable_price)
    )
    mfe_r = (
        (best - state.entry_price) / state.risk
        if side is Side.LONG
        else (state.entry_price - best) / state.risk
    )
    stop = state.current_stop
    if mfe_r >= parameters.breakeven_trigger_r:
        direction = Decimal("1") if side is Side.LONG else Decimal("-1")
        breakeven = (
            state.entry_price + direction * parameters.breakeven_stop_r * state.risk
        )
        choose = max if side is Side.LONG else min
        stop = choose(stop, breakeven)
        if mfe_r >= parameters.profit_lock_trigger_r:
            profit_lock = (
                state.entry_price
                + direction * parameters.profit_lock_stop_r * state.risk
            )
            stop = choose(stop, profit_lock)
            if mfe_r > parameters.profit_lock_trigger_r:
                trailing = (
                    best - parameters.trailing_distance_r * state.risk
                    if side is Side.LONG
                    else best + parameters.trailing_distance_r * state.risk
                )
                stop = choose(stop, trailing)
    return RMultipleTrailingState(
        entry_price=state.entry_price,
        initial_stop=state.initial_stop,
        risk=state.risk,
        current_stop=stop,
        best_favorable=best,
    )


def r_multiple_stop_detail(
    state: RMultipleTrailingState,
    side: Side,
    parameters: RMultipleTrailingCoreParameters,
) -> ExitReasonDetail:
    if state.current_stop == state.initial_stop:
        return ExitReasonDetail.INITIAL_STOP
    direction = Decimal("1") if side is Side.LONG else Decimal("-1")
    breakeven = state.entry_price + direction * parameters.breakeven_stop_r * state.risk
    if state.current_stop == breakeven:
        return ExitReasonDetail.BREAKEVEN_STOP
    profit_lock = state.entry_price + direction * parameters.profit_lock_stop_r * state.risk
    if state.current_stop == profit_lock:
        return ExitReasonDetail.PROFIT_LOCK
    return ExitReasonDetail.TRAILING_STOP
