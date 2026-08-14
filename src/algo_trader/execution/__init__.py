"""Broker-neutral historical execution simulation."""

from algo_trader.execution.dynamic_exit import (
    RMultipleTrailingCoreParameters,
    RMultipleTrailingState,
    advance_r_multiple_state,
    initialize_r_multiple_state,
    r_multiple_stop_detail,
)
from algo_trader.execution.historical import ExitResult, HistoricalExecutionSimulator
from algo_trader.execution.slippage import (
    ExecutionAction,
    FixedBasisPointsSlippage,
    NoSlippage,
    SlippageModel,
)

__all__ = [
    "ExecutionAction",
    "ExitResult",
    "FixedBasisPointsSlippage",
    "HistoricalExecutionSimulator",
    "NoSlippage",
    "RMultipleTrailingCoreParameters",
    "RMultipleTrailingState",
    "SlippageModel",
    "advance_r_multiple_state",
    "initialize_r_multiple_state",
    "r_multiple_stop_detail",
]
