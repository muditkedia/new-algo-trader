"""Event-driven historical backtest orchestration."""

from algo_trader.backtest.engine import (
    BACKTESTER_VERSION,
    HistoricalBacktester,
)
from algo_trader.backtest.exit_policies import (
    BacktestExitPolicyResolver,
    RMultipleTrailingExitPolicyResolver,
)
from algo_trader.backtest.models import (
    BacktestConfig,
    BacktestIntegrityError,
    BacktestRequestOutcome,
    BacktestRequestResult,
    BacktestRunResult,
    BacktestTradeRecord,
    BacktestTradeRequest,
    DynamicExitPolicySpec,
)

__all__ = [
    "BACKTESTER_VERSION",
    "BacktestConfig",
    "BacktestExitPolicyResolver",
    "BacktestIntegrityError",
    "BacktestRequestOutcome",
    "BacktestRequestResult",
    "BacktestRunResult",
    "BacktestTradeRecord",
    "BacktestTradeRequest",
    "DynamicExitPolicySpec",
    "HistoricalBacktester",
    "RMultipleTrailingExitPolicyResolver",
]
