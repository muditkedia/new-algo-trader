"""Behavioral prefix-invariance validation for strategy signal generation."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import polars as pl

from algo_trader.data import bar_available_at
from algo_trader.domain import Signal, SignalStatus
from algo_trader.strategies.contract import Strategy
from algo_trader.strategies.validation import validate_strategy_input

STRATEGY_CAUSALITY_GATE_VERSION = "1"
_TIMEFRAME_MINUTES = 5


class StrategyCausalityViolation(RuntimeError):
    """Raised when strategy behavior violates the causality contract."""

    def __init__(
        self,
        category: str,
        *,
        strategy_id: str,
        strategy_version: str,
        prefix_length: int,
        detail: str,
        decision_timestamp: datetime | None = None,
    ) -> None:
        self.category = category
        context = (
            f"strategy_id={strategy_id!r} strategy_version={strategy_version!r} "
            f"prefix_length={prefix_length}"
        )
        if decision_timestamp is not None:
            context += f" decision_timestamp={decision_timestamp.isoformat()}"
        super().__init__(f"[{category}] {context}: {detail}")


@dataclass(frozen=True)
class StrategyCausalityReport:
    """Deterministic evidence returned after a successful causality-gate run."""

    gate_version: str
    strategy_id: str
    strategy_version: str
    symbol: str
    timeframe_minutes: int
    warmup_bars: int
    row_count: int
    first_tested_prefix_length: int
    tested_prefix_count: int
    full_signal_count: int
    first_candle_timestamp: datetime
    last_candle_timestamp: datetime
    last_information_available_at: datetime


def _violation(
    category: str,
    *,
    strategy_id: str,
    strategy_version: str,
    prefix_length: int,
    detail: str,
    decision_timestamp: datetime | None = None,
) -> StrategyCausalityViolation:
    return StrategyCausalityViolation(
        category,
        strategy_id=strategy_id,
        strategy_version=strategy_version,
        prefix_length=prefix_length,
        detail=detail,
        decision_timestamp=decision_timestamp,
    )


def _snapshot_parameters(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        return {
            deepcopy(key): _snapshot_parameter_value(value)
            for key, value in parameters.items()
        }
    except Exception as error:
        raise TypeError("strategy.parameters must support a detached deep snapshot") from error


def _snapshot_parameter_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            deepcopy(key): _snapshot_parameter_value(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_snapshot_parameter_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_parameter_value(item) for item in value)
    if isinstance(value, set):
        return {_snapshot_parameter_value(item) for item in value}
    if isinstance(value, frozenset):
        return frozenset(_snapshot_parameter_value(item) for item in value)
    return deepcopy(value)


def _parameters_equal(current: Mapping[str, Any], original: Mapping[str, Any]) -> bool:
    try:
        result = current == original
    except Exception:
        return False
    return isinstance(result, bool) and result


def _validate_strategy(strategy: object) -> tuple[str, str, int, Mapping[str, Any]]:
    if not isinstance(strategy, Strategy):
        raise TypeError("strategy must structurally satisfy the Strategy protocol")

    strategy_id = strategy.strategy_id
    if not isinstance(strategy_id, str) or not strategy_id.strip():
        raise ValueError("strategy.strategy_id must be a nonempty string")

    strategy_version = strategy.strategy_version
    if not isinstance(strategy_version, str) or not strategy_version.strip():
        raise ValueError("strategy.strategy_version must be a nonempty string")

    warmup_bars = strategy.warmup_bars
    if isinstance(warmup_bars, bool) or not isinstance(warmup_bars, int):
        raise TypeError("strategy.warmup_bars must be an integer and not Boolean")
    if warmup_bars < 0:
        raise ValueError("strategy.warmup_bars must be non-negative")

    parameters = strategy.parameters
    if not isinstance(parameters, Mapping):
        raise TypeError("strategy.parameters must be a Mapping")

    return strategy_id, strategy_version, warmup_bars, parameters


def _assert_configuration_unchanged(
    strategy: Strategy,
    original_parameters: Mapping[str, Any],
    *,
    strategy_id: str,
    strategy_version: str,
    prefix_length: int,
) -> None:
    current_parameters = strategy.parameters
    if not isinstance(current_parameters, Mapping) or not _parameters_equal(
        current_parameters, original_parameters
    ):
        raise _violation(
            "STRATEGY_CONFIGURATION_MUTATION",
            strategy_id=strategy_id,
            strategy_version=strategy_version,
            prefix_length=prefix_length,
            detail="strategy.parameters changed during evaluation",
        )


def _validate_signals(
    output: object,
    *,
    strategy_id: str,
    strategy_version: str,
    symbol: str,
    prefix_length: int,
    earliest_information_time: datetime,
    prefix_information_cutoff: datetime,
) -> list[Signal]:
    if not isinstance(output, list):
        raise _violation(
            "OUTPUT_CONTRACT",
            strategy_id=strategy_id,
            strategy_version=strategy_version,
            prefix_length=prefix_length,
            detail="generate_signals must return an actual list[Signal]",
        )

    previous_timestamp: datetime | None = None
    unique_signals: list[Signal] = []
    for index, signal in enumerate(output):
        if not isinstance(signal, Signal):
            raise _violation(
                "OUTPUT_CONTRACT",
                strategy_id=strategy_id,
                strategy_version=strategy_version,
                prefix_length=prefix_length,
                detail=f"output item {index} is not a Signal",
            )
        if signal.strategy_id != strategy_id:
            raise _violation(
                "SIGNAL_METADATA",
                strategy_id=strategy_id,
                strategy_version=strategy_version,
                prefix_length=prefix_length,
                decision_timestamp=signal.timestamp,
                detail="signal.strategy_id does not match the strategy",
            )
        if signal.strategy_version != strategy_version:
            raise _violation(
                "SIGNAL_METADATA",
                strategy_id=strategy_id,
                strategy_version=strategy_version,
                prefix_length=prefix_length,
                decision_timestamp=signal.timestamp,
                detail="signal.strategy_version does not match the strategy",
            )
        if signal.symbol != symbol:
            raise _violation(
                "SIGNAL_METADATA",
                strategy_id=strategy_id,
                strategy_version=strategy_version,
                prefix_length=prefix_length,
                decision_timestamp=signal.timestamp,
                detail="signal.symbol does not match the candle frame symbol",
            )
        if signal.status is not SignalStatus.GENERATED:
            raise _violation(
                "SIGNAL_METADATA",
                strategy_id=strategy_id,
                strategy_version=strategy_version,
                prefix_length=prefix_length,
                decision_timestamp=signal.timestamp,
                detail="strategy signals must have GENERATED status",
            )
        if previous_timestamp is not None and signal.timestamp < previous_timestamp:
            raise _violation(
                "SIGNAL_ORDER",
                strategy_id=strategy_id,
                strategy_version=strategy_version,
                prefix_length=prefix_length,
                decision_timestamp=signal.timestamp,
                detail="signal timestamps must be nondecreasing without gate-side sorting",
            )
        if signal in unique_signals:
            raise _violation(
                "DUPLICATE_SIGNAL",
                strategy_id=strategy_id,
                strategy_version=strategy_version,
                prefix_length=prefix_length,
                decision_timestamp=signal.timestamp,
                detail="output contains an exact duplicate Signal",
            )
        if signal.timestamp < earliest_information_time:
            raise _violation(
                "EARLY_SIGNAL_TIMESTAMP",
                strategy_id=strategy_id,
                strategy_version=strategy_version,
                prefix_length=prefix_length,
                decision_timestamp=signal.timestamp,
                detail=(
                    "signal predates the first supplied candle's information availability "
                    f"at {earliest_information_time.isoformat()}"
                ),
            )
        if signal.timestamp > prefix_information_cutoff:
            raise _violation(
                "FUTURE_SIGNAL_TIMESTAMP",
                strategy_id=strategy_id,
                strategy_version=strategy_version,
                prefix_length=prefix_length,
                decision_timestamp=signal.timestamp,
                detail=(
                    "signal is later than the prefix information cutoff "
                    f"{prefix_information_cutoff.isoformat()}"
                ),
            )
        previous_timestamp = signal.timestamp
        unique_signals.append(signal)

    return output


def _evaluate_prefix(
    strategy: Strategy,
    prefix: pl.DataFrame,
    *,
    original_parameters: Mapping[str, Any],
    strategy_id: str,
    strategy_version: str,
    symbol: str,
    prefix_length: int,
    earliest_information_time: datetime,
    prefix_information_cutoff: datetime,
) -> list[Signal]:
    before = prefix.clone()
    try:
        output = strategy.generate_signals(prefix)
    except BaseException:
        if not prefix.equals(before):
            raise _violation(
                "INPUT_MUTATION",
                strategy_id=strategy_id,
                strategy_version=strategy_version,
                prefix_length=prefix_length,
                detail="strategy mutated its supplied candle prefix",
            ) from None
        _assert_configuration_unchanged(
            strategy,
            original_parameters,
            strategy_id=strategy_id,
            strategy_version=strategy_version,
            prefix_length=prefix_length,
        )
        raise

    if not prefix.equals(before):
        raise _violation(
            "INPUT_MUTATION",
            strategy_id=strategy_id,
            strategy_version=strategy_version,
            prefix_length=prefix_length,
            detail="strategy mutated its supplied candle prefix",
        )
    _assert_configuration_unchanged(
        strategy,
        original_parameters,
        strategy_id=strategy_id,
        strategy_version=strategy_version,
        prefix_length=prefix_length,
    )
    return _validate_signals(
        output,
        strategy_id=strategy_id,
        strategy_version=strategy_version,
        symbol=symbol,
        prefix_length=prefix_length,
        earliest_information_time=earliest_information_time,
        prefix_information_cutoff=prefix_information_cutoff,
    )


def _assert_same_output(
    expected: list[Signal],
    actual: list[Signal],
    *,
    category: str,
    strategy_id: str,
    strategy_version: str,
    prefix_length: int,
    detail: str,
) -> None:
    if actual != expected:
        raise _violation(
            category,
            strategy_id=strategy_id,
            strategy_version=strategy_version,
            prefix_length=prefix_length,
            detail=detail,
        )


def assert_strategy_prefix_invariant(
    strategy: Strategy,
    candles: pl.DataFrame,
) -> StrategyCausalityReport:
    """Exhaustively prove behavioral prefix invariance on ``candles`` or raise."""
    strategy_id, strategy_version, warmup_bars, parameters = _validate_strategy(strategy)
    original_parameters = _snapshot_parameters(parameters)

    if not isinstance(candles, pl.DataFrame):
        raise TypeError("candles must be a Polars DataFrame")
    if candles.is_empty():
        raise ValueError("candles must not be empty")
    validate_strategy_input(candles)

    first_tested_prefix_length = max(1, warmup_bars)
    row_count = candles.height
    if row_count < first_tested_prefix_length + 1:
        raise ValueError(
            "candles must contain at least two eligible prefixes: "
            f"row_count={row_count}, first_tested_prefix_length={first_tested_prefix_length}"
        )

    source_before = candles.clone()
    symbol = candles["symbol"].item(0)
    timestamps: list[datetime] = candles["timestamp"].to_list()
    availability_times = [
        bar_available_at(timestamp, _TIMEFRAME_MINUTES) for timestamp in timestamps
    ]
    earliest_information_time = availability_times[0]
    outputs: dict[int, list[Signal]] = {}

    try:
        initial_full_output = _evaluate_prefix(
            strategy,
            candles.clone(),
            original_parameters=original_parameters,
            strategy_id=strategy_id,
            strategy_version=strategy_version,
            symbol=symbol,
            prefix_length=row_count,
            earliest_information_time=earliest_information_time,
            prefix_information_cutoff=availability_times[-1],
        )
        initial_full_repeat = _evaluate_prefix(
            strategy,
            candles.clone(),
            original_parameters=original_parameters,
            strategy_id=strategy_id,
            strategy_version=strategy_version,
            symbol=symbol,
            prefix_length=row_count,
            earliest_information_time=earliest_information_time,
            prefix_information_cutoff=availability_times[-1],
        )
        _assert_same_output(
            initial_full_output,
            initial_full_repeat,
            category="NONDETERMINISTIC_OUTPUT",
            strategy_id=strategy_id,
            strategy_version=strategy_version,
            prefix_length=row_count,
            detail="repeated initial evaluation of the full prefix changed output",
        )

        for prefix_length in range(first_tested_prefix_length, row_count + 1):
            cutoff = availability_times[prefix_length - 1]
            first_output = _evaluate_prefix(
                strategy,
                candles.slice(0, prefix_length).clone(),
                original_parameters=original_parameters,
                strategy_id=strategy_id,
                strategy_version=strategy_version,
                symbol=symbol,
                prefix_length=prefix_length,
                earliest_information_time=earliest_information_time,
                prefix_information_cutoff=cutoff,
            )
            repeated_output = _evaluate_prefix(
                strategy,
                candles.slice(0, prefix_length).clone(),
                original_parameters=original_parameters,
                strategy_id=strategy_id,
                strategy_version=strategy_version,
                symbol=symbol,
                prefix_length=prefix_length,
                earliest_information_time=earliest_information_time,
                prefix_information_cutoff=cutoff,
            )
            _assert_same_output(
                first_output,
                repeated_output,
                category="NONDETERMINISTIC_OUTPUT",
                strategy_id=strategy_id,
                strategy_version=strategy_version,
                prefix_length=prefix_length,
                detail="repeated evaluation of the same semantic prefix changed output",
            )
            outputs[prefix_length] = first_output

        _assert_same_output(
            initial_full_output,
            outputs[row_count],
            category="NONDETERMINISTIC_OUTPUT",
            strategy_id=strategy_id,
            strategy_version=strategy_version,
            prefix_length=row_count,
            detail="full-prefix output changed after shorter-prefix evaluations",
        )

        for prefix_length in range(first_tested_prefix_length + 1, row_count + 1):
            previous_cutoff = availability_times[prefix_length - 2]
            historical_current = [
                signal
                for signal in outputs[prefix_length]
                if signal.timestamp <= previous_cutoff
            ]
            _assert_same_output(
                outputs[prefix_length - 1],
                historical_current,
                category="PREFIX_INVARIANCE",
                strategy_id=strategy_id,
                strategy_version=strategy_version,
                prefix_length=prefix_length,
                detail=(
                    "adding one future candle inserted, removed, altered, or reordered "
                    "an already-observable Signal"
                ),
            )

        for prefix_length, output in outputs.items():
            for signal in output:
                knowledge_prefix_length = sum(
                    available_at <= signal.timestamp
                    for available_at in availability_times[:prefix_length]
                )
                if knowledge_prefix_length < first_tested_prefix_length:
                    raise _violation(
                        "KNOWLEDGE_PREFIX",
                        strategy_id=strategy_id,
                        strategy_version=strategy_version,
                        prefix_length=prefix_length,
                        decision_timestamp=signal.timestamp,
                        detail=(
                            "signal claims a decision before enough tested candle information "
                            "was available"
                        ),
                    )
                if signal not in outputs[knowledge_prefix_length]:
                    raise _violation(
                        "KNOWLEDGE_PREFIX",
                        strategy_id=strategy_id,
                        strategy_version=strategy_version,
                        prefix_length=prefix_length,
                        decision_timestamp=signal.timestamp,
                        detail=(
                            "exact Signal was absent from the prefix containing only candles "
                            "available by its decision timestamp"
                        ),
                    )

        final_output = _evaluate_prefix(
            strategy,
            candles.clone(),
            original_parameters=original_parameters,
            strategy_id=strategy_id,
            strategy_version=strategy_version,
            symbol=symbol,
            prefix_length=row_count,
            earliest_information_time=earliest_information_time,
            prefix_information_cutoff=availability_times[-1],
        )
        _assert_same_output(
            initial_full_output,
            final_output,
            category="NONDETERMINISTIC_OUTPUT",
            strategy_id=strategy_id,
            strategy_version=strategy_version,
            prefix_length=row_count,
            detail="final full-prefix output changed after all prefix checks",
        )
    finally:
        if not candles.equals(source_before):
            raise _violation(
                "INPUT_MUTATION",
                strategy_id=strategy_id,
                strategy_version=strategy_version,
                prefix_length=row_count,
                detail="the gate or strategy mutated the full source candle DataFrame",
            )

    return StrategyCausalityReport(
        gate_version=STRATEGY_CAUSALITY_GATE_VERSION,
        strategy_id=strategy_id,
        strategy_version=strategy_version,
        symbol=symbol,
        timeframe_minutes=_TIMEFRAME_MINUTES,
        warmup_bars=warmup_bars,
        row_count=row_count,
        first_tested_prefix_length=first_tested_prefix_length,
        tested_prefix_count=row_count - first_tested_prefix_length + 1,
        full_signal_count=len(outputs[row_count]),
        first_candle_timestamp=timestamps[0],
        last_candle_timestamp=timestamps[-1],
        last_information_available_at=availability_times[-1],
    )
