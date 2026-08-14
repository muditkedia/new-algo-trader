"""Pure historical-equity horizon and scan-universe selection."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from algo_trader.data import SymbolCoverage
from algo_trader.oos.models import (
    STANDARD_OOS_PARTITION_POLICY,
    OOSPartitionPolicy,
)

MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")


def derive_equity_data_horizon(
    coverages: Iterable[SymbolCoverage],
    *,
    policy: OOSPartitionPolicy = STANDARD_OOS_PARTITION_POLICY,
) -> tuple[date, date]:
    """Return the earliest through latest-plus-one equity coverage dates."""
    normalized = _normalize_nonempty_coverages(coverages, policy)
    if not normalized:
        raise ValueError("no non-reference equity coverage is available")
    return (
        min(first_date for _, first_date, _ in normalized),
        max(last_date for _, _, last_date in normalized) + timedelta(days=1),
    )


def select_historically_available_equities(
    coverages: Iterable[SymbolCoverage],
    *,
    start_date: date,
    end_date: date,
    policy: OOSPartitionPolicy = STANDARD_OOS_PARTITION_POLICY,
) -> tuple[str, ...]:
    """Select every equity whose raw coverage intersects ``[start_date, end_date)``."""
    _require_date(start_date, "start_date")
    _require_date(end_date, "end_date")
    if start_date >= end_date:
        raise ValueError("start_date must be earlier than end_date")
    normalized = _normalize_nonempty_coverages(coverages, policy)
    return tuple(
        sorted(
            symbol
            for symbol, first_date, last_date in normalized
            if first_date < end_date and last_date >= start_date
        )
    )


def _normalize_nonempty_coverages(
    coverages: Iterable[SymbolCoverage],
    policy: OOSPartitionPolicy,
) -> tuple[tuple[str, date, date], ...]:
    if isinstance(coverages, str) or coverages is None:
        raise TypeError("coverages must be a non-string iterable")
    if not isinstance(policy, OOSPartitionPolicy):
        raise TypeError("policy must be an OOSPartitionPolicy")
    try:
        selected = tuple(coverages)
    except TypeError as error:
        raise TypeError("coverages must be iterable") from error

    excluded = set(policy.excluded_reference_symbols)
    normalized: list[tuple[str, date, date]] = []
    seen_symbols: set[str] = set()
    for coverage in selected:
        if not isinstance(coverage, SymbolCoverage):
            raise TypeError("all coverages must be SymbolCoverage instances")
        symbol = coverage.symbol.strip()
        if not symbol:
            raise ValueError("coverage symbol must be non-empty")
        if symbol in seen_symbols:
            raise ValueError(f"duplicate SymbolCoverage for symbol: {symbol}")
        seen_symbols.add(symbol)
        if coverage.row_count < 0:
            raise ValueError("coverage row_count must be non-negative")
        if coverage.row_count == 0:
            if coverage.first_timestamp is not None or coverage.last_timestamp is not None:
                raise ValueError("empty coverage must not contain timestamp bounds")
            continue
        if coverage.first_timestamp is None or coverage.last_timestamp is None:
            raise ValueError("non-empty coverage requires both timestamp bounds")
        first_date = _market_date(coverage.first_timestamp, "first_timestamp")
        last_date = _market_date(coverage.last_timestamp, "last_timestamp")
        if first_date > last_date:
            raise ValueError("coverage first_timestamp must not follow last_timestamp")
        if symbol not in excluded:
            normalized.append((symbol, first_date, last_date))
    return tuple(sorted(normalized))


def _market_date(value: datetime, name: str) -> date:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(MARKET_TIMEZONE).date()


def _require_date(value: object, name: str) -> None:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(f"{name} must be a date")
