"""Read-only early-path diagnostics for Strategy 1 research evidence.

Uses the cumulative research JSON plus raw canonical 5-minute Parquets. It does
not run a backtest, mutate research artifacts, or change OOS governance.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from statistics import median
from typing import Any

from algo_trader.data import MarketDataConfig, ParquetMarketDataStore

SCOPE = "liquidity-shock-exhaustion-reclaim"
STRATEGY_VERSION = "1.1.0"
DEFAULT_RESEARCH_JSON = Path(
    "results/research/liquidity-shock-exhaustion-reclaim/UPLOAD_FOR_REVIEW/"
    "strategy1_research_history.json"
)
DEFAULT_DATA_PATH = Path("data/market/NSE/5M")
DEFAULT_DERIVED_JSON = Path(
    "results/research/liquidity-shock-exhaustion-reclaim/diagnostics/"
    "strategy1_early_path_diagnostics.json"
)


@dataclass(frozen=True)
class PathRow:
    run_id: str
    sample: str
    symbol: str
    side: str
    entry_timestamp: str
    exit_timestamp: str
    exit_reason: str
    exit_reason_detail: str | None
    net_pnl: float
    winner: bool
    reclaim_atr: float | None
    initial_r_per_unit: float
    mfe_5m_r: float
    mae_5m_r: float
    mfe_10m_r: float
    mae_10m_r: float
    mfe_15m_r: float
    mae_15m_r: float
    minutes_to_0_25r: float | None
    minutes_to_0_50r: float | None
    minutes_to_0_75r: float | None
    first_level_failure_minutes: float | None
    max_favorable_r_before_level_failure: float | None
    reached_0_25r_before_minus_0_50r: bool | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Strategy 1 early post-entry path using development plus "
            "TRAINING_ALLOWED OOS evidence only."
        )
    )
    parser.add_argument("--research-json", type=Path, default=DEFAULT_RESEARCH_JSON)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_DERIVED_JSON)
    return parser.parse_args()


def _d(value: Any) -> Decimal:
    return Decimal(str(value))


def _dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"expected timezone-aware timestamp: {value}")
    return parsed


def _oos_states(payload: dict[str, Any]) -> dict[str, str]:
    plan = payload.get("oos_plan") or {}
    return {
        item["window_id"]: item["state"]
        for item in plan.get("oos_windows", [])
        if isinstance(item, dict) and "window_id" in item and "state" in item
    }


def _strategy_version(run: dict[str, Any]) -> str | None:
    versions = run.get("strategy_versions") or []
    for pair in versions:
        if isinstance(pair, (list, tuple)) and len(pair) == 2 and pair[0] == SCOPE:
            return str(pair[1])
    return None


def _selected_runs(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    states = _oos_states(payload)
    runs = payload.get("runs") or []

    development = [
        run
        for run in runs
        if run.get("phase") == "development"
        and _strategy_version(run) == STRATEGY_VERSION
    ]
    if not development:
        raise RuntimeError("no Strategy 1 v1.1.0 development run found")
    # The cumulative artifact is chronological; use the most recent repaired
    # development run and do not double-count older equivalent reruns.
    selected: list[tuple[str, dict[str, Any]]] = [("DEV", development[-1])]

    for run in runs:
        if run.get("phase") != "oos" or _strategy_version(run) != STRATEGY_VERSION:
            continue
        window_id = run.get("window_id")
        if states.get(window_id) == "TRAINING_ALLOWED":
            selected.append((str(window_id).upper(), run))

    if len(selected) == 1:
        raise RuntimeError(
            "no TRAINING_ALLOWED Strategy 1 OOS evidence found; rebuild the two-file "
            "review bundle after authorization before running this analyzer"
        )
    return selected


def _trade_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    result = run.get("backtest_result") or {}
    return result.get("actual_trade_records") or []


def _excursions_r(
    rows: list[dict[str, Any]],
    *,
    side: str,
    entry: Decimal,
    initial_r: Decimal,
) -> tuple[Decimal, Decimal]:
    if not rows:
        return Decimal("0"), Decimal("0")
    max_high = max(_d(row["high"]) for row in rows)
    min_low = min(_d(row["low"]) for row in rows)
    if side == "LONG":
        favorable = max(max_high - entry, Decimal("0"))
        adverse = min(min_low - entry, Decimal("0"))
    else:
        favorable = max(entry - min_low, Decimal("0"))
        adverse = min(entry - max_high, Decimal("0"))
    return favorable / initial_r, adverse / initial_r


def _analyze_trade(
    store: ParquetMarketDataStore,
    run_id: str,
    sample: str,
    record: dict[str, Any],
) -> PathRow:
    trade = record["trade"]
    signal = trade["signal"]
    feature = signal["feature_snapshot"]
    params = signal["strategy_parameters"]
    symbol = signal["symbol"]
    side = signal["side"]
    entry_at = _dt(trade["entry_fill"]["timestamp"])
    exit_at = _dt(trade["exit_fill"]["timestamp"])
    entry = _d(trade["entry_fill"]["price"])
    stop = _d(feature["stop_reference_price"])
    level = _d(feature["level_price"])
    timeframe = int(params.get("timeframe_minutes", 5))
    if timeframe <= 0:
        raise RuntimeError(f"invalid timeframe for {run_id} {symbol}: {timeframe}")

    initial_r = entry - stop if side == "LONG" else stop - entry
    if initial_r <= 0:
        raise RuntimeError(
            f"invalid actual-fill R geometry for {run_id} {symbol}: "
            f"entry={entry} stop={stop} side={side}"
        )

    frame = store.load_candles(symbol, entry_at, exit_at)
    rows = frame.select("timestamp", "high", "low", "close").to_dicts()

    horizon: dict[int, tuple[Decimal, Decimal]] = {}
    for minutes in (5, 10, 15):
        cutoff = entry_at + timedelta(minutes=minutes)
        chosen = [row for row in rows if row["timestamp"] < cutoff]
        horizon[minutes] = _excursions_r(
            chosen, side=side, entry=entry, initial_r=initial_r
        )

    thresholds = {
        Decimal("0.25"): None,
        Decimal("0.50"): None,
        Decimal("0.75"): None,
    }
    first_minus_half: datetime | None = None
    level_failure: datetime | None = None
    max_favorable_r = Decimal("0")
    max_before_failure: Decimal | None = None

    for row in rows:
        bar_start = row["timestamp"]
        known_at = bar_start + timedelta(minutes=timeframe)
        high = _d(row["high"])
        low = _d(row["low"])
        close = _d(row["close"])
        if side == "LONG":
            favorable_r = max((high - entry) / initial_r, Decimal("0"))
            adverse_r = min((low - entry) / initial_r, Decimal("0"))
            failed = close < level
        else:
            favorable_r = max((entry - low) / initial_r, Decimal("0"))
            adverse_r = min((entry - high) / initial_r, Decimal("0"))
            failed = close > level

        max_favorable_r = max(max_favorable_r, favorable_r)
        for threshold in thresholds:
            if thresholds[threshold] is None and favorable_r >= threshold:
                thresholds[threshold] = known_at
        if first_minus_half is None and adverse_r <= Decimal("-0.50"):
            first_minus_half = known_at
        if level_failure is None and failed:
            level_failure = known_at
            max_before_failure = max_favorable_r

    first_quarter = thresholds[Decimal("0.25")]
    if first_quarter is None and first_minus_half is None:
        quarter_before_half = None
    elif first_quarter is None:
        quarter_before_half = False
    elif first_minus_half is None:
        quarter_before_half = True
    elif first_quarter == first_minus_half:
        # OHLC does not reveal intrabar ordering.
        quarter_before_half = None
    else:
        quarter_before_half = first_quarter < first_minus_half

    def minutes_to(value: datetime | None) -> float | None:
        if value is None:
            return None
        return (value - entry_at).total_seconds() / 60.0

    mfe5, mae5 = horizon[5]
    mfe10, mae10 = horizon[10]
    mfe15, mae15 = horizon[15]
    net_pnl = float(_d(trade["net_pnl"]))
    reclaim = feature.get("reclaim_atr")
    return PathRow(
        run_id=run_id,
        sample=sample,
        symbol=symbol,
        side=side,
        entry_timestamp=entry_at.isoformat(),
        exit_timestamp=exit_at.isoformat(),
        exit_reason=trade["exit_reason"],
        exit_reason_detail=trade.get("exit_reason_detail"),
        net_pnl=net_pnl,
        winner=net_pnl > 0,
        reclaim_atr=float(reclaim) if reclaim is not None else None,
        initial_r_per_unit=float(initial_r),
        mfe_5m_r=float(mfe5),
        mae_5m_r=float(mae5),
        mfe_10m_r=float(mfe10),
        mae_10m_r=float(mae10),
        mfe_15m_r=float(mfe15),
        mae_15m_r=float(mae15),
        minutes_to_0_25r=minutes_to(first_quarter),
        minutes_to_0_50r=minutes_to(thresholds[Decimal("0.50")]),
        minutes_to_0_75r=minutes_to(thresholds[Decimal("0.75")]),
        first_level_failure_minutes=minutes_to(level_failure),
        max_favorable_r_before_level_failure=(
            float(max_before_failure) if max_before_failure is not None else None
        ),
        reached_0_25r_before_minus_0_50r=quarter_before_half,
    )


def _fmt(value: float | None, width: int = 6) -> str:
    return "-".rjust(width) if value is None else f"{value:.2f}".rjust(width)


def _med(rows: list[PathRow], field: str) -> float | None:
    values = [getattr(row, field) for row in rows if getattr(row, field) is not None]
    return float(median(values)) if values else None


def _print_trade_table(rows: list[PathRow]) -> None:
    print("\n=== PER-TRADE EARLY PATH ===")
    print(
        "sample   W/L symbol       side   net_pnl  "
        "MFE5  MAE5 MFE10 MAE10 MFE15 MAE15  t25  t50  t75  lvlFail"
    )
    for row in rows:
        print(
            f"{row.sample:<8} {'W' if row.winner else 'L':<3} "
            f"{row.symbol:<12} {row.side:<6} {row.net_pnl:>8.2f} "
            f"{_fmt(row.mfe_5m_r, 5)} {_fmt(row.mae_5m_r, 5)} "
            f"{_fmt(row.mfe_10m_r, 5)} {_fmt(row.mae_10m_r, 5)} "
            f"{_fmt(row.mfe_15m_r, 5)} {_fmt(row.mae_15m_r, 5)} "
            f"{_fmt(row.minutes_to_0_25r, 4)} {_fmt(row.minutes_to_0_50r, 4)} "
            f"{_fmt(row.minutes_to_0_75r, 4)} {_fmt(row.first_level_failure_minutes, 7)}"
        )


def _print_summary(rows: list[PathRow]) -> None:
    print("\n=== WINNER / LOSER MEDIANS (R) ===")
    print("sample   class  n   MFE5   MAE5  MFE10  MAE10  MFE15  MAE15")
    samples = list(dict.fromkeys(row.sample for row in rows))
    for sample in samples + ["ALL"]:
        source = rows if sample == "ALL" else [row for row in rows if row.sample == sample]
        for winner, label in ((True, "WIN"), (False, "LOSS")):
            group = [row for row in source if row.winner is winner]
            if not group:
                continue
            print(
                f"{sample:<8} {label:<5} {len(group):>2} "
                f"{_fmt(_med(group, 'mfe_5m_r'))} {_fmt(_med(group, 'mae_5m_r'))} "
                f"{_fmt(_med(group, 'mfe_10m_r'))} {_fmt(_med(group, 'mae_10m_r'))} "
                f"{_fmt(_med(group, 'mfe_15m_r'))} {_fmt(_med(group, 'mae_15m_r'))}"
            )


def _rate(rows: list[PathRow], predicate: Any) -> tuple[int, int, float]:
    count = sum(1 for row in rows if predicate(row))
    total = len(rows)
    return count, total, (count / total if total else 0.0)


def _print_candidate_diagnostics(rows: list[PathRow]) -> None:
    print("\n=== CANDIDATE FAILURE / PROGRESS DIAGNOSTICS ===")
    print("Rates are descriptive only; no trading rule is selected by this script.")
    for sample in list(dict.fromkeys(row.sample for row in rows)) + ["ALL"]:
        source = rows if sample == "ALL" else [row for row in rows if row.sample == sample]
        print(f"\n[{sample}] n={len(source)}")
        for label, predicate in (
            (
                "+0.25R achieved by 5m",
                lambda r: r.minutes_to_0_25r is not None and r.minutes_to_0_25r <= 5,
            ),
            (
                "+0.25R achieved by 10m",
                lambda r: r.minutes_to_0_25r is not None and r.minutes_to_0_25r <= 10,
            ),
            (
                "+0.25R achieved by 15m",
                lambda r: r.minutes_to_0_25r is not None and r.minutes_to_0_25r <= 15,
            ),
            (
                "+0.50R achieved by 10m",
                lambda r: r.minutes_to_0_50r is not None and r.minutes_to_0_50r <= 10,
            ),
            (
                "level failed by 10m",
                lambda r: r.first_level_failure_minutes is not None
                and r.first_level_failure_minutes <= 10,
            ),
            (
                "level failed by 15m",
                lambda r: r.first_level_failure_minutes is not None
                and r.first_level_failure_minutes <= 15,
            ),
            ("+0.25R before -0.50R", lambda r: r.reached_0_25r_before_minus_0_50r is True),
        ):
            count, total, rate = _rate(source, predicate)
            winners = [row for row in source if row.winner]
            losers = [row for row in source if not row.winner]
            wc, wt, wr = _rate(winners, predicate)
            lc, lt, lr = _rate(losers, predicate)
            print(
                f"  {label:<27} {count:>2}/{total:<2} {rate:>6.1%} | "
                f"WIN {wc:>2}/{wt:<2} {wr:>6.1%} | LOSS {lc:>2}/{lt:<2} {lr:>6.1%}"
            )


def main() -> None:
    args = _parse_args()
    payload = json.loads(args.research_json.read_text(encoding="utf-8"))
    if payload.get("research_scope_id") != SCOPE:
        raise RuntimeError("research JSON does not belong to Strategy 1 scope")

    selected = _selected_runs(payload)
    print("=== STRATEGY 1 EARLY-PATH DIAGNOSTIC ===")
    print("READ ONLY: no backtest, no OOS mutation, no artifact mutation")
    print("Selected evidence:")
    for sample, run in selected:
        print(f"  {sample}: {run['run_id']}")

    store = ParquetMarketDataStore(MarketDataConfig(dataset_path=args.data_path))
    rows: list[PathRow] = []
    for sample, run in selected:
        records = _trade_rows(run)
        print(f"Analyzing {sample}: {len(records)} actual trades")
        for record in records:
            rows.append(_analyze_trade(store, run["run_id"], sample, record))

    _print_trade_table(rows)
    _print_summary(rows)
    _print_candidate_diagnostics(rows)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "schema_version": "1",
        "research_scope_id": SCOPE,
        "strategy_version": STRATEGY_VERSION,
        "source_research_json": str(args.research_json),
        "selected_run_ids": [run["run_id"] for _, run in selected],
        "rows": [asdict(row) for row in rows],
    }
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8", newline="\n")
    print(f"\nDerived diagnostic JSON (internal only): {args.output_json}")
    print("Do not treat this file as a backtest result or OOS artifact.")


if __name__ == "__main__":
    main()
