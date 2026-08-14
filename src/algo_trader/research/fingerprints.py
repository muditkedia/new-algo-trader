"""Deterministic non-secret research input and environment fingerprints."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import polars as pl

RESEARCH_PACKAGES = (
    "duckdb",
    "lightgbm",
    "numpy",
    "openpyxl",
    "polars",
    "pydantic",
    "scikit-learn",
    "smartapi-python",
    "tzdata",
)


@dataclass(frozen=True, slots=True)
class MarketDataFileManifest:
    relative_path: str
    symbol: str
    sha256: str
    row_count: int
    first_timestamp: str | None
    last_timestamp: str | None
    schema: tuple[tuple[str, str], ...]
    size: int


def canonical_fingerprint(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_market_data_manifest(
    paths: tuple[Path, ...],
    *,
    dataset_root: Path,
    cache_path: Path,
) -> tuple[tuple[MarketDataFileManifest, ...], str]:
    """Hash relevant Parquet files, reusing strong hashes only for unchanged metadata."""
    root = Path(dataset_root).resolve()
    cache = _read_cache(cache_path)
    updated: dict[str, object] = {}
    records: list[MarketDataFileManifest] = []
    for raw_path in sorted(paths, key=lambda value: str(value).casefold()):
        path = Path(raw_path).resolve()
        relative = path.relative_to(root).as_posix()
        stat = path.stat()
        identity = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        cached = cache.get(relative)
        if isinstance(cached, dict) and cached.get("identity") == identity:
            cached_manifest = dict(cached["manifest"])
            cached_manifest["schema"] = tuple(
                tuple(item) for item in cached_manifest["schema"]
            )
            record = MarketDataFileManifest(**cached_manifest)
        else:
            schema = tuple(
                (name, str(dtype)) for name, dtype in pl.read_parquet_schema(path).items()
            )
            column_names = {name for name, _dtype in schema}
            if "date" not in column_names:
                raise ValueError(
                    f"market-data Parquet is missing canonical 'date' column: {path}"
                )
            summary = (
                pl.scan_parquet(path)
                .select(
                    pl.len().alias("row_count"),
                    pl.col("date").min().alias("first_timestamp"),
                    pl.col("date").max().alias("last_timestamp"),
                )
                .collect()
                .row(0, named=True)
            )
            record = MarketDataFileManifest(
                relative_path=relative,
                symbol=path.stem,
                sha256=file_sha256(path),
                row_count=summary["row_count"],
                first_timestamp=_timestamp_text(summary["first_timestamp"]),
                last_timestamp=_timestamp_text(summary["last_timestamp"]),
                schema=schema,
                size=stat.st_size,
            )
        updated[relative] = {"identity": identity, "manifest": asdict(record)}
        records.append(record)
    _write_cache(cache_path, updated)
    canonical = [asdict(record) for record in records]
    return tuple(records), canonical_fingerprint(canonical)


def environment_snapshot() -> dict[str, object]:
    packages: dict[str, str] = {}
    for name in RESEARCH_PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "NOT_INSTALLED"
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


def _timestamp_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, datetime):
        raise TypeError("Parquet timestamp summary must materialize as datetime")
    return value.isoformat()


def _read_cache(path: Path) -> dict[str, object]:
    source = Path(path)
    if not source.exists():
        return {}
    parsed = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("market-data fingerprint cache must be an object")
    return parsed


def _write_cache(path: Path, value: dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, destination)
