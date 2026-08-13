"""Read-only DuckDB/Polars access to canonical Parquet market data."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import duckdb
import polars as pl
from pydantic import BaseModel, ConfigDict

MARKET_TIMEZONE_NAME = "Asia/Kolkata"
MARKET_TIMEZONE = ZoneInfo(MARKET_TIMEZONE_NAME)

CANONICAL_CANDLE_COLUMNS = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "symbol",
)


class MarketDataConfig(BaseModel):
    """Validated assumptions for the canonical five-minute NSE dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_path: Path = Path("data/market/NSE/5M")
    timeframe_minutes: Literal[5] = 5
    market_timezone: Literal["Asia/Kolkata"] = MARKET_TIMEZONE_NAME


class SymbolCoverage(BaseModel):
    """Available raw history for one symbol."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    first_timestamp: datetime | None
    last_timestamp: datetime | None
    row_count: int


class ParquetMarketDataStore:
    """Thin, read-only query layer over one-Parquet-file-per-symbol data.

    Candle windows use half-open ``[start, end)`` semantics. Query boundaries
    must be timezone-aware and may use any timezone; they are normalized to
    Asia/Kolkata before DuckDB applies the predicates.
    """

    def __init__(self, config: MarketDataConfig | None = None) -> None:
        self.config = config or MarketDataConfig()
        self._dataset_path = self.config.dataset_path.resolve()
        if not self._dataset_path.is_dir():
            raise ValueError(f"market-data directory does not exist: {self.config.dataset_path}")

        parquet_files = sorted(
            (
                path
                for path in self._dataset_path.iterdir()
                if path.is_file() and path.suffix.lower() == ".parquet"
            ),
            key=lambda path: path.stem,
        )
        self._symbol_files = {path.stem: path for path in parquet_files}

    def list_symbols(self) -> list[str]:
        """Return symbols derived verbatim from filenames in sorted order."""
        return list(self._symbol_files)

    def get_symbol_coverage(self, symbol: str) -> SymbolCoverage:
        """Return timestamp bounds and row count without loading candle rows into Python."""
        path = self._file_for_symbol(symbol)
        with self._connect() as connection:
            return self._query_coverage(connection, symbol, path)

    def get_symbols_coverage(
        self,
        symbols: Iterable[str] | None = None,
    ) -> list[SymbolCoverage]:
        """Return requested coverage; passing ``None`` explicitly requests all symbols."""
        selected = self._normalize_symbols(self.list_symbols() if symbols is None else symbols)
        with self._connect() as connection:
            return [
                self._query_coverage(connection, symbol, self._symbol_files[symbol])
                for symbol in selected
            ]

    def load_candles(
        self,
        symbols: str | Iterable[str],
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """Load raw candles for the half-open interval ``[start, end)``.

        Only the requested files and source columns are scanned. Timestamps in
        the returned Polars frame always have the ``Asia/Kolkata`` time zone.
        """
        selected = self._normalize_symbols([symbols] if isinstance(symbols, str) else symbols)
        normalized_start = self._normalize_boundary(start, "start")
        normalized_end = self._normalize_boundary(end, "end")
        if normalized_start >= normalized_end:
            raise ValueError("start must be earlier than end")

        paths = [str(self._symbol_files[symbol]) for symbol in selected]
        query = """
            SELECT
                date AS timestamp,
                open,
                high,
                low,
                close,
                CAST(volume AS DOUBLE) AS volume,
                parse_filename(filename, true)::VARCHAR AS symbol
            FROM read_parquet(?, filename = true, union_by_name = true)
            WHERE date >= ? AND date < ?
            ORDER BY timestamp, symbol
        """
        with self._connect() as connection:
            arrow_result = connection.execute(
                query,
                [paths, normalized_start, normalized_end],
            ).arrow()
            candles = pl.from_arrow(arrow_result.read_all())

        return candles.select(CANONICAL_CANDLE_COLUMNS)

    def _normalize_symbols(self, symbols: Iterable[str]) -> list[str]:
        selected = sorted(set(symbols))
        if not selected:
            raise ValueError("at least one symbol is required")

        unknown = [symbol for symbol in selected if symbol not in self._symbol_files]
        if unknown:
            raise ValueError(f"unknown symbol(s): {', '.join(unknown)}")
        return selected

    def _file_for_symbol(self, symbol: str) -> Path:
        try:
            return self._symbol_files[symbol]
        except KeyError as error:
            raise ValueError(f"unknown symbol: {symbol}") from error

    @staticmethod
    def _normalize_boundary(value: datetime, name: str) -> datetime:
        if not isinstance(value, datetime):
            raise TypeError(f"{name} must be a datetime")
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{name} must be timezone-aware")
        return value.astimezone(MARKET_TIMEZONE)

    @staticmethod
    def _connect() -> duckdb.DuckDBPyConnection:
        connection = duckdb.connect()
        connection.execute(f"SET TimeZone = '{MARKET_TIMEZONE_NAME}'")
        return connection

    @staticmethod
    def _query_coverage(
        connection: duckdb.DuckDBPyConnection,
        symbol: str,
        path: Path,
    ) -> SymbolCoverage:
        first_value, last_value, row_count = connection.execute(
            """
            SELECT min(date)::VARCHAR, max(date)::VARCHAR, count(*)
            FROM read_parquet(?)
            """,
            [str(path)],
        ).fetchone()
        return SymbolCoverage(
            symbol=symbol,
            first_timestamp=_parse_market_timestamp(first_value),
            last_timestamp=_parse_market_timestamp(last_value),
            row_count=row_count,
        )


def _parse_market_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value).astimezone(MARKET_TIMEZONE)
