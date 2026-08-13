from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from algo_trader.data import MarketDataConfig, ParquetMarketDataStore

IST = timezone(timedelta(hours=5, minutes=30), name="Asia/Kolkata")


def ist_datetime(day: int, hour: int = 9, minute: int = 15) -> datetime:
    return datetime(2024, 1, day, hour, minute, tzinfo=IST)


def write_candles(path: Path, timestamps: list[datetime], *, integer_volume: bool) -> None:
    count = len(timestamps)
    volume_type = pa.int64() if integer_volume else pa.float64()
    table = pa.table(
        {
            "date": pa.array(
                [timestamp.astimezone(UTC) for timestamp in timestamps],
                type=pa.timestamp("us", tz="UTC"),
            ),
            "open": pa.array([100.0 + index for index in range(count)], type=pa.float64()),
            "high": pa.array([101.0 + index for index in range(count)], type=pa.float64()),
            "low": pa.array([99.0 + index for index in range(count)], type=pa.float64()),
            "close": pa.array([100.5 + index for index in range(count)], type=pa.float64()),
            "volume": pa.array([1_000 + index for index in range(count)], type=volume_type),
        }
    )
    pq.write_table(table, path)


@pytest.fixture
def market_data_path(tmp_path: Path) -> Path:
    write_candles(
        tmp_path / "A&B.parquet",
        [ist_datetime(1, 9, 15), ist_datetime(1, 9, 20), ist_datetime(1, 9, 25)],
        integer_volume=True,
    )
    write_candles(
        tmp_path / "NEW-LIST.parquet",
        [ist_datetime(2, 9, 15), ist_datetime(2, 9, 20)],
        integer_volume=False,
    )
    write_candles(
        tmp_path / "ZED.parquet",
        [ist_datetime(1, 9, 15), ist_datetime(1, 9, 25)],
        integer_volume=False,
    )
    return tmp_path


@pytest.fixture
def store(market_data_path: Path) -> ParquetMarketDataStore:
    return ParquetMarketDataStore(MarketDataConfig(dataset_path=market_data_path))


def test_store_requires_existing_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="directory does not exist"):
        ParquetMarketDataStore(MarketDataConfig(dataset_path=tmp_path / "missing"))


def test_symbol_discovery_is_sorted_and_preserves_filename_characters(
    store: ParquetMarketDataStore,
) -> None:
    assert store.list_symbols() == ["A&B", "NEW-LIST", "ZED"]


def test_one_symbol_query_uses_half_open_window(store: ParquetMarketDataStore) -> None:
    candles = store.load_candles("A&B", ist_datetime(1, 9, 15), ist_datetime(1, 9, 25))

    assert candles.height == 2
    assert candles["symbol"].to_list() == ["A&B", "A&B"]
    assert candles["timestamp"].cast(pl.Int64).to_list() == [
        int(ist_datetime(1, 9, 15).timestamp() * 1_000_000),
        int(ist_datetime(1, 9, 20).timestamp() * 1_000_000),
    ]


def test_query_accepts_aware_boundaries_in_another_timezone(
    store: ParquetMarketDataStore,
) -> None:
    start_utc = ist_datetime(1, 9, 15).astimezone(UTC)
    end_utc = ist_datetime(1, 9, 20).astimezone(UTC)

    candles = store.load_candles("A&B", start_utc, end_utc)

    assert candles.height == 1
    assert candles.schema["timestamp"] == pl.Datetime("us", "Asia/Kolkata")
    timestamp = candles["timestamp"].item()
    assert timestamp.tzinfo == ZoneInfo("Asia/Kolkata")
    assert timestamp == ist_datetime(1, 9, 15)


def test_multi_symbol_query_is_ordered_by_timestamp_then_symbol(
    store: ParquetMarketDataStore,
) -> None:
    candles = store.load_candles(
        ["ZED", "A&B"],
        ist_datetime(1, 9, 15),
        ist_datetime(1, 9, 30),
    )
    ordering = candles.select(pl.col("timestamp").cast(pl.Int64), "symbol").rows()

    assert ordering == sorted(ordering)
    assert candles["symbol"].to_list() == ["A&B", "ZED", "A&B", "A&B", "ZED"]


@pytest.mark.parametrize("boundary", ["start", "end"])
def test_query_rejects_timezone_naive_boundaries(
    store: ParquetMarketDataStore,
    boundary: str,
) -> None:
    kwargs = {"start": ist_datetime(1), "end": ist_datetime(2)}
    kwargs[boundary] = datetime(2024, 1, 1, 9, 15)

    with pytest.raises(ValueError, match=f"{boundary} must be timezone-aware"):
        store.load_candles("A&B", **kwargs)


def test_unknown_symbol_is_rejected(store: ParquetMarketDataStore) -> None:
    with pytest.raises(ValueError, match="unknown symbol"):
        store.load_candles("MISSING", ist_datetime(1), ist_datetime(2))


@pytest.mark.parametrize(
    ("start", "end"),
    [(ist_datetime(1), ist_datetime(1)), (ist_datetime(2), ist_datetime(1))],
)
def test_invalid_time_range_is_rejected(
    store: ParquetMarketDataStore,
    start: datetime,
    end: datetime,
) -> None:
    with pytest.raises(ValueError, match="start must be earlier than end"):
        store.load_candles("A&B", start, end)


def test_symbols_with_different_coverage_query_without_failure(
    store: ParquetMarketDataStore,
) -> None:
    candles = store.load_candles(["A&B", "NEW-LIST"], ist_datetime(1), ist_datetime(3))

    assert candles.group_by("symbol").len().sort("symbol").rows() == [
        ("A&B", 3),
        ("NEW-LIST", 2),
    ]


def test_empty_valid_window_has_stable_schema(store: ParquetMarketDataStore) -> None:
    candles = store.load_candles("A&B", ist_datetime(5), ist_datetime(6))

    assert candles.is_empty()
    assert candles.schema == pl.Schema(
        {
            "timestamp": pl.Datetime("us", "Asia/Kolkata"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "symbol": pl.String,
        }
    )


def test_symbol_coverage_uses_market_timezone(store: ParquetMarketDataStore) -> None:
    coverage = store.get_symbol_coverage("A&B")

    assert coverage.symbol == "A&B"
    assert coverage.first_timestamp == ist_datetime(1, 9, 15)
    assert coverage.last_timestamp == ist_datetime(1, 9, 25)
    assert coverage.row_count == 3
    assert coverage.first_timestamp is not None
    assert coverage.first_timestamp.tzinfo == ZoneInfo("Asia/Kolkata")


def test_bulk_coverage_is_explicit_sorted_and_handles_different_listing_dates(
    store: ParquetMarketDataStore,
) -> None:
    coverages = store.get_symbols_coverage(["NEW-LIST", "A&B"])

    assert [coverage.symbol for coverage in coverages] == ["A&B", "NEW-LIST"]
    assert [coverage.row_count for coverage in coverages] == [3, 2]
    assert coverages[1].first_timestamp == ist_datetime(2, 9, 15)


def test_all_symbol_coverage_is_only_scanned_when_explicitly_requested(
    store: ParquetMarketDataStore,
) -> None:
    coverages = store.get_symbols_coverage()

    assert [coverage.symbol for coverage in coverages] == ["A&B", "NEW-LIST", "ZED"]
