from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import pytest

from synthgen.config import ProfilingSettings
from synthgen.connectors.driver import max_columns, sqlite_driver
from synthgen.connectors.sqlite import SqliteConnector
from synthgen.metadata.provider import SourceDoc, YamlMetadataProvider
from synthgen.profiling.batching import batched_sample, sample_rows
from synthgen.profiling.models import InferredType
from synthgen.profiling.profiler import profile_table

PROFILE_NOW = datetime(2026, 1, 1, tzinfo=UTC)

ROWS = 40


def build(driver: ModuleType, path: Path, width: int) -> Path:
    """A table with an id, `width` numeric columns and a text column every 100th position."""
    conn = driver.connect(str(path))
    kinds = ["TEXT" if i % 100 == 0 else "REAL" for i in range(width)]
    declared = ", ".join(f"c{i} {k}" for i, k in enumerate(kinds))
    conn.execute(f"CREATE TABLE wide (id INTEGER PRIMARY KEY, {declared})")
    rows = [
        (r, *(f"v{r % 3}" if k == "TEXT" else float(r * (i + 1)) for i, k in enumerate(kinds)))
        for r in range(1, ROWS + 1)
    ]
    conn.executemany(f"INSERT INTO wide VALUES ({', '.join('?' * (width + 1))})", rows)
    conn.commit()
    conn.close()
    return path


def profile(path: Path, settings: ProfilingSettings) -> tuple[int, int]:
    connector = SqliteConnector(path)
    try:
        provider = YamlMetadataProvider(SourceDoc(source="wide"), connector)
        result = profile_table(
            "wide",
            connector,
            provider.table("wide"),
            [],
            parents={},
            settings=settings,
            seed=1,
            now=PROFILE_NOW,
        )
    finally:
        connector.close()
    numeric = sum(c.inferred_type is InferredType.FLOAT for c in result.columns)
    return len(result.columns), numeric


def test_sample_rows_shrink_with_width(profiling_settings: ProfilingSettings) -> None:
    assert sample_rows(10, profiling_settings) == profiling_settings.sample_size
    assert sample_rows(3_000, profiling_settings) == profiling_settings.max_sample_cells // 3_000
    tiny = profiling_settings.model_copy(update={"max_sample_cells": 5})
    assert sample_rows(3_000, tiny) == 1


def test_batches_select_the_same_rows(tmp_path: Path) -> None:
    path = build(sqlite3, tmp_path / "w.db", 30)
    connector = SqliteConnector(path)
    columns = ["id", *(f"c{i}" for i in range(30))]
    whole = batched_sample(connector, "wide", columns, rows=15, seed=4, batch=1_000)
    pieces = batched_sample(connector, "wide", columns, rows=15, seed=4, batch=7)
    connector.close()
    assert pieces.equals(whole)
    assert whole.num_rows == 15


def test_fifteen_hundred_columns_on_standard_sqlite(
    tmp_path: Path, profiling_settings: ProfilingSettings
) -> None:
    columns, numeric = profile(build(sqlite3, tmp_path / "w.db", 1_500), profiling_settings)
    assert columns == 1_501
    assert numeric == 1_485


@pytest.fixture(scope="module")
def wide_driver() -> Iterator[ModuleType]:
    driver = sqlite_driver()
    if max_columns(driver) < 3_001:
        pytest.skip("wide SQLite build not installed (make sqlite-wide)")
    yield driver


def test_three_thousand_columns(
    wide_driver: ModuleType, tmp_path: Path, profiling_settings: ProfilingSettings
) -> None:
    columns, numeric = profile(build(wide_driver, tmp_path / "w.db", 3_000), profiling_settings)
    assert columns == 3_001
    assert numeric == 2_970
