from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from synthgen.connectors import sqlite as connector_module
from synthgen.connectors.driver import DEFAULT_MAX_COLUMN, max_columns, sqlite_driver
from synthgen.connectors.sqlite import SqliteConnector
from synthgen.domain.errors import ConnectorError

WIDTH = 3_000
ROWS = 50


def test_stdlib_limit_is_detected() -> None:
    assert max_columns(sqlite3) == DEFAULT_MAX_COLUMN


@pytest.fixture(scope="module")
def wide_driver() -> ModuleType:
    driver = sqlite_driver()
    if max_columns(driver) < WIDTH + 1:
        pytest.skip("wide SQLite build not installed (make sqlite-wide)")
    return driver


@pytest.fixture(scope="module")
def wide_db(wide_driver: ModuleType, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("wide") / "wide.db"
    conn = wide_driver.connect(str(path))
    columns = ", ".join(f"m{i} REAL" for i in range(WIDTH))
    conn.execute(f"CREATE TABLE wide (id INTEGER PRIMARY KEY, {columns})")
    conn.executemany(
        f"INSERT INTO wide VALUES ({', '.join('?' * (WIDTH + 1))})",
        [(r, *(float(r + i) for i in range(WIDTH))) for r in range(1, ROWS + 1)],
    )
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def wide(wide_db: Path) -> Iterator[SqliteConnector]:
    opened = SqliteConnector(wide_db)
    yield opened
    opened.close()


def test_three_thousand_columns(wide: SqliteConnector) -> None:
    columns = wide.get_columns("wide")
    assert len(columns) == WIDTH + 1
    assert wide.row_count("wide") == ROWS
    sample = wide.sample("wide", [c.name for c in columns], 10, seed=1)
    assert sample.num_columns == WIDTH + 1
    assert sample.num_rows == 10


def test_narrow_driver_explains_the_limit(wide_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(connector_module, "sqlite_driver", lambda: sqlite3)
    narrow = SqliteConnector(wide_db)
    with pytest.raises(ConnectorError, match="make sqlite-wide"):
        narrow.get_columns("wide")
    narrow.close()
