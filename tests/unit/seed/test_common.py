from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from synthgen.domain.errors import SeedError
from synthgen.seed import common as c


def test_luhn_check_digit_known_value() -> None:
    body = np.array([[7, 9, 9, 2, 7, 3, 9, 8, 7, 1]], dtype=np.int64)
    assert c.digits_to_text(c.luhn_complete(body)) == ["79927398713"]


def test_zipf_indices_are_skewed_and_in_range() -> None:
    rng = np.random.default_rng(1)
    draws = c.zipf_indices(rng, 100, 50_000, exponent=1.0)
    assert np.min(draws) >= 0
    assert np.max(draws) < 100
    counts = np.sort(np.bincount(draws, minlength=100))[::-1]
    assert counts[0] > 10 * counts[50]


def test_iso_conversions() -> None:
    seconds = np.array([c.epoch("2024-02-29 13:05:09")])
    assert c.iso_dates(seconds) == ["2024-02-29"]
    assert c.iso_timestamps(seconds) == ["2024-02-29 13:05:09"]


def test_weighted_times_stay_in_bounds_and_follow_weights() -> None:
    rng = np.random.default_rng(2)
    n = 20_000
    low = np.full(n, c.epoch("2024-01-01"))
    high = np.full(n, c.epoch("2024-12-31"))
    weekday_weights = [1, 1, 1, 1, 1, 1, 0.1]
    hour_weights = [0.0] * 23 + [1.0]
    times = c.weighted_times(rng, low, high, weekday_weights, hour_weights)
    assert np.min(times) >= low[0]
    assert np.max(times) < high[0] + c.SECONDS_PER_DAY
    hours = (times % c.SECONDS_PER_DAY) // 3600
    assert set(hours.tolist()) == {23}
    weekday = ((times // c.SECONDS_PER_DAY + 3) % 7).astype(int)
    sunday_share = np.mean(weekday == 6)
    assert sunday_share < 0.03


def test_with_nulls_rate() -> None:
    rng = np.random.default_rng(3)
    values = c.with_nulls(rng, list(range(10_000)), 0.2)
    assert 0.17 < sum(v is None for v in values) / 10_000 < 0.23


def test_name_pool_is_distinct_letters() -> None:
    names = iter(["Ann", "O'Neil", "Ann", "Bo-Bo", "Cy"])
    pool = c.name_pool(lambda: next(names), 4)
    assert pool == ["Ann", "ONeil", "BoBo", "Cy"]


def test_name_pool_gives_up_when_source_repeats() -> None:
    with pytest.raises(SeedError):
        c.name_pool(lambda: "Same", 2)


def test_write_database_is_atomic(tmp_path: Path) -> None:
    target = tmp_path / "x.db"

    def broken(conn: sqlite3.Connection) -> dict[str, int]:
        conn.execute("INSERT INTO missing_table VALUES (1)")
        return {}

    with pytest.raises(SeedError):
        c.write_database(target, "CREATE TABLE t (a INTEGER);", broken)
    assert list(tmp_path.iterdir()) == []


def test_write_database_replaces_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "x.db"
    target.write_text("old", encoding="utf-8")

    def fill(conn: sqlite3.Connection) -> dict[str, int]:
        return {"t": c.insert(conn, "t", {"a": np.array([1, 2, 3])})}

    assert c.write_database(target, "CREATE TABLE t (a INTEGER);", fill) == {"t": 3}
    with closing(sqlite3.connect(target)) as conn:
        assert conn.execute("SELECT SUM(a) FROM t").fetchone() == (6,)
