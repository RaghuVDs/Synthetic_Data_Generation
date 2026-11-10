"""Helpers shared by the seed builders: weighted draws, dates, check digits and SQLite writes."""

from __future__ import annotations

import os
import re
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from synthgen.domain.errors import SeedError

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]
StrArray = NDArray[np.str_]
BoolArray = NDArray[np.bool_]
Column = Sequence[Any] | NDArray[Any]  # Any: a column holds ints, floats, text or None

SECONDS_PER_DAY = 86_400
MAX_REDRAWS = 64
"""Rejection-sampling rounds for weekday patterns; rows still rejected keep their last draw."""


def scaled(base: int, scale: float) -> int:
    """Row count for a table at the given scale, never below one."""
    return max(1, round(base * scale))


def choice(
    rng: np.random.Generator, values: Sequence[str], weights: Sequence[float], size: int
) -> StrArray:
    """Draw labels with the given (unnormalised) weights."""
    p = np.asarray(weights, dtype=np.float64)
    return np.asarray(values)[rng.choice(len(values), size=size, p=p / p.sum())]


def zipf_indices(rng: np.random.Generator, n_items: int, size: int, exponent: float) -> IntArray:
    """Indices into n_items with Zipf-skewed popularity; popular items are randomly placed."""
    weights = 1.0 / np.arange(1, n_items + 1, dtype=np.float64) ** exponent
    popularity = rng.permutation(n_items)
    ranks = rng.choice(n_items, size=size, p=weights / weights.sum())
    return popularity[ranks].astype(np.int64)


def uniform_between(rng: np.random.Generator, low: FloatArray, high: FloatArray) -> FloatArray:
    """One uniform draw per row between per-row bounds (epoch seconds)."""
    return low + rng.random(low.shape[0]) * np.maximum(high - low, 0.0)


def epoch(text: str) -> float:
    """Seconds since 1970 for an ISO date or timestamp."""
    return float(np.datetime64(text, "s").astype(np.int64))


def iso_dates(seconds: FloatArray) -> list[str]:
    """ISO dates (YYYY-MM-DD) from epoch seconds."""
    return [str(v) for v in np.datetime_as_string(seconds.astype("datetime64[s]"), unit="D")]


def iso_timestamps(seconds: FloatArray) -> list[str]:
    """ISO timestamps (YYYY-MM-DD HH:MM:SS) from epoch seconds."""
    text = np.datetime_as_string(seconds.astype("datetime64[s]"), unit="s")
    return [str(v) for v in np.char.replace(text, "T", " ")]


def weighted_times(
    rng: np.random.Generator,
    day_low: FloatArray,
    day_high: FloatArray,
    weekday_weights: Sequence[float],
    hour_weights: Sequence[float],
) -> FloatArray:
    """Timestamps between per-row day bounds with weekday and hour-of-day patterns."""
    size = day_low.shape[0]
    accept = np.asarray(weekday_weights, dtype=np.float64) / max(weekday_weights)
    days = np.floor(uniform_between(rng, day_low, day_high) / SECONDS_PER_DAY)
    pending = np.ones(size, dtype=bool)
    for _ in range(MAX_REDRAWS):  # rejection sampling on the weekday weights
        weekday = ((days + 3) % 7).astype(np.int64)  # 1970-01-01 was a Thursday; Monday = 0
        pending &= rng.random(size) >= accept[weekday]
        if not pending.any():
            break
        redraw = uniform_between(rng, day_low[pending], day_high[pending])
        days[pending] = np.floor(redraw / SECONDS_PER_DAY)
    hours = choice(rng, [str(h) for h in range(24)], hour_weights, size).astype(np.int64)
    within = hours * 3600 + rng.integers(0, 3600, size)
    return np.asarray(days * SECONDS_PER_DAY + within, dtype=np.float64)


def luhn_complete(body: NDArray[np.int64]) -> NDArray[np.int64]:
    """Append a Luhn check digit to each row of digits (rows are numbers, digits are columns)."""
    digits = body[:, ::-1].copy()
    doubled = digits[:, ::2] * 2
    digits[:, ::2] = doubled // 10 + doubled % 10  # digit sum of the doubled digit
    check = (10 - digits.sum(axis=1) % 10) % 10
    return np.concatenate([body, check[:, None]], axis=1)


def digits_to_text(digits: NDArray[np.int64]) -> list[str]:
    """Join digit rows into strings."""
    return ["".join(map(str, row)) for row in digits.tolist()]


def name_pool(make: Callable[[], str], size: int) -> list[str]:
    """Distinct ASCII-letter values from a generator such as a seeded Faker provider."""
    pool: dict[str, None] = {}
    attempts = 0
    while len(pool) < size:
        attempts += 1
        if attempts > size * 50:
            raise SeedError(f"could not draw {size} distinct names")
        value = re.sub(r"[^A-Za-z]", "", make())
        if value:
            pool[value] = None
    return list(pool)


def with_nulls(rng: np.random.Generator, values: Column, rate: float) -> list[Any]:
    """Replace a random share of values with None."""
    mask = rng.random(len(values)) < rate
    return [None if m else v for v, m in zip(list(values), mask.tolist(), strict=True)]


def insert(conn: sqlite3.Connection, table: str, columns: Mapping[str, Column]) -> int:
    """Insert column-oriented data; names are fixed by the seed schema, values are parameters."""
    names = list(columns)
    data = [c.tolist() if isinstance(c, np.ndarray) else list(c) for c in columns.values()]
    placeholders = ", ".join("?" for _ in names)
    sql = f"INSERT INTO {table} ({', '.join(names)}) VALUES ({placeholders})"  # noqa: S608
    rows = list(zip(*data, strict=True))
    conn.executemany(sql, rows)
    return len(rows)


def write_database(
    path: Path, schema: str, fill: Callable[[sqlite3.Connection], dict[str, int]]
) -> dict[str, int]:
    """Build a database in a temporary file, then move it into place in one step."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".building")
    partial.unlink(missing_ok=True)
    try:
        conn = sqlite3.connect(partial)
        try:
            conn.executescript(schema)
            counts = fill(conn)
            conn.commit()
        finally:
            conn.close()
        os.replace(partial, path)
    except (sqlite3.Error, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise SeedError(f"could not build {path}: {exc}") from exc
    return counts
