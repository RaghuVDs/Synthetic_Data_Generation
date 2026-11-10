from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from synthgen.config import ProfilingSettings, load_settings

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def profiling_settings() -> ProfilingSettings:
    return load_settings(ROOT / "config" / "settings.yaml", {}).profiling


@pytest.fixture(scope="session")
def shop_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Customers (with a sensitive email) and orders whose counts per customer are known."""
    path = tmp_path_factory.mktemp("profiling") / "shop.db"
    rng = np.random.default_rng(0)
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY, email TEXT NOT NULL, tier TEXT, joined TEXT, score REAL
        );
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL REFERENCES customers(id),
            amount REAL NOT NULL,
            placed TEXT NOT NULL,
            paid INTEGER NOT NULL
        );
        """
    )
    conn.executemany(
        "INSERT INTO customers VALUES (?, ?, ?, ?, ?)",
        [
            (
                i,
                f"person{i}@mail.example",
                ["gold", "silver", "bronze"][i % 3] if i % 10 else None,
                f"2024-01-{i % 28 + 1:02d}",
                round(float(rng.normal(50, 10)), 2),
            )
            for i in range(1, 301)
        ],
    )
    orders = [(cid, k) for cid in range(1, 301) for k in range(cid % 4)]  # 0..3 per customer
    conn.executemany(
        "INSERT INTO orders VALUES (?, ?, ?, ?, ?)",
        [
            (
                n + 1,
                cid,
                round(float(rng.lognormal(3, 0.5)), 2),
                f"2024-03-04 {k + 9:02d}:30:00",
                k % 2,
            )
            for n, (cid, k) in enumerate(orders)
        ],
    )
    conn.commit()
    conn.close()
    return path
