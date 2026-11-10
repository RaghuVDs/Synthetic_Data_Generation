from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pytest

from synthgen.seed.orders import build_orders

SEED = 5
SCALE = 0.05


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, int]]:
    path = tmp_path_factory.mktemp("seed") / "orders.db"
    return path, build_orders(path, SEED, SCALE)


@pytest.fixture(scope="module")
def conn(built: tuple[Path, dict[str, int]]) -> Iterator[sqlite3.Connection]:
    opened = sqlite3.connect(built[0])
    yield opened
    opened.close()


def scalar(conn: sqlite3.Connection, sql: str) -> float:
    return float(conn.execute(sql).fetchone()[0])


def test_row_counts_follow_scale(built: tuple[Path, dict[str, int]]) -> None:
    counts = built[1]
    assert counts["customers"] == 750
    assert counts["products"] == 100
    assert counts["orders"] == 3_000
    assert 9_000 < counts["order_items"] < 11_000
    assert 2_400 < counts["shipments"] < 2_900


def test_every_foreign_key_resolves(conn: sqlite3.Connection) -> None:
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize(
    "violations",
    [
        "SELECT COUNT(*) FROM order_items"
        " WHERE ABS(line_total - ROUND(quantity * unit_price, 2)) > 0.001",
        "SELECT COUNT(*) FROM orders o JOIN (SELECT order_id, ROUND(SUM(line_total), 2) AS s"
        " FROM order_items GROUP BY order_id) i USING (order_id)"
        " WHERE ABS(o.total_amount - i.s) > 0.011",
        "SELECT COUNT(*) FROM orders o JOIN customers c USING (customer_id)"
        " WHERE DATE(o.order_ts) < c.signup_date",
        "SELECT COUNT(*) FROM shipments s JOIN orders o USING (order_id)"
        " WHERE s.shipped_date < DATE(o.order_ts)",
        "SELECT COUNT(*) FROM shipments WHERE delivered_date < shipped_date",
        "SELECT COUNT(*) FROM shipments s JOIN orders o USING (order_id)"
        " WHERE o.status IN ('placed', 'cancelled')",
    ],
)
def test_business_rules_hold(conn: sqlite3.Connection, violations: str) -> None:
    assert scalar(conn, violations) == 0


def test_late_label_matches_dates(conn: sqlite3.Connection) -> None:
    mismatches = scalar(
        conn,
        "SELECT COUNT(*) FROM shipments WHERE delivered_date IS NOT NULL"
        " AND is_late <> (delivered_date > promised_date)",
    )
    assert mismatches == 0
    assert 0.05 < scalar(conn, "SELECT AVG(is_late) FROM shipments") < 0.3


def test_in_transit_shipments_have_no_delivery(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT s.delivered_date, s.is_late FROM shipments s JOIN orders o USING (order_id)"
        " WHERE o.status = 'shipped'"
    ).fetchall()
    assert rows
    assert all(d is None and late is None for d, late in rows)


def test_sku_prefix_follows_category(conn: sqlite3.Connection) -> None:
    pairs = conn.execute("SELECT DISTINCT category, SUBSTR(sku, 1, 3) FROM products").fetchall()
    categories = [category for category, _ in pairs]
    assert len(categories) == len(set(categories))


def test_same_seed_same_data(built: tuple[Path, dict[str, int]], tmp_path: Path) -> None:
    again = tmp_path / "again.db"
    build_orders(again, SEED, SCALE)
    with closing(sqlite3.connect(built[0])) as first, closing(sqlite3.connect(again)) as second:
        assert list(first.iterdump()) == list(second.iterdump())
