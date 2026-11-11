from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from synthgen.config import ProfilingSettings
from synthgen.connectors.sqlite import SqliteConnector
from synthgen.domain.metadata import Sensitivity
from synthgen.metadata.provider import ColumnDoc, SourceDoc, TableDoc, YamlMetadataProvider
from synthgen.profiling.models import ChildrenPerParent
from synthgen.profiling.profiler import profile_table
from synthgen.profiling.relationships import gini, relationship_shape

CUSTOMERS = 400
ORPHANS = 25


@pytest.fixture(scope="module")
def shop(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rng = np.random.default_rng(5)
    path = tmp_path_factory.mktemp("rel") / "shop.db"
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE customers (id INTEGER PRIMARY KEY, segment TEXT, region TEXT,"
        " joined TEXT, email TEXT);"
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES"
        " customers(id), ordered_at TEXT, amount REAL);"
    )
    start = datetime(2023, 1, 1)
    order_id = 0
    for i in range(CUSTOMERS):
        segment = "vip" if i % 4 == 0 else "retail"
        region = "north" if i < CUSTOMERS - 5 else "islands"  # five parents: below the fold
        joined = start + timedelta(days=int(rng.integers(0, 365)))
        con.execute(
            "INSERT INTO customers VALUES (?, ?, ?, ?, ?)",
            (i, segment, region, joined.isoformat(sep=" "), f"c{i}@mail.test"),
        )
        for _ in range(int(rng.poisson(8 if segment == "vip" else 2))):
            ordered = joined + timedelta(
                days=int(rng.integers(0, 90)), hours=int(rng.integers(0, 24))
            )
            con.execute(
                "INSERT INTO orders VALUES (?, ?, ?, ?)",
                (order_id, i, ordered.isoformat(sep=" "), float(rng.uniform(5, 50))),
            )
            order_id += 1
    for k in range(ORPHANS):
        con.execute(
            "INSERT INTO orders VALUES (?, ?, ?, ?)",
            (order_id + k, 10_000 + k, "2024-06-01 10:00:00", 9.0),
        )
    con.commit()
    con.close()
    return path


def _shape(path: Path, settings: ProfilingSettings) -> ChildrenPerParent:
    doc = SourceDoc(
        source="shop",
        tables={
            "customers": TableDoc(columns={"region": ColumnDoc(sensitivity=Sensitivity.INTERNAL)})
        },
    )
    connector = SqliteConnector(path)
    try:
        provider = YamlMetadataProvider(doc, connector)
        profile = profile_table(
            "shop",
            connector,
            provider.table("orders"),
            provider.relationships(),
            parents={"customers": provider.table("customers")},
            settings=settings,
            seed=1,
            now=datetime(2026, 1, 1, tzinfo=UTC),
        )
    finally:
        connector.close()
    (shape,) = profile.relationships
    return shape


def test_coverage_orphans_and_fits(shop: Path, profiling_settings: ProfilingSettings) -> None:
    rel = _shape(shop, profiling_settings)
    shape = rel.shape.value
    assert shape.orphan_rate == pytest.approx(ORPHANS / shape.child_rows)
    assert rel.orphan_rows.value == ORPHANS
    assert 0.8 < shape.parent_coverage < 1.0
    assert shape.children.mean == pytest.approx(0.25 * 8 + 0.75 * 2, abs=0.4)
    fits = shape.count_fits
    assert fits is not None and fits.nbinom_aic is not None
    assert fits.nbinom_aic < fits.poisson_aic  # a mix of two Poissons is over-dispersed
    assert shape.lognorm is not None and shape.zipf_exponent is not None
    assert 0.0 < shape.gini < 1.0 and shape.top_share > 0.01


def test_child_dates_follow_parent_dates(shop: Path, profiling_settings: ProfilingSettings) -> None:
    orders = _shape(shop, profiling_settings).temporal_orders.value
    (order,) = orders
    assert (order.parent_column, order.child_column) == ("joined", "ordered_at")
    assert order.after_share == 1.0
    assert 0 <= order.lag_seconds.p50 <= 90 * 86_400


def test_parent_categories(shop: Path, profiling_settings: ProfilingSettings) -> None:
    categories = {
        c.parent_column: c for c in _shape(shop, profiling_settings).by_parent_category.value
    }
    assert "email" not in categories  # sensitive parent columns never take part
    segment = {level.level: level for level in categories["segment"].levels}
    assert segment["vip"].parent_share == pytest.approx(0.25)
    assert segment["vip"].mean_children > 3 * segment["retail"].mean_children
    region = [level.level for level in categories["region"].levels]
    assert region == ["north", "other"]  # internal levels under 10 parents fold away


def test_relationship_shape_edges() -> None:
    empty = relationship_shape(np.array([], dtype=np.int64), 0, 0)
    assert empty.orphan_rate == 0.0 and empty.top_share == 0.0 and empty.count_fits is not None
    assert empty.lognorm is None
    assert gini(np.array([3, 3, 3])) == pytest.approx(0.0)
    assert gini(np.array([0, 0, 0, 12])) == pytest.approx(0.75)
    assert gini(np.zeros(3, dtype=np.int64)) == 0.0
