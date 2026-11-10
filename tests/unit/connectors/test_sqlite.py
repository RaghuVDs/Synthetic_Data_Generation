from __future__ import annotations

import itertools
import sqlite3
from pathlib import Path

import pyarrow as pa
import pytest

from synthgen.connectors.sqlite import SqliteConnector
from synthgen.domain.errors import ConnectorError, MetadataError, UnknownIdentifierError
from synthgen.domain.query import AggExpr, AggFunc, Condition, Filter, Operator

ROWS = 10_000


@pytest.fixture(scope="module")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("src") / "source.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE parents (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
        CREATE TABLE items (
            id INTEGER PRIMARY KEY,
            parent_id INTEGER NOT NULL REFERENCES parents(id),
            owner_id INTEGER REFERENCES parents,
            amount REAL,
            kind TEXT,
            mixed
        );
        CREATE TABLE pairs (a INTEGER, b INTEGER, PRIMARY KEY (a, b));
        CREATE TABLE links (
            a INTEGER, b INTEGER, FOREIGN KEY (a, b) REFERENCES pairs(a, b)
        );
        """
    )
    conn.executemany("INSERT INTO parents VALUES (?, ?)", [(i, f"p{i}") for i in range(1, 11)])
    conn.executemany(
        "INSERT INTO items VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                i,
                i % 10 + 1,
                None,
                None if i % 100 == 0 else float(i),
                "a" if i % 2 else "b",
                i if i % 3 else f"t{i}",
            )
            for i in range(1, ROWS + 1)
        ],
    )
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def connector(db_path: Path) -> SqliteConnector:
    return SqliteConnector(db_path)


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConnectorError):
        SqliteConnector(tmp_path / "absent.db")


def test_connection_is_read_only(connector: SqliteConnector) -> None:
    with pytest.raises(ConnectorError, match="readonly"), connector._connect() as conn:
        conn.execute("DELETE FROM items")


def test_tables_and_columns(connector: SqliteConnector) -> None:
    assert connector.list_tables() == ["items", "links", "pairs", "parents"]
    columns = {c.name: c for c in connector.get_columns("items")}
    assert list(columns) == ["id", "parent_id", "owner_id", "amount", "kind", "mixed"]
    assert columns["id"].primary_key_position == 1
    assert not columns["id"].nullable
    assert not columns["parent_id"].nullable
    assert columns["amount"].nullable
    assert columns["amount"].declared_type == "REAL"
    assert columns["mixed"].declared_type == ""


def test_foreign_keys_resolve_implicit_target(connector: SqliteConnector) -> None:
    keys = {k.column: k for k in connector.foreign_keys("items")}
    assert keys["parent_id"].parent_column == "id"
    assert keys["owner_id"].parent_table == "parents"
    assert keys["owner_id"].parent_column == "id"


def test_composite_foreign_key_is_rejected(connector: SqliteConnector) -> None:
    with pytest.raises(MetadataError, match="composite"):
        connector.foreign_keys("links")


@pytest.mark.parametrize("table", ["missing", 'items"; DROP TABLE items; --'])
def test_unknown_table_is_rejected(connector: SqliteConnector, table: str) -> None:
    with pytest.raises(UnknownIdentifierError):
        connector.row_count(table)


def test_row_count(connector: SqliteConnector) -> None:
    assert connector.row_count("items") == ROWS


def test_aggregates_match_python(connector: SqliteConnector) -> None:
    exprs = [
        AggExpr(func=AggFunc.COUNT, column=None, alias="n"),
        AggExpr(func=AggFunc.NULL_COUNT, column="amount", alias="nulls"),
        AggExpr(func=AggFunc.SUM, column="amount", alias="s"),
        AggExpr(func=AggFunc.SUM_SQ, column="amount", alias="ss"),
        AggExpr(func=AggFunc.COUNT_DISTINCT, column="kind", alias="kinds"),
        AggExpr(func=AggFunc.MIN, column="amount", alias="lo"),
        AggExpr(func=AggFunc.MAX, column="amount", alias="hi"),
    ]
    [row] = connector.aggregate("items", exprs)
    amounts = [float(i) for i in range(1, ROWS + 1) if i % 100]
    assert row["n"] == ROWS
    assert row["nulls"] == ROWS // 100
    assert row["s"] == pytest.approx(sum(amounts))
    assert row["ss"] == pytest.approx(sum(a * a for a in amounts))
    assert row["kinds"] == 2
    assert (row["lo"], row["hi"]) == (1.0, float(ROWS - 1))


def test_group_by_and_filter(connector: SqliteConnector) -> None:
    where = Filter(
        conditions=(
            Condition(column="parent_id", op=Operator.IN, value=(1, 2)),
            Condition(column="amount", op=Operator.NOT_NULL),
        )
    )
    rows = connector.aggregate(
        "items",
        [AggExpr(func=AggFunc.COUNT, column=None, alias="n")],
        group_by=["parent_id"],
        where=where,
    )
    assert [r["parent_id"] for r in rows] == [1, 2]
    assert [r["n"] for r in rows] == [900, 1000]


def test_filter_values_are_parameters(connector: SqliteConnector) -> None:
    where = Filter(conditions=(Condition(column="kind", op=Operator.EQ, value="a' OR '1'='1"),))
    [row] = connector.aggregate(
        "items", [AggExpr(func=AggFunc.COUNT, column=None, alias="n")], where=where
    )
    assert row["n"] == 0


def test_aggregate_rejects_unknown_columns(connector: SqliteConnector) -> None:
    with pytest.raises(UnknownIdentifierError):
        connector.aggregate("items", [AggExpr(func=AggFunc.SUM, column="nope", alias="s")])
    with pytest.raises(UnknownIdentifierError):
        connector.aggregate(
            "items", [AggExpr(func=AggFunc.COUNT, column=None, alias="n")], group_by=["nope"]
        )
    with pytest.raises(ConnectorError):
        connector.aggregate("items", [])


def test_sample_is_deterministic_and_exact(connector: SqliteConnector) -> None:
    first = connector.sample("items", ["id", "amount"], 1_000, seed=7)
    again = connector.sample("items", ["id", "amount"], 1_000, seed=7)
    other = connector.sample("items", ["id", "amount"], 1_000, seed=8)
    assert first.num_rows == 1_000
    assert first.equals(again)
    overlap = set(first.column("id").to_pylist()) & set(other.column("id").to_pylist())
    assert len(overlap) < 500
    assert first.column_names == ["id", "amount"]


def test_sample_is_spread_over_the_table(connector: SqliteConnector) -> None:
    raw = connector.sample("items", ["id"], 1_000, seed=3).column("id").to_pylist()
    ids = sorted(int(i) for i in raw if i is not None)
    gaps = [b - a for a, b in itertools.pairwise(ids)]
    assert max(gaps) < 60
    assert 4_000 < sum(ids) / len(ids) < 6_000


def test_small_table_is_read_whole(connector: SqliteConnector) -> None:
    table = connector.sample("parents", ["id", "name"], 50, seed=1)
    assert table.column("id").to_pylist() == list(range(1, 11))


def test_mixed_storage_classes_fall_back_to_text(connector: SqliteConnector) -> None:
    table = connector.sample("items", ["mixed"], 100, seed=1)
    assert table.schema.field("mixed").type == pa.string()


@pytest.mark.parametrize(("columns", "n", "seed"), [(["id"], 0, 1), (["id"], 5, -1), ([], 5, 1)])
def test_sample_rejects_bad_arguments(
    connector: SqliteConnector, columns: list[str], n: int, seed: int
) -> None:
    with pytest.raises(ConnectorError):
        connector.sample("items", columns, n, seed)


def test_sample_rejects_unknown_column(connector: SqliteConnector) -> None:
    with pytest.raises(UnknownIdentifierError):
        connector.sample("items", ["id", "nope"], 5, seed=1)
