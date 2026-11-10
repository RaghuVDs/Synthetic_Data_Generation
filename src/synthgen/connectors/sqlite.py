"""Read-only SQLite connector (SPEC 8.2).

Identifiers are checked against the database schema, then quoted. Values are always bound as
parameters. The file is opened with ``mode=ro`` so no statement can change it.
"""

from __future__ import annotations

import hashlib
import math
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pyarrow as pa

from synthgen.domain.errors import ConnectorError, MetadataError, UnknownIdentifierError
from synthgen.domain.query import (
    AggExpr,
    AggFunc,
    ColumnInfo,
    Condition,
    Filter,
    ForeignKeyInfo,
    Operator,
)

HASH_MODULUS = 2_147_483_647
"""Prime modulus (2**31 - 1) of the rowid hash; products stay inside 64-bit integers."""
HASH_MULTIPLIER = 1_327_217_885
"""About 0.618 x modulus: a golden-ratio step spreads selected rowids evenly over the table."""
PAGE_CACHE_KIB = 262_144
"""SQLite page cache per connection (256 MiB), enough to hold a seed-size table in memory."""
SAMPLE_OVERDRAW = 1.2
"""Draw this much more than requested through the hash filter, then trim to exactly n."""

_COMPARISONS = {
    Operator.EQ: "=",
    Operator.NE: "<>",
    Operator.LT: "<",
    Operator.LE: "<=",
    Operator.GT: ">",
    Operator.GE: ">=",
}

SqlParam = int | float | str


def quote(identifier: str) -> str:
    """Quote an identifier that has already been validated against the schema."""
    return '"' + identifier.replace('"', '""') + '"'


class SqliteConnector:
    """Reads one SQLite source file through one read-only connection, kept open until close().

    Keeping the connection keeps SQLite's page cache, so repeated scans of a table during
    profiling read the file once instead of once per query.
    """

    def __init__(self, path: Path) -> None:
        """Remember the file; fail early when it does not exist."""
        if not path.is_file():
            raise ConnectorError(f"source database {path} does not exist")
        self._uri = f"{path.resolve().as_uri()}?mode=ro"
        self._conn: sqlite3.Connection | None = None
        self._tables: list[str] | None = None
        self._columns: dict[str, list[ColumnInfo]] = {}

    def close(self) -> None:
        """Close the connection; a later call opens a new one."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if self._conn is None:
            try:
                self._conn = sqlite3.connect(self._uri, uri=True, check_same_thread=False)
                self._conn.execute(f"PRAGMA cache_size = -{PAGE_CACHE_KIB}")
            except sqlite3.Error as exc:
                raise ConnectorError(f"cannot open {self._uri}: {exc}") from exc
        try:
            yield self._conn
        except sqlite3.Error as exc:
            raise ConnectorError(f"query failed on {self._uri}: {exc}") from exc

    def _query(self, sql: str, params: Sequence[SqlParam] = ()) -> list[tuple[Any, ...]]:
        with self._connect() as conn:
            return conn.execute(sql, params).fetchall()

    def list_tables(self) -> list[str]:
        """User tables in the source, sorted by name."""
        if self._tables is None:
            rows = self._query(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                " ORDER BY name"
            )
            self._tables = [str(r[0]) for r in rows]
        return list(self._tables)

    def _check_table(self, table: str) -> None:
        if table not in self.list_tables():
            raise UnknownIdentifierError(f"unknown table {table!r}")

    def _check_columns(self, table: str, columns: Sequence[str]) -> None:
        known = {c.name for c in self.get_columns(table)}
        unknown = [c for c in columns if c not in known]
        if unknown:
            raise UnknownIdentifierError(f"unknown columns in {table!r}: {unknown}")

    def get_columns(self, table: str) -> list[ColumnInfo]:
        """Declared columns of a table, in database order."""
        if table not in self._columns:
            self._check_table(table)
            rows = self._query(f"PRAGMA table_info({quote(table)})")
            self._columns[table] = [
                ColumnInfo(
                    name=str(name),
                    declared_type=str(declared).upper(),
                    nullable=not notnull and not pk,
                    primary_key_position=int(pk),
                )
                for _, name, declared, notnull, _default, pk in rows
            ]
        return self._columns[table]

    def foreign_keys(self, table: str) -> list[ForeignKeyInfo]:
        """Foreign keys the database declares on a table; composite keys are rejected."""
        self._check_table(table)
        rows = self._query(f"PRAGMA foreign_key_list({quote(table)})")
        if any(int(seq) > 0 for _, seq, *_rest in rows):
            raise MetadataError(f"composite foreign keys on {table!r} are not supported")
        keys: list[ForeignKeyInfo] = []
        for _, _, parent, column, parent_column, *_ in rows:
            target = parent_column or self._single_primary_key(str(parent))
            keys.append(
                ForeignKeyInfo(column=str(column), parent_table=str(parent), parent_column=target)
            )
        return keys

    def _single_primary_key(self, table: str) -> str:
        pk = [c.name for c in self.get_columns(table) if c.primary_key_position]
        if len(pk) != 1:
            raise MetadataError(f"cannot resolve implicit foreign key target in {table!r}")
        return pk[0]

    def row_count(self, table: str) -> int:
        """Exact number of rows in a table."""
        self._check_table(table)
        return int(self._query(f"SELECT COUNT(*) FROM {quote(table)}")[0][0])

    def aggregate(
        self,
        table: str,
        exprs: list[AggExpr],
        group_by: list[str] | None = None,
        where: Filter | None = None,
    ) -> list[dict[str, Any]]:  # Any: SQLite returns int, float, str, bytes or None per cell
        """Run whitelisted aggregates; one dict per group keyed by alias and group column."""
        if not exprs:
            raise ConnectorError("aggregate needs at least one expression")
        groups = group_by or []
        self._check_table(table)
        self._check_columns(table, [e.column for e in exprs if e.column] + groups)
        select = [quote(g) for g in groups] + [f"{_agg_sql(e)} AS {quote(e.alias)}" for e in exprs]
        sql = f"SELECT {', '.join(select)} FROM {quote(table)}"
        where_sql, params = self._where(table, where)
        sql += where_sql
        if groups:
            group_sql = ", ".join(quote(g) for g in groups)
            sql += f" GROUP BY {group_sql} ORDER BY {group_sql}"
        names = groups + [e.alias for e in exprs]
        return [dict(zip(names, row, strict=True)) for row in self._query(sql, params)]

    def _where(self, table: str, where: Filter | None) -> tuple[str, list[SqlParam]]:
        if where is None:
            return "", []
        self._check_columns(table, [c.column for c in where.conditions])
        clauses: list[str] = []
        params: list[SqlParam] = []
        for condition in where.conditions:
            clause, values = _condition_sql(condition)
            clauses.append(clause)
            params.extend(values)
        return " WHERE " + " AND ".join(clauses), params

    def sample(self, table: str, columns: list[str], n: int, seed: int) -> pa.Table:
        """Deterministic sample of at most n rows; the whole table when it is small enough."""
        if n < 1 or seed < 0 or not columns:
            raise ConnectorError("sample needs n >= 1, seed >= 0 and at least one column")
        self._check_columns(table, columns)
        select = f"SELECT {', '.join(quote(c) for c in columns)} FROM {quote(table)}"
        total = self.row_count(table)
        if total <= n:
            return _to_arrow(columns, self._query(f"{select} ORDER BY rowid"))
        key = f"(((rowid * {HASH_MULTIPLIER}) % {HASH_MODULUS}) + ?) % {HASH_MODULUS}"
        offset = _seed_offset(seed)
        threshold = math.ceil(min(1.0, SAMPLE_OVERDRAW * n / total) * HASH_MODULUS)
        rows = self._query(
            f"{select} WHERE {key} < ? ORDER BY {key} LIMIT ?", (offset, threshold, offset, n)
        )
        if len(rows) < n:
            rows = self._query(f"{select} ORDER BY {key} LIMIT ?", (offset, n))
        return _to_arrow(columns, rows)


def _seed_offset(seed: int) -> int:
    """Spread a seed over the whole hash range so nearby seeds select unrelated rows."""
    digest = hashlib.sha256(str(seed).encode()).digest()
    return int.from_bytes(digest[:8], "big") % HASH_MODULUS


def _agg_sql(expr: AggExpr) -> str:
    if expr.column is None:
        return "COUNT(*)"
    col = quote(expr.column)
    templates = {
        AggFunc.COUNT: f"COUNT({col})",
        AggFunc.COUNT_DISTINCT: f"COUNT(DISTINCT {col})",
        AggFunc.NULL_COUNT: f"COALESCE(SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END), 0)",
        AggFunc.SUM: f"SUM({col})",
        AggFunc.SUM_SQ: f"SUM(CAST({col} AS REAL) * CAST({col} AS REAL))",
        AggFunc.MIN: f"MIN({col})",
        AggFunc.MAX: f"MAX({col})",
        AggFunc.AVG: f"AVG({col})",
    }
    return templates[expr.func]


def _condition_sql(condition: Condition) -> tuple[str, list[SqlParam]]:
    col = quote(condition.column)
    value = condition.value
    if condition.op is Operator.IS_NULL:
        return f"{col} IS NULL", []
    if condition.op is Operator.NOT_NULL:
        return f"{col} IS NOT NULL", []
    if isinstance(value, tuple):
        return f"{col} IN ({', '.join('?' for _ in value)})", list(value)
    if value is None:
        raise ConnectorError(f"condition on {condition.column!r} has no value")
    return f"{col} {_COMPARISONS[condition.op]} ?", [value]


def _to_arrow(columns: list[str], rows: list[tuple[Any, ...]]) -> pa.Table:
    """Build a table; a column with mixed SQLite storage classes falls back to text."""
    arrays: list[pa.Array[Any]] = []
    for index, _ in enumerate(columns):
        values = [row[index] for row in rows]
        try:
            arrays.append(pa.array(values))
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            arrays.append(
                pa.array([None if v is None else str(v) for v in values], type=pa.string())
            )
    return pa.table(arrays, names=columns)
