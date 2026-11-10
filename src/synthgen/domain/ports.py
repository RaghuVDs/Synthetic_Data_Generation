"""Protocols that inner packages depend on; concrete classes are wired in app.py (PLAN Q2)."""

from __future__ import annotations

from typing import Any, Protocol

import pyarrow as pa

from synthgen.domain.metadata import Relationship, TableMeta
from synthgen.domain.query import AggExpr, ColumnInfo, Filter, ForeignKeyInfo


class Connector(Protocol):
    """Read-only access to one source database (SPEC 8.2)."""

    def list_tables(self) -> list[str]:
        """User tables in the source, sorted by name."""
        ...

    def get_columns(self, table: str) -> list[ColumnInfo]:
        """Declared columns of a table, in database order."""
        ...

    def foreign_keys(self, table: str) -> list[ForeignKeyInfo]:
        """Foreign keys the database declares on a table."""
        ...

    def row_count(self, table: str) -> int:
        """Exact number of rows in a table."""
        ...

    def aggregate(
        self,
        table: str,
        exprs: list[AggExpr],
        group_by: list[str] | None = None,
        where: Filter | None = None,
    ) -> list[dict[str, Any]]:  # Any: SQLite returns int, float, str, bytes or None per cell
        """Run whitelisted aggregates; one dict per group keyed by alias and group column."""
        ...

    def sample(self, table: str, columns: list[str], n: int, seed: int) -> pa.Table:
        """Deterministic sample of at most n rows; the whole table when it is small enough."""
        ...


class MetadataProvider(Protocol):
    """Tables and relationships of one source, merged from metadata YAML and the database."""

    def tables(self) -> list[TableMeta]:
        """Every table in the source, sorted by name."""
        ...

    def table(self, name: str) -> TableMeta:
        """One table; raises UnknownIdentifierError when absent."""
        ...

    def relationships(self) -> list[Relationship]:
        """Every relationship, declared in metadata or found through the database."""
        ...
