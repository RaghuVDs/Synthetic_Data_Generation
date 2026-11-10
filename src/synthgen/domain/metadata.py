"""Source metadata: tables, columns, relationships and the ordered table set (SPEC 8.1)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from synthgen.domain.base import StrictModel


class Sensitivity(StrEnum):
    """Privacy class of a column; decides what the profile may store (SPEC 8.3)."""

    PUBLIC = "public"
    INTERNAL = "internal"
    SENSITIVE = "sensitive"


class RelationshipOrigin(StrEnum):
    """Where a relationship was declared."""

    METADATA = "metadata"
    PRAGMA = "pragma"


class ColumnMeta(StrictModel):
    """One column: database facts merged with hand-written description and sensitivity tag."""

    name: str
    declared_type: str
    nullable: bool
    primary_key: bool
    description: str = ""
    sensitivity: Sensitivity | None = None


class TableMeta(StrictModel):
    """One table with its columns in database order."""

    name: str
    description: str = ""
    columns: tuple[ColumnMeta, ...] = Field(min_length=1)

    @property
    def primary_key(self) -> tuple[str, ...]:
        """Names of the primary-key columns."""
        return tuple(c.name for c in self.columns if c.primary_key)

    def column(self, name: str) -> ColumnMeta | None:
        """The named column, or None when the table has no such column."""
        return next((c for c in self.columns if c.name == name), None)


class Relationship(StrictModel):
    """A foreign key: child column references parent column."""

    child_table: str
    child_column: str
    parent_table: str
    parent_column: str
    nullable: bool
    drives_cardinality: bool
    origin: RelationshipOrigin

    @property
    def is_self_reference(self) -> bool:
        """True when a table references itself."""
        return self.child_table == self.parent_table

    def label(self) -> str:
        """Readable form, e.g. `accounts.customer_id -> customers.customer_id`."""
        return f"{self.child_table}.{self.child_column} -> {self.parent_table}.{self.parent_column}"


class TableSet(StrictModel):
    """Tables a job generates: selected ones, added parents and the generation order."""

    selected: tuple[str, ...]
    support: tuple[str, ...]
    order: tuple[str, ...]
    relationships: tuple[Relationship, ...]
    deferred: tuple[Relationship, ...] = ()
    """Keys filled in a second pass: self-references and broken mutual references."""


class TableSummary(StrictModel):
    """One line of a source listing."""

    name: str
    description: str
    row_count: int
    column_count: int


class SourceOverview(StrictModel):
    """What `tables --source S` shows: tables with sizes, and every relationship."""

    source: str
    description: str
    tables: tuple[TableSummary, ...]
    relationships: tuple[Relationship, ...]
