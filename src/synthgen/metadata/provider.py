"""Metadata from `config/metadata/<source>.yaml`, merged with the database schema (SPEC 8.1).

The YAML adds descriptions, sensitivity tags and relationships. Any foreign key the database
declares but the YAML omits is added from `PRAGMA foreign_key_list`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import Field, ValidationError

from synthgen.domain.base import StrictModel
from synthgen.domain.errors import MetadataError, UnknownIdentifierError
from synthgen.domain.metadata import (
    ColumnMeta,
    Relationship,
    RelationshipOrigin,
    Sensitivity,
    TableMeta,
)
from synthgen.domain.ports import Connector

_QUALIFIED = r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$"


class ColumnDoc(StrictModel):
    """Hand-written notes on one column."""

    description: str = ""
    sensitivity: Sensitivity | None = None


class TableDoc(StrictModel):
    """Hand-written notes on one table."""

    description: str = ""
    columns: dict[str, ColumnDoc] = Field(default_factory=dict)


class RelationshipDoc(StrictModel):
    """A relationship written as `child_table.column` -> `parent_table.column`."""

    child: str = Field(pattern=_QUALIFIED)
    parent: str = Field(pattern=_QUALIFIED)
    drives_cardinality: bool = False


class SourceDoc(StrictModel):
    """The whole metadata file for one source."""

    source: str
    description: str = ""
    tables: dict[str, TableDoc] = Field(default_factory=dict)
    relationships: list[RelationshipDoc] = Field(default_factory=list)


def load_source_doc(path: Path) -> SourceDoc:
    """Read and validate one metadata YAML file."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return SourceDoc.model_validate(raw)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise MetadataError(f"invalid metadata file {path}: {exc}") from exc


class YamlMetadataProvider:
    """Tables and relationships of one source. Built once at construction, then read-only."""

    def __init__(self, doc: SourceDoc, connector: Connector) -> None:
        """Merge the metadata document with the live schema; fail on any disagreement."""
        self._tables = _merge_tables(doc, connector)
        self._relationships = _merge_relationships(doc, connector, self._tables)

    def tables(self) -> list[TableMeta]:
        """Every table in the source, sorted by name."""
        return [self._tables[name] for name in sorted(self._tables)]

    def table(self, name: str) -> TableMeta:
        """One table; raises UnknownIdentifierError when absent."""
        if name not in self._tables:
            raise UnknownIdentifierError(f"unknown table {name!r}")
        return self._tables[name]

    def relationships(self) -> list[Relationship]:
        """Every relationship, declared in metadata or found through the database."""
        return list(self._relationships)


def _merge_tables(doc: SourceDoc, connector: Connector) -> dict[str, TableMeta]:
    db_tables = connector.list_tables()
    unknown = sorted(set(doc.tables) - set(db_tables))
    if unknown:
        raise UnknownIdentifierError(f"metadata names tables missing from the source: {unknown}")
    merged: dict[str, TableMeta] = {}
    for name in db_tables:
        table_doc = doc.tables.get(name, TableDoc())
        infos = connector.get_columns(name)
        missing = sorted(set(table_doc.columns) - {i.name for i in infos})
        if missing:
            raise UnknownIdentifierError(f"metadata names columns missing from {name}: {missing}")
        columns = tuple(
            ColumnMeta(
                name=info.name,
                declared_type=info.declared_type,
                nullable=info.nullable,
                primary_key=info.primary_key_position > 0,
                description=table_doc.columns.get(info.name, ColumnDoc()).description,
                sensitivity=table_doc.columns.get(info.name, ColumnDoc()).sensitivity,
            )
            for info in infos
        )
        merged[name] = TableMeta(name=name, description=table_doc.description, columns=columns)
    return merged


def _column(tables: dict[str, TableMeta], qualified: str) -> ColumnMeta:
    table_name, column_name = qualified.split(".")
    table = tables.get(table_name)
    column = table.column(column_name) if table else None
    if column is None:
        raise UnknownIdentifierError(f"relationship names unknown column {qualified!r}")
    return column


def _merge_relationships(
    doc: SourceDoc, connector: Connector, tables: dict[str, TableMeta]
) -> list[Relationship]:
    candidates: list[tuple[str, str, bool, RelationshipOrigin]] = []
    for rel in doc.relationships:
        _column(tables, rel.parent)
        candidates.append(
            (rel.child, rel.parent, rel.drives_cardinality, RelationshipOrigin.METADATA)
        )
    declared = {child for child, *_ in candidates}
    for table in sorted(tables):
        for key in connector.foreign_keys(table):
            child = f"{table}.{key.column}"
            if child not in declared:
                parent = f"{key.parent_table}.{key.parent_column}"
                candidates.append((child, parent, False, RelationshipOrigin.PRAGMA))
                declared.add(child)
    relationships = [
        _relationship(child, parent, drives, origin, _column(tables, child).nullable)
        for child, parent, drives, origin in candidates
    ]
    return _assign_cardinality_drivers(relationships)


def _relationship(
    child: str, parent: str, drives: bool, origin: RelationshipOrigin, nullable: bool
) -> Relationship:
    child_table, child_column = child.split(".")
    parent_table, parent_column = parent.split(".")
    if drives and child_table == parent_table:
        raise MetadataError(f"self-reference {child} cannot drive cardinality")
    return Relationship(
        child_table=child_table,
        child_column=child_column,
        parent_table=parent_table,
        parent_column=parent_column,
        nullable=nullable,
        drives_cardinality=drives,
        origin=origin,
    )


def _assign_cardinality_drivers(relationships: Sequence[Relationship]) -> list[Relationship]:
    """Exactly one non-self parent per child sizes it (PLAN Q4).

    Default: the first non-nullable relationship listed, else the first listed.
    """
    chosen: dict[str, Relationship] = {}
    for child in dict.fromkeys(r.child_table for r in relationships):
        own = [r for r in relationships if r.child_table == child and not r.is_self_reference]
        marked = [r for r in own if r.drives_cardinality]
        if len(marked) > 1:
            labels = [r.label() for r in marked]
            raise MetadataError(f"{child} has several cardinality drivers: {labels}")
        if own:
            chosen[child] = (
                marked[0] if marked else next((r for r in own if not r.nullable), own[0])
            )
    return [
        r.model_copy(update={"drives_cardinality": chosen.get(r.child_table) == r})
        for r in relationships
    ]
