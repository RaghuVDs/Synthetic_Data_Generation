"""Table-level statistics (SPEC 8.3 A) and the schema hash that keys cached profiles."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

import pyarrow as pa

from synthgen.domain.metadata import Relationship, TableMeta
from synthgen.profiling.models import ColumnProfile, Measured, StatSource

PROFILE_FORMAT = 4
"""Bumped when profile models change, so cached profiles of an older shape are not reused."""


def schema_hash(meta: TableMeta) -> str:
    """Hash of the profile format and the table's columns; a change invalidates cached profiles."""
    shape = [(c.name, c.declared_type, c.nullable, c.primary_key, c.unique) for c in meta.columns]
    tags = [(c.name, c.sensitivity) for c in meta.columns]
    payload = json.dumps([PROFILE_FORMAT, meta.name, shape, tags])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def primary_key_unique(
    meta: TableMeta, columns: Sequence[ColumnProfile], rows: int
) -> Measured[bool] | None:
    """Whether a single-column primary key holds a distinct value on every row."""
    keys = [c for c in columns if c.name in meta.primary_key]
    if len(keys) != 1:
        return None
    return Measured[bool](value=keys[0].distinct_count.value == rows, source=StatSource.DERIVED)


def identifier_columns(meta: TableMeta, relationships: Sequence[Relationship]) -> set[str]:
    """Primary-key, declared-unique and foreign-key columns: identifiers, not measurements."""
    keys = {c.name for c in meta.columns if c.primary_key or c.unique}
    return keys | {r.child_column for r in relationships if r.child_table == meta.name}


def duplicate_rate(sample: pa.Table, columns: list[str]) -> float:
    """Share of sampled rows that repeat an earlier row on the given columns."""
    if not columns or sample.num_rows == 0:
        return 0.0
    rows = list(zip(*(sample.column(c).to_pylist() for c in columns), strict=True))
    return 1.0 - len(set(rows)) / len(rows)
