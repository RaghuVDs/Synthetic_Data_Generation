"""Column batching for wide tables: no single query or sample grows with table width.

A table may have thousands of columns. SQLite caps result columns per SELECT, and a sample of
every column holds rows x columns values in memory, so both run in batches of columns and the
sample shrinks in rows as the table widens.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pyarrow as pa

from synthgen.config import ProfilingSettings
from synthgen.domain.ports import Connector
from synthgen.domain.query import AggExpr


def sample_rows(columns: int, settings: ProfilingSettings) -> int:
    """Rows to sample: the configured size, reduced so rows x columns fits the cell budget."""
    return max(1, min(settings.sample_size, settings.max_sample_cells // max(1, columns)))


def batched_sample(  # noqa: PLR0913 - a sample request plus its batch size
    connector: Connector,
    table: str,
    columns: Sequence[str],
    *,
    rows: int,
    seed: int,
    batch: int,
) -> pa.Table:
    """One sample read in column batches. The rowid hash picks the same rows in every batch."""
    parts = [
        connector.sample(table, list(columns[start : start + batch]), rows, seed)
        for start in range(0, len(columns), batch)
    ]
    arrays = [part.column(i) for part in parts for i in range(part.num_columns)]
    names = [name for part in parts for name in part.column_names]
    return pa.table(arrays, names=names)


def batched_aggregate(
    connector: Connector, table: str, exprs: Sequence[AggExpr], batch: int
) -> dict[str, Any]:  # Any: SQLite aggregate cells (int, float, str or None)
    """Whole-table aggregates in batches of at most `batch` result columns, merged into one row."""
    merged: dict[str, Any] = {}
    for start in range(0, len(exprs), batch):
        [row] = connector.aggregate(table, list(exprs[start : start + batch]))
        merged.update(row)
    return merged
