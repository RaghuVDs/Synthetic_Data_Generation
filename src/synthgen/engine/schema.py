"""Output columns: Arrow types per column type, the label columns every row carries, conversion."""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from synthgen.engine.generators import Values
from synthgen.engine.writer import ColumnSchema
from synthgen.spec.models import ColumnSpec, ColumnType, TableSpec

SCENARIO_COLUMN = "_scenario"
CATEGORY_COLUMN = "_category"
DEFECT_COLUMN = "_defect"
LABEL_COLUMNS = (SCENARIO_COLUMN, CATEGORY_COLUMN, DEFECT_COLUMN)
ARROW_TYPES = {
    ColumnType.INTEGER: pa.int64(),
    ColumnType.FLOAT: pa.float64(),
    ColumnType.BOOLEAN: pa.int64(),
    ColumnType.TEXT: pa.string(),
    ColumnType.DATE: pa.string(),
    ColumnType.DATETIME: pa.string(),
}


def output_schema(table: TableSpec) -> list[ColumnSchema]:
    """Every written column with its Arrow type, label columns last."""
    columns = [ColumnSchema(name=c.name, type=str(ARROW_TYPES[c.type])) for c in table.columns]
    return columns + [ColumnSchema(name=n, type=str(pa.string())) for n in LABEL_COLUMNS]


def to_arrow(table: TableSpec, values: dict[str, Values], labels: dict[str, Values]) -> pa.Table:
    """One chunk as an Arrow table with fixed column order and types."""
    arrays = [pa.array(_cast(values[c.name], c), type=ARROW_TYPES[c.type]) for c in table.columns]
    arrays += [pa.array(labels[n], type=pa.string()) for n in LABEL_COLUMNS]
    return pa.table(arrays, names=[c.name for c in table.columns] + list(LABEL_COLUMNS))


def _cast(values: Values, column: ColumnSpec) -> list[Any]:  # Any: Arrow takes mixed Python values
    """Python values Arrow can type: whole numbers for integer columns, text for text columns."""
    out: list[Any] = []
    for v in values.tolist():
        if v is None:
            out.append(None)
        elif column.type in (ColumnType.INTEGER, ColumnType.BOOLEAN):
            out.append(round(float(v)))
        elif column.type is ColumnType.FLOAT:
            out.append(float(v))
        else:
            out.append(str(v))
    return out
