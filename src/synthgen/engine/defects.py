"""Deliberately invalid values (rule_break, opt-in), each row labelled in `_defect` (SPEC 8.5).

Defects run last, after overlays and sequences, so nothing later repairs them. A label reads
`kind:column`; several defects on one row are joined with `;`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from synthgen.domain.errors import EngineError
from synthgen.engine.columns import TableDraw
from synthgen.engine.generators import KeyLookup, Values
from synthgen.spec.constraints import DefectKind, DefectSpec
from synthgen.spec.evaluator import to_internal
from synthgen.spec.expressions import ExprType
from synthgen.spec.models import ColumnType, ScenarioSpec, TableSpec

OUT_OF_RANGE_FACTOR = 10.0
"""An out-of-range number is this many times the column's largest absolute value, plus one."""
FAR_FUTURE = {ColumnType.DATE: "2999-12-31", ColumnType.DATETIME: "2999-12-31 23:59:59"}
ORPHAN_PREFIX = "orphan-"
Rows = NDArray[np.int64]


def apply_defects(
    draw: TableDraw,
    scenarios: Sequence[ScenarioSpec],
    values: dict[str, Values],
    assigned: NDArray[np.object_],
) -> NDArray[np.object_]:
    """Break values on a share of each scenario's rows; returns the `_defect` column."""
    labels = np.full(assigned.size, None, dtype=object)
    for scenario in scenarios:
        rows = np.flatnonzero(assigned == scenario.name).astype(np.int64)
        for defect in (d for d in scenario.defects if d.table == draw.table.name and rows.size):
            chosen = _break(draw, defect, values, rows)
            label = f"{defect.kind.value}:{defect.column}"
            labels[chosen] = [label if old is None else f"{old};{label}" for old in labels[chosen]]
    return labels


def _break(draw: TableDraw, defect: DefectSpec, values: dict[str, Values], rows: Rows) -> Rows:
    """Break one column on `rate` of the rows (at least one); returns the rows broken."""
    count = min(rows.size, max(1, round(defect.rate * rows.size)))
    chosen = np.sort(draw.rng.choice(rows, size=count, replace=False)).astype(np.int64)
    column = draw.table.column(defect.column)
    if column is None:
        raise EngineError(f"defect on unknown column {draw.table.name}.{defect.column}")
    if defect.kind is DefectKind.ORPHAN_KEY:
        new = _orphans(draw.table, defect.column, chosen.size, draw.ctx.parent_keys)
    else:
        new = _broken(defect.kind, column.type, values[defect.column], chosen, draw.rng)
    updated = values[defect.column].astype(object)
    updated[chosen] = new
    values[defect.column] = updated
    return chosen


def _broken(
    kind: DefectKind,
    column_type: ColumnType,
    current: Values,
    chosen: Rows,
    rng: np.random.Generator,
) -> Values:
    """Null, duplicate, out-of-range or misformatted values (orphan keys are made elsewhere)."""
    if kind is DefectKind.NULL:
        return np.full(chosen.size, None, dtype=object)
    if kind is DefectKind.DUPLICATE_KEY:
        donors = np.setdiff1d(np.arange(current.size), chosen)
        source = donors if donors.size else np.arange(current.size)
        return current[rng.choice(source, size=chosen.size)]
    if kind is DefectKind.OUT_OF_RANGE:
        if column_type in FAR_FUTURE:
            return np.full(chosen.size, FAR_FUTURE[column_type], dtype=object)
        numbers = to_internal(current, ExprType.NUMBER)
        largest = float(np.nanmax(np.abs(numbers))) if np.isfinite(numbers).any() else 0.0
        return np.full(chosen.size, largest * OUT_OF_RANGE_FACTOR + 1.0, dtype=object)
    return np.array([_misformat(v, column_type) for v in current[chosen]], dtype=object)


def _misformat(value: object, column_type: ColumnType) -> str:
    """A value breaking its column's format: a date as day/month/year, else text reversed."""
    text = "" if value is None else str(value)
    if column_type in FAR_FUTURE and len(text) >= len("2000-01-01"):
        return f"{text[8:10]}/{text[5:7]}/{text[0:4]}{text[10:]}"
    return f"#{text[::-1]}#"


def _orphans(table: TableSpec, name: str, count: int, parent_keys: KeyLookup) -> Values:
    """Keys that match no parent row: past the largest integer key, or marked text."""
    rule = table.row_rule
    column = table.column(name)
    gen = column.generator if column else None
    if gen is not None and gen.type == "foreign_key":
        keys = parent_keys(gen.parent_table, gen.parent_column)
    elif rule.kind == "per_parent" and rule.child_column == name:
        keys = parent_keys(rule.parent_table, rule.parent_column)
    else:
        keys = np.array([], dtype=object)
    if keys.size and np.issubdtype(keys.dtype, np.integer):
        return (int(keys.max()) + 1 + np.arange(count)).astype(object)
    return np.array([f"{ORPHAN_PREFIX}{i}" for i in range(count)], dtype=object)
