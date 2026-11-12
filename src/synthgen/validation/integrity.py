"""Integrity checks (SPEC 8.6): keys, references, required values, types and formats.

Integrity is a hard gate. When every failing row belongs to scenario rows, the result names
those scenarios so routing can send the failure back to the overlay (PLAN A5).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from synthgen.domain.report import CheckFamily, CheckResult, CheckStatus
from synthgen.engine.defects import DEFECT_COLUMN
from synthgen.engine.runner import ARROW_TYPES, SCENARIO_COLUMN
from synthgen.engine.writer import Manifest, read_table
from synthgen.spec.models import TYPICAL, ColumnSpec, ColumnType, GenerationSpec, TableSpec

_FORMATS = {
    ColumnType.DATE: re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    ColumnType.DATETIME: re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"),
}


def check_integrity(spec: GenerationSpec, out_dir: Path, manifest: Manifest) -> list[CheckResult]:
    """Every integrity check on every written table."""
    loaded = {
        t.name: read_table(out_dir, entry)
        for t in spec.tables
        if (entry := manifest.table(t.name)) is not None
    }
    results: list[CheckResult] = []
    for table in spec.tables:
        data = loaded.get(table.name)
        if data is None:
            results.append(_result((table.name, None), "table_written", 1, detail="table missing"))
            continue
        scenarios = np.asarray(data.column(SCENARIO_COLUMN).to_pylist(), dtype=object)
        rows = RowInfo(scenarios=scenarios, clean=_clean_rows(data))
        for column in table.columns:
            values = data.column(column.name)
            results += _column_checks(table, column, values, rows, loaded)
    return results


@dataclass(frozen=True)
class RowInfo:
    """Per-row facts every column check needs: the scenario, and whether a defect is labelled."""

    scenarios: np.ndarray[Any, Any]
    clean: np.ndarray[Any, Any]


def _clean_rows(data: pa.Table) -> np.ndarray[Any, Any]:
    """Rows without a deliberate, labelled defect; only these must pass integrity."""
    if DEFECT_COLUMN not in data.column_names:
        return np.ones(data.num_rows, dtype=bool)
    return np.array([d is None for d in data.column(DEFECT_COLUMN).to_pylist()], dtype=bool)


def _result(
    where: tuple[str, str | None],
    check: str,
    violations: int,
    *,
    scenario: str | None = None,
    detail: str | None = None,
) -> CheckResult:
    table, column = where
    return CheckResult(
        family=CheckFamily.INTEGRITY,
        check=check,
        table=table,
        column=column,
        metric="violations",
        value=float(violations),
        threshold=0.0,
        status=CheckStatus.PASS if violations == 0 else CheckStatus.FAIL,
        scenario=scenario,
        detail=detail,
    )


def _traced(bad: np.ndarray[Any, Any], scenarios: np.ndarray[Any, Any]) -> str | None:
    """The overlay scenarios behind failing rows, or None when any failing row is typical."""
    names = set(scenarios[bad].tolist())
    if not names or TYPICAL in names:
        return None
    return ",".join(sorted(names))


def _column_checks(
    table: TableSpec,
    column: ColumnSpec,
    values: pa.ChunkedArray[Any],
    rows: RowInfo,
    loaded: dict[str, pa.Table],
) -> list[CheckResult]:
    py = np.asarray(values.to_pylist(), dtype=object)
    present = np.array([v is not None for v in py], dtype=bool)
    checks: list[tuple[str, np.ndarray[Any, Any]]] = []
    if not column.nullable or column.primary_key:
        checks.append(("not_null", ~present))
    if column.primary_key or column.unique:
        checks.append(
            (
                "primary_key_unique" if column.primary_key else "unique",
                _repeats(py, present & rows.clean),
            )
        )
    reference = _reference(table, column)
    if reference is not None and reference[0] in loaded:
        parent_keys = set(loaded[reference[0]].column(reference[1]).to_pylist())
        checks.append(
            ("foreign_key_resolves", present & np.array([v not in parent_keys for v in py]))
        )
    pattern = _FORMATS.get(column.type)
    if pattern is not None:
        checks.append(("format", present & np.array([not pattern.match(str(v)) for v in py])))
    results = [
        _result(
            (table.name, column.name),
            name,
            int((bad & rows.clean).sum()),
            scenario=_traced(bad & rows.clean, rows.scenarios),
        )
        for name, bad in checks
    ]
    type_ok = values.type == ARROW_TYPES[column.type]
    results.append(
        _result(
            (table.name, column.name),
            "type",
            0 if type_ok else 1,
            detail=None if type_ok else str(values.type),
        )
    )
    return results


def _repeats(values: np.ndarray[Any, Any], present: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """True on every non-null row whose value appeared earlier."""
    seen: set[Any] = set()
    out = np.zeros(values.size, dtype=bool)
    for i in np.flatnonzero(present):
        value = values[i]
        out[i] = value in seen
        seen.add(value)
    return out


def _reference(table: TableSpec, column: ColumnSpec) -> tuple[str, str] | None:
    """The (table, column) a key column must point into, if it is a key."""
    rule = table.row_rule
    if rule.kind == "per_parent" and rule.child_column == column.name:
        return rule.parent_table, rule.parent_column
    gen = column.generator
    if gen.type == "foreign_key":
        return gen.parent_table, gen.parent_column
    return None


def failed(results: Sequence[CheckResult]) -> list[CheckResult]:
    """Results that failed."""
    return [r for r in results if r.status is CheckStatus.FAIL]
