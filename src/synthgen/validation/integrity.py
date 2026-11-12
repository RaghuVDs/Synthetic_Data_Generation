"""Integrity checks (SPEC 8.6): keys, references, required values, types, formats, constraints.

Integrity is a hard gate. Rows with a labelled, deliberate defect are exempt. When every failing
row belongs to scenario rows, the result names those scenarios so routing can send the failure
back to the overlay (PLAN A5).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from synthgen.domain.report import CheckFamily, CheckResult, CheckStatus
from synthgen.engine.schema import ARROW_TYPES, DEFECT_COLUMN, SCENARIO_COLUMN
from synthgen.spec.models import ColumnSpec, GenerationSpec, TableSpec
from synthgen.validation.integrity_rules import (
    RowInfo,
    constraint_violations,
    format_violations,
    traced,
)
from synthgen.validation.output import Cells, OutputData


def check_integrity(spec: GenerationSpec, data: OutputData) -> list[CheckResult]:
    """Every integrity check on every written table."""
    results: list[CheckResult] = []
    for table in spec.tables:
        if data.entry(table.name) is None:
            results.append(_result((table.name, None), "table_written", 1, detail="table missing"))
            continue
        rows = RowInfo(
            scenarios=data.label(table.name, SCENARIO_COLUMN),
            clean=np.equal(data.label(table.name, DEFECT_COLUMN), None),
        )
        for column in table.columns:
            results += _column_checks(spec, table, column, rows, data)
        for name, where, bad in constraint_violations(spec, table, data):
            results.append(_counted((table.name, where), name, bad, rows))
    return results


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


def _counted(where: tuple[str, str | None], check: str, bad: Cells, rows: RowInfo) -> CheckResult:
    """A result over clean rows only, traced to the scenarios of its failing rows."""
    failing = np.asarray(bad, dtype=bool) & rows.clean
    return _result(where, check, int(failing.sum()), scenario=traced(failing, rows.scenarios))


def _column_checks(
    spec: GenerationSpec, table: TableSpec, column: ColumnSpec, rows: RowInfo, data: OutputData
) -> list[CheckResult]:
    values = data.column(table.name, column.name)
    present = np.not_equal(values, None)
    checks: list[tuple[str, Cells]] = []
    if not column.nullable or column.primary_key:
        checks.append(("not_null", ~present))
    if column.primary_key or column.unique:
        name = "primary_key_unique" if column.primary_key else "unique"
        checks.append((name, _repeats(values, present & rows.clean)))
    reference = _reference(table, column)
    if reference is not None and data.entry(reference[0]) is not None:
        parent_keys = set(data.column(*reference).tolist())
        checks.append(
            ("foreign_key_resolves", present & np.array([v not in parent_keys for v in values]))
        )
    found = format_violations(spec, table, column, values, rows.scenarios)
    checks += [("format", bad) for bad in found]
    results = [_counted((table.name, column.name), name, bad, rows) for name, bad in checks]
    written = data.arrow_type(table.name, column.name)
    type_ok = written == str(ARROW_TYPES[column.type])
    results.append(
        _result(
            (table.name, column.name),
            "type",
            0 if type_ok else 1,
            detail=None if type_ok else written,
        )
    )
    return results


def _repeats(values: Cells, considered: Cells) -> Cells:
    """True on every considered row whose value appeared on an earlier considered row."""
    seen: set[Any] = set()
    out = np.zeros(values.size, dtype=bool)
    for i in np.flatnonzero(considered):
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
