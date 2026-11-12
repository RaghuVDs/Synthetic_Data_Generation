"""Integrity rules beyond keys: value formats and table constraints, as per-row violation masks."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from synthgen.spec.constraints import (
    Constraint,
    ExpressionConstraint,
    OrderConstraint,
    RangeConstraint,
    UniqueConstraint,
)
from synthgen.spec.dependencies import constraint_reads
from synthgen.spec.evaluator import evaluate, to_internal
from synthgen.spec.expressions import ExprType, parse
from synthgen.spec.generators import Generator, PatternGen, SemanticGen, SemanticKind
from synthgen.spec.models import (
    EXPR_TYPES,
    TYPICAL,
    ColumnSpec,
    ColumnType,
    GenerationSpec,
    TableSpec,
)
from synthgen.validation.output import Cells, OutputData

DATE_FORMATS = {
    ColumnType.DATE: re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    ColumnType.DATETIME: re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"),
}
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
LUHN_BASE = 10
Violation = tuple[str, str, Cells]
"""(check name, column or columns, per-row violation mask)."""


@dataclass(frozen=True)
class RowInfo:
    """Per-row facts every check needs: the scenario, and whether no defect is labelled."""

    scenarios: Cells
    clean: Cells


def traced(failing: Cells, scenarios: Cells) -> str | None:
    """The overlay scenarios behind failing rows, or None when any failing row is typical."""
    names = set(scenarios[np.asarray(failing, dtype=bool)].tolist())
    if not names or TYPICAL in names:
        return None
    return ",".join(sorted(names))


def luhn_valid(text: str) -> bool:
    """True when the digits of `text` pass the Luhn check."""
    digits = [int(c) for c in text if c.isdigit()][::-1]
    if not digits:
        return False
    total = sum(d if i % 2 == 0 else sum(divmod(2 * d, LUHN_BASE)) for i, d in enumerate(digits))
    return total % LUHN_BASE == 0


def format_violations(
    spec: GenerationSpec, table: TableSpec, column: ColumnSpec, values: Cells, scenarios: Cells
) -> list[Cells]:
    """Rows whose value breaks its column's format: ISO dates, masks, check digits, emails.

    Generator formats apply only to rows the baseline generator made; rows of scenarios that
    rewrite the column follow their overlay instead.
    """
    present = np.not_equal(values, None)
    found: list[Cells] = []
    date_format = DATE_FORMATS.get(column.type)
    if date_format is not None:
        found.append(present & np.array([not date_format.match(str(v)) for v in values]))
    check = _generator_format(column.generator)
    if check is not None:
        rewritten = _rewriting_scenarios(spec, table.name, column.name)
        governed = present & ~np.isin(scenarios, list(rewritten))
        found.append(governed & np.array([not check(str(v)) for v in values]))
    return found


def _rewriting_scenarios(spec: GenerationSpec, table: str, column: str) -> set[str]:
    """Scenarios that give a column other values than its baseline generator."""
    found: set[str] = set()
    for scenario in spec.scenarios:
        columns = {o.column for o in scenario.overrides if o.table == table}
        columns |= {n for c in scenario.correlations if c.table == table for n in c.columns}
        for sequence in (q for q in scenario.sequences if q.table == table):
            columns |= {sequence.time_column}
            columns |= {c.column for step in sequence.steps for c in step.columns}
        if column in columns:
            found.add(scenario.name)
    return found


def _generator_format(gen: Generator) -> Callable[[str], bool] | None:
    if isinstance(gen, PatternGen):
        pattern = re.compile("^(?:" + "|".join(_mask_regex(m) for m in gen.masks) + ")$")
        if gen.check_digit == "luhn":
            return lambda text: bool(pattern.match(text)) and luhn_valid(text)
        return lambda text: bool(pattern.match(text))
    if isinstance(gen, SemanticGen) and gen.kind is SemanticKind.EMAIL:
        return lambda text: bool(EMAIL.match(text))
    if isinstance(gen, SemanticGen) and gen.kind is SemanticKind.CARD_NUMBER:
        return luhn_valid
    return None


def _mask_regex(mask: str) -> str:
    return "".join("[A-Za-z]" if c == "A" else "[0-9]" if c == "9" else re.escape(c) for c in mask)


def constraint_violations(
    spec: GenerationSpec, table: TableSpec, data: OutputData
) -> list[Violation]:
    """One violation mask per table constraint (orders against a parent join the driving key)."""
    types = {c.name: EXPR_TYPES[c.type] for c in table.columns}
    columns = _ColumnSource(spec, table, data)
    found: list[Violation] = []
    for constraint in table.constraints:
        names = sorted(constraint_reads(constraint))
        found.append(
            (f"constraint_{constraint.kind}", ",".join(names), _broken(constraint, columns, types))
        )
    return found


class _ColumnSource:
    """Columns of one table, and driving-parent columns aligned to its rows."""

    def __init__(self, spec: GenerationSpec, table: TableSpec, data: OutputData) -> None:
        """Read lazily from `data`."""
        self._spec = spec
        self._table = table
        self._data = data

    def values(self, name: str) -> Cells:
        if "." not in name:
            return self._data.column(self._table.name, name)
        rule = self._table.row_rule
        parent, column = name.split(".", 1)
        if rule.kind != "per_parent" or rule.parent_table != parent:
            raise ValueError(f"{name} is not a column of the driving parent")
        keys = self._data.column(parent, rule.parent_column)
        order = np.argsort(keys, kind="stable")
        children = self._data.column(self._table.name, rule.child_column)
        at = np.clip(np.searchsorted(keys[order], children), 0, max(keys.size - 1, 0))
        found = keys[order][at] == children if keys.size else np.zeros(children.size, bool)
        aligned = self._data.column(parent, column)[order][at].astype(object)
        aligned[~np.asarray(found, dtype=bool)] = None
        return aligned

    def kind(self, name: str, types: dict[str, ExprType]) -> ExprType:
        if "." not in name:
            return types[name]
        parent, column = name.split(".", 1)
        spec_table = self._spec.table(parent)
        found = spec_table.column(column) if spec_table else None
        return EXPR_TYPES[found.type] if found else ExprType.TEXT


def _broken(constraint: Constraint, columns: _ColumnSource, types: dict[str, ExprType]) -> Cells:
    if isinstance(constraint, OrderConstraint):
        lower = to_internal(columns.values(constraint.lower), columns.kind(constraint.lower, types))
        upper = to_internal(columns.values(constraint.upper), types[constraint.upper])
        return np.asarray(lower > upper, dtype=bool)
    if isinstance(constraint, RangeConstraint):
        kind = types[constraint.column]
        current = to_internal(columns.values(constraint.column), kind)
        low, high = (
            to_internal(np.array([b], dtype=object), kind)[0] if b is not None else np.nan
            for b in (constraint.minimum, constraint.maximum)
        )
        below = current < low if not np.isnan(low) else np.zeros(current.size, bool)
        above = current > high if not np.isnan(high) else np.zeros(current.size, bool)
        return np.asarray(below | above, dtype=bool)
    if isinstance(constraint, UniqueConstraint):
        rows = zip(*(columns.values(c).tolist() for c in constraint.columns), strict=True)
        seen: set[Any] = set()
        flags = []
        for row in rows:
            flags.append(row in seen)
            seen.add(row)
        return np.array(flags, dtype=bool)
    if isinstance(constraint, ExpressionConstraint):
        tree = parse(constraint.expression)
        names = sorted(constraint_reads(constraint))
        frame = {n: (columns.kind(n, types), columns.values(n)) for n in names}
        n = next(iter(frame.values()))[1].size if frame else 0
        return np.asarray(evaluate(tree, frame, n) == 0, dtype=bool)
    return np.equal(columns.values(constraint.column), None)
