"""Checks on table constraints and scenario defects (SPEC 8.4 spec validation)."""

from __future__ import annotations

from synthgen.domain.errors import ExpressionError
from synthgen.spec.checks import GeneratorChecker
from synthgen.spec.constraints import (
    Constraint,
    DefectKind,
    DefectSpec,
    ExpressionConstraint,
    OrderConstraint,
    RangeConstraint,
)
from synthgen.spec.expressions import TIMES, ExprType, parse, type_of
from synthgen.spec.models import ColumnSpec, ColumnType, GenerationSpec, ScenarioSpec, TableSpec
from synthgen.spec.shape_checks import ancestors

_ORDERED = (ExprType.NUMBER, *TIMES)
_DEFECT_TYPES: dict[DefectKind, tuple[ColumnType, ...]] = {
    DefectKind.OUT_OF_RANGE: (
        ColumnType.INTEGER,
        ColumnType.FLOAT,
        ColumnType.DATE,
        ColumnType.DATETIME,
    ),
    DefectKind.BAD_FORMAT: (ColumnType.TEXT, ColumnType.DATE, ColumnType.DATETIME),
}


def constraint_problem(
    checker: GeneratorChecker, table: TableSpec, constraint: Constraint
) -> str | None:
    """Why the constraint cannot apply to this table, or None when it can."""
    types = checker.expression_types(table)
    if isinstance(constraint, OrderConstraint):
        return _order_problem(constraint, types)
    if isinstance(constraint, RangeConstraint):
        return _range_problem(constraint, table)
    if isinstance(constraint, ExpressionConstraint):
        try:
            found = type_of(parse(constraint.expression), types)
        except ExpressionError as error:
            return f"constraint {constraint.expression!r}: {error}"
        return None if found is ExprType.BOOLEAN else "an expression constraint must be true/false"
    names = constraint.columns if constraint.kind == "unique" else [constraint.column]
    missing = [n for n in names if table.column(n) is None]
    return f"unknown column {table.name}.{missing[0]}" if missing else None


def _order_problem(constraint: OrderConstraint, types: dict[str, ExprType]) -> str | None:
    for name in (constraint.lower, constraint.upper):
        if name not in types:
            return f"unknown column {name}"
    if "." in constraint.upper:
        return "the upper side of an order must be a column of the table itself"
    pair = {types[constraint.lower], types[constraint.upper]}
    if not pair <= set(_ORDERED) or (ExprType.NUMBER in pair and len(pair) > 1):
        return f"{constraint.lower} and {constraint.upper} cannot be ordered against each other"
    return None


def _range_problem(constraint: RangeConstraint, table: TableSpec) -> str | None:
    column = table.column(constraint.column)
    if column is None:
        return f"unknown column {table.name}.{constraint.column}"
    numeric = column.type in (ColumnType.INTEGER, ColumnType.FLOAT, ColumnType.BOOLEAN)
    for bound in (constraint.minimum, constraint.maximum):
        if bound is None:
            continue
        if isinstance(bound, bool) or numeric != isinstance(bound, float | int):
            return f"range bound {bound!r} does not suit {column.type} column {column.name}"
    return None


def defect_problem(
    spec: GenerationSpec, scenario: ScenarioSpec, defect: DefectSpec, *, allowed: bool
) -> str | None:
    """Why the defect cannot apply, or None when it can. Defects are opt-in (include_invalid)."""
    if not allowed:
        return "defects need a request that asks for invalid data (include_invalid)"
    table = spec.table(defect.table)
    if table is None:
        return f"unknown table {defect.table}"
    if defect.table != scenario.entity and scenario.entity not in ancestors(spec, table):
        return f"{defect.table} is neither the scenario entity nor below it"
    column = table.column(defect.column)
    if column is None:
        return f"unknown column {table.name}.{defect.column}"
    return _defect_column_problem(table, column, defect.kind)


def _defect_column_problem(table: TableSpec, column: ColumnSpec, kind: DefectKind) -> str | None:
    """Why this kind of defect makes no sense on this column, or None when it does."""
    wanted = _DEFECT_TYPES.get(kind)
    if wanted is not None and column.type not in wanted:
        return f"{kind.value} does not suit {column.type} column {column.name}"
    if kind is DefectKind.DUPLICATE_KEY and not (column.primary_key or column.unique):
        return f"duplicate_key needs a key or unique column, not {column.name}"
    rule = table.row_rule
    is_driving = rule.kind == "per_parent" and rule.child_column == column.name
    if kind is DefectKind.ORPHAN_KEY and not (is_driving or column.generator.type == "foreign_key"):
        return f"orphan_key needs a foreign key column, not {column.name}"
    return None
