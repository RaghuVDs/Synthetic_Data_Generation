"""Semantic checks on a merged spec (SPEC 8.4). Schema checks already ran when the spec parsed.

Each error says whether it comes from the baseline (a code defect, the job fails) or from an
overlay (the overlay writer can repair it).
"""

from __future__ import annotations

from typing import Literal

from synthgen.config import GenerationSettings
from synthgen.spec.generators import Generator
from synthgen.spec.models import (
    ColumnOverride,
    ColumnSpec,
    ColumnType,
    GenerationSpec,
    SpecError,
    TableSpec,
)

SHARE_TOLERANCE = 1e-9
_NUMERIC = (ColumnType.INTEGER, ColumnType.FLOAT, ColumnType.BOOLEAN)
_SUITS: dict[str, tuple[ColumnType, ...]] = {
    "sequence": (ColumnType.INTEGER,),
    "uuid": (ColumnType.TEXT,),
    "empirical": _NUMERIC,
    "datetime": (ColumnType.DATE, ColumnType.DATETIME),
    "pattern": (ColumnType.TEXT,),
}
_SENSITIVE_OK = ("pattern", "constant")
Origin = Literal["baseline", "overlay"]


def validate_spec(spec: GenerationSpec, settings: GenerationSettings) -> list[SpecError]:
    """Every problem found; an empty list means the spec may run."""
    errors: list[SpecError] = []
    seen: list[str] = []
    for i, table in enumerate(spec.tables):
        errors += _table_errors(spec, table, f"/tables/{i}", seen, settings)
        seen.append(table.name)
    errors += _scenario_errors(spec)
    return errors


def _error(origin: Origin, path: str, message: str, scenario: str | None = None) -> SpecError:
    return SpecError(origin=origin, scenario=scenario, path=path, message=message)


def _table_errors(
    spec: GenerationSpec,
    table: TableSpec,
    path: str,
    earlier: list[str],
    settings: GenerationSettings,
) -> list[SpecError]:
    errors: list[SpecError] = []
    if table.name in earlier:
        errors.append(_error("baseline", path, f"table {table.name} appears twice"))
    names = [c.name for c in table.columns]
    if len(set(names)) != len(names):
        errors.append(_error("baseline", f"{path}/columns", "column names repeat"))
    rule = table.row_rule
    if rule.kind == "fixed" and rule.rows > settings.max_rows_per_table:
        errors.append(_error("baseline", f"{path}/row_rule", f"{rule.rows} rows exceed the limit"))
    if rule.kind == "per_parent" and rule.parent_table not in earlier:
        errors.append(
            _error("baseline", f"{path}/row_rule", f"parent {rule.parent_table} is not earlier")
        )
    for j, column in enumerate(table.columns):
        problem = _generator_problem(spec, column, column.generator, earlier)
        if problem:
            errors.append(_error("baseline", f"{path}/columns/{j}/generator", problem))
    return errors


def _generator_problem(
    spec: GenerationSpec, column: ColumnSpec, gen: Generator, earlier: list[str]
) -> str | None:
    """Why this generator cannot fill this column, or None when it can."""
    allowed = _SUITS.get(gen.type)
    if allowed is not None and column.type not in allowed:
        return f"{gen.type} generator does not suit {column.type} column {column.name}"
    if column.sensitive and gen.type not in _SENSITIVE_OK and not column.primary_key:
        return f"sensitive column {column.name} must use a pattern generator"
    if gen.type == "datetime" and (gen.granularity == "datetime") != (
        column.type is ColumnType.DATETIME
    ):
        return f"datetime granularity {gen.granularity} does not suit {column.type} column"
    if gen.type == "foreign_key":
        parent = spec.table(gen.parent_table)
        if gen.parent_table not in earlier or parent is None:
            return f"foreign key to {gen.parent_table}, which is not generated earlier"
        key = parent.column(gen.parent_column)
        if key is None or not (key.primary_key or key.unique):
            return f"foreign key target {gen.parent_table}.{gen.parent_column} is not a unique key"
    return None


def _scenario_errors(spec: GenerationSpec) -> list[SpecError]:
    errors: list[SpecError] = []
    names = [s.name for s in spec.scenarios]
    if len(set(names)) != len(names):
        errors.append(_error("overlay", "/scenarios", "scenario names repeat"))
    total = sum(s.share for s in spec.scenarios)
    if total > 1.0 + SHARE_TOLERANCE:
        errors.append(_error("overlay", "/scenarios", f"shares sum to {total:.4f}, above 1"))
    for i, scenario in enumerate(spec.scenarios):
        path = f"/scenarios/{i}"
        problems: list[tuple[str, str | None]] = [
            (
                f"{path}/entity",
                None if spec.table(scenario.entity) else f"unknown table {scenario.entity}",
            )
        ]
        problems += [
            (f"{path}/labels/{j}", _missing(spec, label.table, label.column))
            for j, label in enumerate(scenario.labels)
        ]
        problems += [
            (f"{path}/overrides/{j}", _override_problem(spec, override))
            for j, override in enumerate(scenario.overrides)
        ]
        errors += [_error("overlay", at, p, scenario.name) for at, p in problems if p]
    return errors


def _override_problem(spec: GenerationSpec, override: ColumnOverride) -> str | None:
    """Why an override cannot apply, or None when it can."""
    missing = _missing(spec, override.table, override.column)
    table = spec.table(override.table)
    column = table.column(override.column) if table else None
    if missing or column is None:
        return missing
    order = [t.name for t in spec.tables]
    return _generator_problem(
        spec, column, override.generator, order[: order.index(override.table)]
    )


def _missing(spec: GenerationSpec, table: str, column: str) -> str | None:
    found = spec.table(table)
    if found is None:
        return f"unknown table {table}"
    if found.column(column) is None:
        return f"unknown column {table}.{column}"
    return None
