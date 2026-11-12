"""Semantic checks on a merged spec (SPEC 8.4). Schema checks already ran when the spec parsed.

Each error says whether it comes from the baseline (a code defect, the job fails) or from an
overlay (the overlay writer can repair it).
"""

from __future__ import annotations

from typing import Literal, TypeVar

from synthgen.config import GenerationSettings
from synthgen.domain.errors import ExpressionError
from synthgen.spec.checks import GeneratorChecker
from synthgen.spec.correlation import as_matrix, is_valid, nearest_correlation
from synthgen.spec.dependencies import generation_order
from synthgen.spec.models import (
    ColumnOverride,
    CorrelationOverride,
    CorrelationSpec,
    GenerationSpec,
    ScenarioSpec,
    SpecError,
    TableSpec,
)

SHARE_TOLERANCE = 1e-9
Origin = Literal["baseline", "overlay"]


def validate_spec(spec: GenerationSpec, settings: GenerationSettings) -> list[SpecError]:
    """Every problem found; an empty list means the spec may run."""
    errors: list[SpecError] = []
    seen: list[str] = []
    checker = GeneratorChecker(spec, settings)
    for i, table in enumerate(spec.tables):
        errors += _table_errors(checker, table, f"/tables/{i}", seen, settings)
        seen.append(table.name)
    errors += _scenario_errors(spec, checker)
    return errors


def _error(origin: Origin, path: str, message: str, scenario: str | None = None) -> SpecError:
    return SpecError(origin=origin, scenario=scenario, path=path, message=message)


def _table_errors(
    checker: GeneratorChecker,
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
        problem = checker.problem(table, column, column.generator, earlier)
        if problem:
            errors.append(_error("baseline", f"{path}/columns/{j}/generator", problem))
    try:
        generation_order(table.columns)
    except ExpressionError as error:
        errors.append(_error("baseline", f"{path}/columns", str(error)))
    if table.correlation is not None:
        generators = {c.name: c.generator for c in table.columns}
        problem = checker.correlation_problem(table, table.correlation, generators)
        if problem:
            errors.append(_error("baseline", f"{path}/correlation", problem))
    return errors


def _scenario_errors(spec: GenerationSpec, checker: GeneratorChecker) -> list[SpecError]:
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
            (f"{path}/overrides/{j}", _override_problem(spec, checker, override))
            for j, override in enumerate(scenario.overrides)
        ]
        problems.append((f"{path}/overrides", _override_cycle(spec, scenario)))
        problems += [
            (f"{path}/correlations/{j}", _correlation_override_problem(spec, checker, scenario, j))
            for j in range(len(scenario.correlations))
        ]
        errors += [_error("overlay", at, p, scenario.name) for at, p in problems if p]
    return errors


def _correlation_override_problem(
    spec: GenerationSpec, checker: GeneratorChecker, scenario: ScenarioSpec, index: int
) -> str | None:
    """The scenario's rows use its own generators where it overrides them, else the baseline."""
    override = scenario.correlations[index]
    table = spec.table(override.table)
    if table is None:
        return f"unknown table {override.table}"
    generators = {c.name: c.generator for c in table.columns} | {
        o.column: o.generator for o in scenario.overrides if o.table == table.name
    }
    return checker.correlation_problem(table, override, generators)


def repair_correlations(spec: GenerationSpec) -> GenerationSpec:
    """The spec with every correlation matrix replaced by the nearest valid one (SPEC 8.4)."""
    tables = [
        t.model_copy(update={"correlation": _repaired(t.correlation)}) if t.correlation else t
        for t in spec.tables
    ]
    scenarios = [
        s.model_copy(update={"correlations": [_repaired(c) for c in s.correlations]})
        for s in spec.scenarios
    ]
    return spec.model_copy(update={"tables": tables, "scenarios": scenarios})


Matrix = TypeVar("Matrix", CorrelationSpec, CorrelationOverride)


def _repaired(correlation: Matrix) -> Matrix:
    matrix = as_matrix(correlation.spearman)
    if is_valid(matrix):
        return correlation
    return correlation.model_copy(update={"spearman": nearest_correlation(matrix).tolist()})


def _override_cycle(spec: GenerationSpec, scenario: ScenarioSpec) -> str | None:
    """Overrides of one table may read each other, but not in a cycle."""
    for table in spec.tables:
        found = {o.column: o.generator for o in scenario.overrides if o.table == table.name}
        replaced = [
            c.model_copy(update={"generator": found[c.name]})
            for c in table.columns
            if c.name in found
        ]
        given = {c.name for c in table.columns} - set(found)
        try:
            generation_order(replaced, given, overlay=True)
        except ExpressionError as error:
            return str(error)
    return None


def _override_problem(
    spec: GenerationSpec, checker: GeneratorChecker, override: ColumnOverride
) -> str | None:
    """Why an override cannot apply, or None when it can."""
    missing = _missing(spec, override.table, override.column)
    table = spec.table(override.table)
    column = table.column(override.column) if table else None
    if missing or table is None or column is None:
        return missing
    order = [t.name for t in spec.tables]
    return checker.problem(table, column, override.generator, order[: order.index(table.name)])


def _missing(spec: GenerationSpec, table: str, column: str) -> str | None:
    found = spec.table(table)
    if found is None:
        return f"unknown table {table}"
    if found.column(column) is None:
        return f"unknown column {table}.{column}"
    return None
