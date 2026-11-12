"""Checks on the event shapes a scenario imposes: cardinality overrides and sequences."""

from __future__ import annotations

from synthgen.config import GenerationSettings
from synthgen.spec.checks import GeneratorChecker
from synthgen.spec.models import ColumnType, GenerationSpec, ScenarioSpec, TableSpec
from synthgen.spec.shapes import SequenceSpec

Problem = tuple[str, str]
"""(path below the scenario, message)."""


def ancestors(spec: GenerationSpec, table: TableSpec) -> list[str]:
    """Tables above `table` along driving keys, nearest first."""
    found: list[str] = []
    current: TableSpec | None = table
    while current is not None and current.row_rule.kind == "per_parent":
        parent = current.row_rule.parent_table
        if parent in found:
            break
        found.append(parent)
        current = spec.table(parent)
    return found


def shape_problems(
    spec: GenerationSpec,
    scenario: ScenarioSpec,
    checker: GeneratorChecker,
    settings: GenerationSettings,
) -> list[Problem]:
    """Every problem with the scenario's cardinality overrides and sequences."""
    problems: list[Problem] = []
    for j, override in enumerate(scenario.cardinality):
        found = _child_problem(spec, scenario, override.table)
        if found:
            problems.append((f"cardinality/{j}", found))
    tables = [q.table for q in scenario.sequences]
    for j, sequence in enumerate(scenario.sequences):
        found = _child_problem(spec, scenario, sequence.table)
        if not found and tables.count(sequence.table) > 1:
            found = f"one sequence per table; {sequence.table} has {tables.count(sequence.table)}"
        if not found:
            found = _sequence_problem(spec, sequence, checker, settings)
        if found:
            problems.append((f"sequences/{j}", found))
    return problems


def _child_problem(spec: GenerationSpec, scenario: ScenarioSpec, name: str) -> str | None:
    """A shaped table must hang, through driving keys, below the scenario's entity."""
    table = spec.table(name)
    if table is None:
        return f"unknown table {name}"
    if table.row_rule.kind != "per_parent":
        return f"{name} is not sized by a parent table, so its children per parent cannot change"
    if scenario.entity not in ancestors(spec, table):
        return f"{name} does not hang below the scenario entity {scenario.entity}"
    return None


def _sequence_problem(
    spec: GenerationSpec,
    sequence: SequenceSpec,
    checker: GeneratorChecker,
    settings: GenerationSettings,
) -> str | None:
    table = spec.table(sequence.table)
    if table is None or table.row_rule.kind != "per_parent":
        return f"unknown child table {sequence.table}"
    if sequence.entity_key != table.row_rule.child_column:
        return f"entity_key must be {table.row_rule.child_column}, the key to the parent"
    time = table.column(sequence.time_column)
    if time is None or time.type not in (ColumnType.DATE, ColumnType.DATETIME):
        return f"time_column {sequence.time_column} is not a date or datetime column"
    if sequence.events > settings.max_sequence_events:
        return f"a sequence has at most {settings.max_sequence_events} events"
    return _steps_problem(spec, table, sequence, checker)


def _steps_problem(
    spec: GenerationSpec, table: TableSpec, sequence: SequenceSpec, checker: GeneratorChecker
) -> str | None:
    earlier = [t.name for t in spec.tables[: spec.tables.index(table)]]
    for setting in (c for step in sequence.steps for c in step.columns):
        column = table.column(setting.column)
        if column is None:
            return f"unknown column {table.name}.{setting.column}"
        if setting.column in (sequence.time_column, sequence.entity_key) or column.primary_key:
            return f"a step cannot set {setting.column}; the sequence controls it"
        found = checker.problem(table, column, setting.generator, earlier)
        if found:
            return found
    return None
