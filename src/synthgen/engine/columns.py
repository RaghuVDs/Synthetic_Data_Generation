"""Draw a table's columns in dependency order, then apply scenario overlays to their rows.

Columns that read other columns (`derived`, `conditional`) come after what they read. When an
overlay replaces a column, baseline columns computed from it are recomputed on the overlay's rows,
so a total stays the sum of its parts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from synthgen.domain.errors import EngineError
from synthgen.engine.copula import correlated_uniforms
from synthgen.engine.generators import GenContext, Values, generate
from synthgen.engine.generators.basic import from_uniform
from synthgen.engine.scenarios import ObjectArray
from synthgen.engine.sequences import SequenceDraw, apply_sequence
from synthgen.spec.dependencies import downstream, generation_order, own_reads, reads
from synthgen.spec.distributions import DistributionGen
from synthgen.spec.evaluator import Column
from synthgen.spec.generators import EmpiricalGen, Generator
from synthgen.spec.models import (
    EXPR_TYPES,
    ColumnSpec,
    CorrelationSpec,
    ScenarioSpec,
    TableSpec,
)

MAX_UNIQUE_REDRAWS = 20


@dataclass(frozen=True)
class TableDraw:
    """One table's rows being drawn: its spec, its random stream and the generator context."""

    table: TableSpec
    rng: np.random.Generator
    ctx: GenContext


def draw_columns(draw: TableDraw, n: int, fixed: Mapping[str, Values]) -> dict[str, Values]:
    """Every column of the table; `fixed` columns (the driving key) are taken as given."""
    table, rng, ctx = draw.table, draw.rng, draw.ctx
    values: dict[str, Values] = {}
    frame: dict[str, Column] = dict(ctx.frame)
    copula = _copula_draws(table.correlation, {c.name: c.generator for c in table.columns}, n, rng)
    for column in generation_order(table.columns):
        drawn = fixed.get(column.name, copula.get(column.name))
        if drawn is None:
            needed = {k: frame[k] for k in reads(column.generator)}
            drawn = _draw(column, n, rng, replace(ctx, frame=needed))
        values[column.name] = _with_nulls(drawn, column, rng)
        frame[column.name] = (EXPR_TYPES[column.type], values[column.name])
    return values


def _copula_draws(
    correlation: CorrelationSpec | None,
    generators: Mapping[str, Generator],
    n: int,
    rng: np.random.Generator,
) -> dict[str, Values]:
    """Values of the correlated columns, drawn together; each keeps its own marginal."""
    if correlation is None:
        return {}
    uniforms = correlated_uniforms(correlation.spearman, n, rng)
    drawn: dict[str, Values] = {}
    for index, name in enumerate(correlation.columns):
        gen = generators[name]
        if not isinstance(gen, EmpiricalGen | DistributionGen):
            raise EngineError(f"correlated column {name} has a {gen.type} generator")
        drawn[name] = from_uniform(gen, uniforms[:, index])
    return drawn


def _draw(column: ColumnSpec, n: int, rng: np.random.Generator, ctx: GenContext) -> Values:
    values = generate(column.generator, n, rng, ctx)
    if not column.unique:
        return values
    for _ in range(MAX_UNIQUE_REDRAWS):
        _, first = np.unique(values.astype(str), return_index=True)
        repeats = np.setdiff1d(np.arange(n), first)
        if repeats.size == 0:
            return values
        values[repeats] = generate(column.generator, repeats.size, rng, ctx.subset(repeats))
    raise EngineError(f"could not make {column.name} unique")


def _with_nulls(values: Values, column: ColumnSpec, rng: np.random.Generator) -> Values:
    if column.null_rate <= 0 or column.primary_key:
        return values
    out = values.astype(object)
    out[rng.random(values.size) < column.null_rate] = None
    return out


def apply_overlays(
    draw: TableDraw,
    scenarios: Sequence[ScenarioSpec],
    values: dict[str, Values],
    assigned: ObjectArray,
) -> None:
    """Scenario rows get the overlay's generators, correlations, sequences and labels.

    Order: overridden columns, then correlated columns, then sequences; afterwards every baseline
    column computed from a changed column is recomputed on the scenario's rows.
    """
    for scenario in scenarios:
        rows = np.flatnonzero(assigned == scenario.name)
        if rows.size:
            _apply_scenario(draw, scenario, values, rows)


def _apply_scenario(
    draw: TableDraw, scenario: ScenarioSpec, values: dict[str, Values], rows: NDArray[np.int64]
) -> None:
    table, rng = draw.table, draw.rng
    overrides = {o.column: o.generator for o in scenario.overrides if o.table == table.name}
    correlated = _scenario_copula(scenario, table, overrides, rows.size, rng)
    replaced = [
        c.model_copy(update={"generator": overrides[c.name]})
        for c in table.columns
        if c.name in overrides and c.name not in correlated
    ]
    given = {c.name for c in table.columns} - set(overrides)
    for column in generation_order(replaced, given | set(correlated), overlay=True):
        _redraw(draw, column.generator, column.name, values, rows)
    for name, drawn in correlated.items():
        values[name] = _set_rows(values[name], rows, drawn)
    changed = set(overrides) | set(correlated)
    for sequence in (q for q in scenario.sequences if q.table == table.name):
        how = SequenceDraw(
            time_type=EXPR_TYPES[_column(table, sequence.time_column).type],
            rng=rng,
            draw=lambda gen, members: _draw_rows(draw, gen, values, members),
        )
        changed |= apply_sequence(sequence, values, rows, how)
    for column in downstream(table.columns, changed):
        _redraw(draw, column.generator, column.name, values, rows)
    for label in (lb for lb in scenario.labels if lb.table == table.name):
        values[label.column] = _set_rows(values[label.column], rows, label.value)


def _column(table: TableSpec, name: str) -> ColumnSpec:
    found = table.column(name)
    if found is None:
        raise EngineError(f"{table.name} has no column {name}")
    return found


def _draw_rows(
    draw: TableDraw, gen: Generator, values: Mapping[str, Values], rows: NDArray[np.int64]
) -> Values:
    """A generator's values for some rows, reading the current values of the columns it needs."""
    types = {c.name: EXPR_TYPES[c.type] for c in draw.table.columns}
    frame = {**draw.ctx.frame, **{k: (types[k], values[k]) for k in own_reads(gen)}}
    return generate(gen, rows.size, draw.rng, replace(draw.ctx, frame=frame).subset(rows))


def _redraw(
    draw: TableDraw, gen: Generator, name: str, values: dict[str, Values], rows: NDArray[np.int64]
) -> None:
    values[name] = _set_rows(values[name], rows, _draw_rows(draw, gen, values, rows))


def _set_rows(values: Values, rows: NDArray[np.int64], new: object) -> Values:
    out = values.astype(object)
    out[rows] = new
    return out


def _scenario_copula(
    scenario: ScenarioSpec,
    table: TableSpec,
    overrides: Mapping[str, Generator],
    n: int,
    rng: np.random.Generator,
) -> dict[str, Values]:
    """Columns a scenario correlates on this table: its own generators where it overrides them."""
    generators = {c.name: c.generator for c in table.columns} | dict(overrides)
    drawn: dict[str, Values] = {}
    for correlation in (c for c in scenario.correlations if c.table == table.name):
        drawn |= _copula_draws(correlation, generators, n, rng)
    return drawn
