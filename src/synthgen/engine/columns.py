"""Draw a table's columns in dependency order, then apply scenario overlays to their rows.

Columns that read other columns (`derived`, `conditional`) come after what they read. When an
overlay replaces a column, baseline columns computed from it are recomputed on the overlay's rows,
so a total stays the sum of its parts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from synthgen.domain.errors import EngineError
from synthgen.engine.generators import GenContext, Values, generate
from synthgen.engine.scenarios import ObjectArray
from synthgen.spec.dependencies import downstream, generation_order, own_reads, reads
from synthgen.spec.evaluator import Column
from synthgen.spec.models import EXPR_TYPES, ColumnSpec, ScenarioSpec, TableSpec

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
    for column in generation_order(table.columns):
        drawn = fixed.get(column.name)
        if drawn is None:
            needed = {k: frame[k] for k in reads(column.generator)}
            drawn = _draw(column, n, rng, replace(ctx, frame=needed))
        values[column.name] = _with_nulls(drawn, column, rng)
        frame[column.name] = (EXPR_TYPES[column.type], values[column.name])
    return values


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
    """Scenario rows get the overlay's generators and label values; other rows keep the baseline."""
    table, rng, ctx = draw.table, draw.rng, draw.ctx
    types = {c.name: EXPR_TYPES[c.type] for c in table.columns}
    for scenario in scenarios:
        rows = np.flatnonzero(assigned == scenario.name)
        if rows.size == 0:
            continue
        overrides = {o.column: o.generator for o in scenario.overrides if o.table == table.name}
        replaced = [
            c.model_copy(update={"generator": overrides[c.name]})
            for c in table.columns
            if c.name in overrides
        ]
        given = {c.name for c in table.columns} - set(overrides)
        for column in [
            *generation_order(replaced, given, overlay=True),
            *downstream(table.columns, set(overrides)),
        ]:
            frame = {**ctx.frame, **{k: (types[k], values[k]) for k in own_reads(column.generator)}}
            updated = values[column.name].astype(object)
            updated[rows] = generate(
                column.generator, rows.size, rng, replace(ctx, frame=frame).subset(rows)
            )
            values[column.name] = updated
        for label in (lb for lb in scenario.labels if lb.table == table.name):
            labelled = values[label.column].astype(object)
            labelled[rows] = label.value
            values[label.column] = labelled
