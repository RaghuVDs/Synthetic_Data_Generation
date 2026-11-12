"""Generate one chunk and write its Parquet part. Self-contained, so a worker process can run it.

Inside a chunk, in order (SPEC 8.5): driving keys and parent attributes from the parent's
Parquet, columns (copula, generators, constraints, nulls), scenario assignment, overlays and
sequences, defects, then one zstd Parquet file.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from synthgen.config import GenerationSettings
from synthgen.domain.errors import EngineError
from synthgen.engine.columns import TableDraw, apply_overlays, draw_columns
from synthgen.engine.defects import apply_defects
from synthgen.engine.generators import GenContext, SemanticPools, Values
from synthgen.engine.planning import ChunkPlan, own_scenarios, stream
from synthgen.engine.reader import WrittenTables
from synthgen.engine.scenarios import assign_scenarios, categories
from synthgen.engine.schema import CATEGORY_COLUMN, DEFECT_COLUMN, SCENARIO_COLUMN, to_arrow
from synthgen.engine.writer import ManifestFile, ScenarioCount, TableManifest, write_part
from synthgen.spec.dependencies import constraint_reads, parent_reads
from synthgen.spec.evaluator import Column
from synthgen.spec.models import EXPR_TYPES, TYPICAL, GenerationSpec, TableSpec

ColumnReader = Callable[[str], Values]
"""A driving-parent column, one value per row of the chunk."""


@dataclass(frozen=True)
class ChunkJob:
    """Everything one chunk needs; picklable for a process pool."""

    spec: GenerationSpec
    plan: ChunkPlan
    out_dir: Path
    written: Mapping[str, TableManifest]
    settings: GenerationSettings


def run_chunk(job: ChunkJob) -> ManifestFile:
    """Generate the chunk from its own stream, write it as part `plan.index`, and describe it."""
    spec, plan, settings = job.spec, job.plan, job.settings
    table = spec.tables[plan.table_index]
    reader = WrittenTables(job.out_dir, job.written)
    pools = SemanticPools(
        spec.seed, settings.semantic_pool_size, settings.faker_locale, settings.birth_dates
    )
    ctx = GenContext(plan.row_offset, spec.seed, reader.column, pools, frame={})
    draw = TableDraw(table, stream(spec.seed, plan.table_index, plan.index + 1), ctx)
    fixed: dict[str, Values] = {}
    inherited = np.full(plan.rows, TYPICAL, dtype=object)
    rule = table.row_rule
    if rule.kind == "per_parent" and plan.parent_counts is not None:
        stop = plan.parent_start + plan.parent_counts.size
        local = np.repeat(np.arange(plan.parent_counts.size), plan.parent_counts)

        def parent(column: str) -> Values:
            return reader.rows(rule.parent_table, column, plan.parent_start, stop)[local]

        fixed[rule.child_column] = parent(rule.parent_column)
        inherited = parent(SCENARIO_COLUMN).astype(object)
        frame = {name: _parent_column(spec, name, parent) for name in _parent_names(spec, table)}
        draw = replace(draw, ctx=replace(ctx, frame=frame))
    values = draw_columns(draw, plan.rows, fixed)
    key = table.primary_key
    keys = values[key.name] if key else plan.row_offset + np.arange(plan.rows)
    assigned = assign_scenarios(keys, inherited, own_scenarios(spec, table), spec.seed, plan.quotas)
    apply_overlays(draw, spec.scenarios, values, assigned)
    labels = {
        SCENARIO_COLUMN: assigned,
        CATEGORY_COLUMN: categories(assigned, spec.scenarios),
        DEFECT_COLUMN: apply_defects(draw, spec.scenarios, values, assigned),
    }
    data = to_arrow(table, values, labels)
    path = write_part(job.out_dir, table.name, plan.index, data, spec.output.compression)
    counts = Counter(assigned.tolist())
    return ManifestFile(
        path=path.relative_to(job.out_dir).as_posix(),
        rows=data.num_rows,
        scenarios=[ScenarioCount(scenario=s, rows=c) for s, c in sorted(counts.items())],
    )


def _parent_names(spec: GenerationSpec, table: TableSpec) -> list[str]:
    """Driving-parent columns (`parent.column`) the table's specs read.

    Generators, overrides, sequence steps and constraints may all read the parent row.
    """
    gens = [c.generator for c in table.columns]
    for scenario in spec.scenarios:
        gens += [o.generator for o in scenario.overrides if o.table == table.name]
        gens += [
            c.generator
            for q in scenario.sequences
            if q.table == table.name
            for step in q.steps
            for c in step.columns
        ]
    names = set().union(*(parent_reads(g) for g in gens))
    names |= {n for c in table.constraints for n in constraint_reads(c) if "." in n}
    return sorted(names)


def _parent_column(spec: GenerationSpec, name: str, read: ColumnReader) -> Column:
    parent_name, column_name = name.split(".", 1)
    parent = spec.table(parent_name)
    column = parent.column(column_name) if parent else None
    if column is None:
        raise EngineError(f"{name} is not a parent column")
    return EXPR_TYPES[column.type], read(column_name)
