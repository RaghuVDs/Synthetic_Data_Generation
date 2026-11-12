"""Generate every table of a spec in order and write Parquet (SPEC 8.5).

M2 writes one chunk per table; M5 adds chunking and workers. Each table draws from its own
stream, `SeedSequence(seed, spawn_key=(table, chunk))`, so a rerun gives identical bytes.
"""

from __future__ import annotations

import logging
import shutil
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from synthgen.config import GenerationSettings
from synthgen.domain.errors import EngineError
from synthgen.engine.columns import TableDraw, apply_overlays, draw_columns
from synthgen.engine.generators import GenContext, SemanticPools, Values
from synthgen.engine.keys import plan_children
from synthgen.engine.scenarios import assign_scenarios, categories
from synthgen.engine.writer import (
    DATA_DIR,
    ColumnSchema,
    Manifest,
    ScenarioCount,
    TableManifest,
    read_table,
    write_manifest,
    write_part,
)
from synthgen.spec.dependencies import parent_reads
from synthgen.spec.evaluator import Column
from synthgen.spec.models import (
    EXPR_TYPES,
    TYPICAL,
    ColumnSpec,
    ColumnType,
    GenerationSpec,
    TableSpec,
)

logger = logging.getLogger(__name__)
SCENARIO_COLUMN = "_scenario"
CATEGORY_COLUMN = "_category"
ARROW_TYPES = {
    ColumnType.INTEGER: pa.int64(),
    ColumnType.FLOAT: pa.float64(),
    ColumnType.BOOLEAN: pa.int64(),
    ColumnType.TEXT: pa.string(),
    ColumnType.DATE: pa.string(),
    ColumnType.DATETIME: pa.string(),
}


@dataclass(frozen=True)
class RunOptions:
    """Dry runs cap root tables and size children by their profiled mean, not the target."""

    max_root_rows: int | None = None

    @property
    def is_sample(self) -> bool:
        """True for a dry run."""
        return self.max_root_rows is not None


def generate_dataset(
    spec: GenerationSpec, out_dir: Path, options: RunOptions, settings: GenerationSettings
) -> Manifest:
    """Write all tables, then the manifest. Any earlier output in `out_dir` is replaced."""
    shutil.rmtree(out_dir / DATA_DIR, ignore_errors=True)
    written: dict[str, TableManifest] = {}
    pools = SemanticPools(
        spec.seed, settings.semantic_pool_size, settings.faker_locale, settings.birth_dates
    )
    for index, table in enumerate(spec.tables):
        rng = np.random.default_rng(np.random.SeedSequence(spec.seed, spawn_key=(index, 0)))
        reader = _Reader(out_dir, written)
        ctx = GenContext(
            row_offset=0, seed=spec.seed, parent_keys=reader.column, pools=pools, frame={}
        )
        data = _generate_table(spec, table, TableDraw(table, rng, ctx), reader, options)
        file = write_part(out_dir, table.name, 0, data, spec.output.compression)
        written[table.name] = _table_manifest(table.name, data, [file])
        logger.info("generated %s rows of %s", data.num_rows, table.name)
    manifest = Manifest(seed=spec.seed, spec_hash=spec.spec_hash(), tables=list(written.values()))
    write_manifest(out_dir, manifest)
    return manifest


class _Reader:
    """Reads columns of tables already written in this run."""

    def __init__(self, out_dir: Path, written: dict[str, TableManifest]) -> None:
        self._out_dir = out_dir
        self._written = written
        self._cache: dict[tuple[str, str], Values] = {}

    def column(self, table: str, column: str) -> Values:
        key = (table, column)
        if key not in self._cache:
            entry = self._written.get(table)
            if entry is None:
                raise EngineError(f"{table} must be generated before it is read")
            data = read_table(self._out_dir, entry, [column]).column(column)
            self._cache[key] = np.asarray(data.to_numpy(zero_copy_only=False))
        return self._cache[key]


def _generate_table(
    spec: GenerationSpec,
    table: TableSpec,
    draw: TableDraw,
    reader: _Reader,
    options: RunOptions,
) -> pa.Table:
    n, parent_rows = _row_plan(spec, table, draw, reader, options)
    inherited = np.full(n, TYPICAL, dtype=object)
    fixed: dict[str, Values] = {}
    rule = table.row_rule
    if rule.kind == "per_parent" and parent_rows is not None:
        fixed[rule.child_column] = reader.column(rule.parent_table, rule.parent_column)[parent_rows]
        inherited = reader.column(rule.parent_table, SCENARIO_COLUMN)[parent_rows].astype(object)
        frame = _parent_frame(spec, table, reader, parent_rows)
        draw = replace(draw, ctx=replace(draw.ctx, frame=frame))
    values = draw_columns(draw, n, fixed)
    own = [s for s in spec.scenarios if s.entity == table.name]
    key = table.primary_key
    keys = values[key.name] if key else np.arange(n)
    assigned = assign_scenarios(keys, inherited, own, spec.seed)
    apply_overlays(draw, spec.scenarios, values, assigned)
    columns = [pa.array(_cast(values[c.name], c), type=ARROW_TYPES[c.type]) for c in table.columns]
    columns += [
        pa.array(assigned, type=pa.string()),
        pa.array(categories(assigned, spec.scenarios), type=pa.string()),
    ]
    return pa.table(
        columns, names=[c.name for c in table.columns] + [SCENARIO_COLUMN, CATEGORY_COLUMN]
    )


def _parent_frame(
    spec: GenerationSpec, table: TableSpec, reader: _Reader, parent_rows: NDArray[np.int64]
) -> dict[str, Column]:
    """Driving-parent columns (`parent.column`) that this table's generators or overlays read."""
    gens = [c.generator for c in table.columns] + [
        o.generator for s in spec.scenarios for o in s.overrides if o.table == table.name
    ]
    frame: dict[str, Column] = {}
    for name in sorted(set().union(*(parent_reads(g) for g in gens))):
        parent_name, column_name = name.split(".", 1)
        parent = spec.table(parent_name)
        column = parent.column(column_name) if parent else None
        if column is None:
            raise EngineError(f"{table.name} reads {name}, which is not a parent column")
        values = reader.column(parent_name, column_name)[parent_rows]
        frame[name] = (EXPR_TYPES[column.type], values)
    return frame


def _row_plan(
    spec: GenerationSpec, table: TableSpec, draw: TableDraw, reader: _Reader, options: RunOptions
) -> tuple[int, Any]:
    """Row count, and for child tables the parent row index of each child row."""
    rule = table.row_rule
    if rule.kind == "fixed":
        cap = options.max_root_rows
        return (min(rule.rows, cap) if cap else rule.rows), None
    parent_scenarios = reader.column(rule.parent_table, SCENARIO_COLUMN).astype(object)
    counts = plan_children(
        rule,
        table.name,
        parent_scenarios,
        spec.scenarios,
        draw.rng,
        use_target=not options.is_sample,
    )
    parents = parent_scenarios.size
    parent_rows = np.repeat(np.arange(parents), counts)
    return int(parent_rows.size), parent_rows


def _cast(values: Values, column: ColumnSpec) -> list[Any]:  # Any: Arrow takes mixed Python values
    """Python values Arrow can type: whole numbers for integer columns, text for text columns."""
    out: list[Any] = []
    for v in values.tolist():
        if v is None:
            out.append(None)
        elif column.type in (ColumnType.INTEGER, ColumnType.BOOLEAN):
            out.append(round(float(v)))
        elif column.type is ColumnType.FLOAT:
            out.append(float(v))
        else:
            out.append(str(v))
    return out


def _table_manifest(name: str, data: pa.Table, files: list[Any]) -> TableManifest:
    counts = Counter(data.column(SCENARIO_COLUMN).to_pylist())
    return TableManifest(
        name=name,
        rows=data.num_rows,
        files=files,
        columns=[ColumnSchema(name=f.name, type=str(f.type)) for f in data.schema],
        scenario_counts=[ScenarioCount(scenario=s, rows=c) for s, c in sorted(counts.items())],
    )
