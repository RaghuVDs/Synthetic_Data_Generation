"""Generate every table of a spec in order and write Parquet (SPEC 8.5).

M2 writes one chunk per table; M5 adds chunking and workers. Each table draws from its own
stream, `SeedSequence(seed, spawn_key=(table, chunk))`, so a rerun gives identical bytes.
"""

from __future__ import annotations

import logging
import shutil
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from synthgen.domain.errors import EngineError
from synthgen.engine.generators import GenContext, Values, generate
from synthgen.engine.keys import children_per_parent
from synthgen.engine.scenarios import ObjectArray, assign_scenarios, categories
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
from synthgen.spec.generators import Generator
from synthgen.spec.models import (
    TYPICAL,
    ColumnSpec,
    ColumnType,
    GenerationSpec,
    ScenarioSpec,
    TableSpec,
)

logger = logging.getLogger(__name__)
SCENARIO_COLUMN = "_scenario"
CATEGORY_COLUMN = "_category"
MAX_UNIQUE_REDRAWS = 20
_ARROW = {
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


def generate_dataset(spec: GenerationSpec, out_dir: Path, options: RunOptions) -> Manifest:
    """Write all tables, then the manifest. Any earlier output in `out_dir` is replaced."""
    shutil.rmtree(out_dir / DATA_DIR, ignore_errors=True)
    written: dict[str, TableManifest] = {}
    for index, table in enumerate(spec.tables):
        rng = np.random.default_rng(np.random.SeedSequence(spec.seed, spawn_key=(index, 0)))
        data = _generate_table(spec, table, rng, _Reader(out_dir, written), options)
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
    rng: np.random.Generator,
    reader: _Reader,
    options: RunOptions,
) -> pa.Table:
    n, parent_rows = _row_plan(table, rng, reader, options)
    inherited = np.full(n, TYPICAL, dtype=object)
    values: dict[str, Values] = {}
    rule = table.row_rule
    ctx = GenContext(row_offset=0, parent_keys=reader.column)
    for column in table.columns:
        if (
            rule.kind == "per_parent"
            and column.name == rule.child_column
            and parent_rows is not None
        ):
            values[column.name] = reader.column(rule.parent_table, rule.parent_column)[parent_rows]
            inherited = reader.column(rule.parent_table, SCENARIO_COLUMN)[parent_rows].astype(
                object
            )
        else:
            values[column.name] = _draw(column, n, rng, ctx)
        values[column.name] = _with_nulls(values[column.name], column, rng)
    own = [s for s in spec.scenarios if s.entity == table.name]
    key = table.primary_key
    keys = values[key.name] if key else np.arange(n)
    assigned = assign_scenarios(keys, inherited, own, spec.seed)
    _apply_overlays(
        spec.scenarios, table.name, values, assigned, lambda gen, k: generate(gen, k, rng, ctx)
    )
    columns = [pa.array(_cast(values[c.name], c), type=_ARROW[c.type]) for c in table.columns]
    columns += [
        pa.array(assigned, type=pa.string()),
        pa.array(categories(assigned, spec.scenarios), type=pa.string()),
    ]
    return pa.table(
        columns, names=[c.name for c in table.columns] + [SCENARIO_COLUMN, CATEGORY_COLUMN]
    )


def _row_plan(
    table: TableSpec, rng: np.random.Generator, reader: _Reader, options: RunOptions
) -> tuple[int, Any]:
    """Row count, and for child tables the parent row index of each child row."""
    rule = table.row_rule
    if rule.kind == "fixed":
        cap = options.max_root_rows
        return (min(rule.rows, cap) if cap else rule.rows), None
    parents = reader.column(rule.parent_table, rule.parent_column).size
    counts = children_per_parent(rule, parents, rng, use_target=not options.is_sample)
    parent_rows = np.repeat(np.arange(parents), counts)
    return int(parent_rows.size), parent_rows


def _draw(column: ColumnSpec, n: int, rng: np.random.Generator, ctx: GenContext) -> Values:
    values = generate(column.generator, n, rng, ctx)
    if not column.unique:
        return values
    for _ in range(MAX_UNIQUE_REDRAWS):
        _, first = np.unique(values.astype(str), return_index=True)
        repeats = np.setdiff1d(np.arange(n), first)
        if repeats.size == 0:
            return values
        values[repeats] = generate(column.generator, repeats.size, rng, ctx)
    raise EngineError(f"could not make {column.name} unique")


def _with_nulls(values: Values, column: ColumnSpec, rng: np.random.Generator) -> Values:
    if column.null_rate <= 0 or column.primary_key:
        return values
    out = values.astype(object)
    out[rng.random(values.size) < column.null_rate] = None
    return out


def _apply_overlays(
    scenarios: Sequence[ScenarioSpec],
    table: str,
    values: dict[str, Values],
    assigned: ObjectArray,
    draw: Callable[[Generator, int], Values],
) -> None:
    """Scenario rows get the overlay's generators and label values; other rows keep the baseline."""
    for scenario in scenarios:
        rows = np.flatnonzero(assigned == scenario.name)
        if rows.size == 0:
            continue
        for override in (o for o in scenario.overrides if o.table == table):
            column = values[override.column].astype(object)
            column[rows] = draw(override.generator, rows.size)
            values[override.column] = column
        for label in (lb for lb in scenario.labels if lb.table == table):
            column = values[label.column].astype(object)
            column[rows] = label.value
            values[label.column] = column


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
