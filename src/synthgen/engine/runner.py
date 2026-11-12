"""Generate every table of a spec in order and write Parquet (SPEC 8.5).

Tables run in dependency order; each table is planned into chunks once its parents are written,
and every chunk is generated from its own stream, so a rerun gives identical bytes.
"""

from __future__ import annotations

import logging
import shutil
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from synthgen.config import GenerationSettings
from synthgen.engine.chunk import ChunkJob, ChunkResult, run_chunk
from synthgen.engine.planning import TablePlan, plan_child, plan_root
from synthgen.engine.reader import WrittenTables
from synthgen.engine.schema import SCENARIO_COLUMN, output_schema
from synthgen.engine.writer import (
    DATA_DIR,
    Manifest,
    ScenarioCount,
    TableManifest,
    write_manifest,
)
from synthgen.spec.models import GenerationSpec, TableSpec

logger = logging.getLogger(__name__)


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
    for index, table in enumerate(spec.tables):
        plan = plan_table(spec, index, WrittenTables(out_dir, written), options, settings)
        results = [
            run_chunk(ChunkJob(spec, chunk, out_dir, dict(written), settings))
            for chunk in plan.chunks
        ]
        written[table.name] = table_manifest(table, results)
        logger.info("generated %s rows of %s in %s chunks", plan.rows, table.name, len(results))
    manifest = Manifest(seed=spec.seed, spec_hash=spec.spec_hash(), tables=list(written.values()))
    write_manifest(out_dir, manifest)
    return manifest


def plan_table(
    spec: GenerationSpec,
    index: int,
    reader: WrittenTables,
    options: RunOptions,
    settings: GenerationSettings,
) -> TablePlan:
    """Chunks of one table; a child table reads its parent's scenarios to plan its rows."""
    rule = spec.tables[index].row_rule
    if rule.kind == "fixed":
        cap = options.max_root_rows
        return plan_root(spec, index, min(rule.rows, cap) if cap else rule.rows, settings)
    parent_scenarios = reader.column(rule.parent_table, SCENARIO_COLUMN).astype(object)
    return plan_child(spec, index, parent_scenarios, settings, use_target=not options.is_sample)


def table_manifest(table: TableSpec, results: Sequence[ChunkResult]) -> TableManifest:
    """A table's manifest entry from its chunks, in part order."""
    counts: Counter[str] = Counter()
    for result in results:
        counts.update(result.scenarios)
    return TableManifest(
        name=table.name,
        rows=sum(r.file.rows for r in results),
        files=[r.file for r in results],
        columns=output_schema(table),
        scenario_counts=[ScenarioCount(scenario=s, rows=c) for s, c in sorted(counts.items())],
    )
