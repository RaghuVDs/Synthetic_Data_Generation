"""Generate every table of a spec in order and write Parquet (SPEC 8.5, 10).

Tables run in dependency order; each table is planned into chunks once its parents are written.
Chunks run in this process or in a pool of workers; each draws from its own stream, so the
output is byte-identical for any number of workers. The manifest is rewritten after every chunk,
and a restart with the same spec, options and settings skips the chunks it lists.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from synthgen.config import GenerationSettings
from synthgen.engine.chunk import ChunkJob, run_chunk
from synthgen.engine.executor import ChunkRunner, chunk_pool, run_jobs
from synthgen.engine.planning import TablePlan, plan_child, plan_root
from synthgen.engine.reader import WrittenTables
from synthgen.engine.schema import SCENARIO_COLUMN, output_schema
from synthgen.engine.writer import (
    DATA_DIR,
    MANIFEST_FILE,
    Manifest,
    ManifestFile,
    ScenarioCount,
    TableManifest,
    read_manifest,
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


def generate_dataset(  # noqa: PLR0913 - the run's inputs; `workers` and `run` are keyword-only
    spec: GenerationSpec,
    out_dir: Path,
    options: RunOptions,
    settings: GenerationSettings,
    *,
    workers: int = 1,
    run: ChunkRunner = run_chunk,
) -> Manifest:
    """Write all tables, then the complete manifest; reuse finished chunks of the same run.

    Output from another spec, options or settings in `out_dir` is replaced.
    """
    state = _Run(spec, out_dir, options, settings, run)
    with chunk_pool(workers) as pool:
        for index in range(len(spec.tables)):
            state.generate_table(index, pool)
    manifest = state.progress.model_copy(update={"complete": True})
    write_manifest(out_dir, manifest)
    return manifest


class _Run:
    """One generation run: what earlier attempts finished, and the manifest written so far."""

    def __init__(
        self,
        spec: GenerationSpec,
        out_dir: Path,
        options: RunOptions,
        settings: GenerationSettings,
        run: ChunkRunner,
    ) -> None:
        """Load what an earlier attempt of the same run finished; clear anything else."""
        run_hash = _run_hash(spec, options, settings)
        self.spec = spec
        self.out_dir = out_dir
        self.options = options
        self.settings = settings
        self.run = run
        self.done = _finished_chunks(out_dir, run_hash)
        self.progress = Manifest(
            seed=spec.seed, spec_hash=spec.spec_hash(), run_hash=run_hash, complete=False, tables=[]
        )

    def generate_table(self, index: int, pool: ProcessPoolExecutor | None) -> None:
        """Plan the table, run its unfinished chunks, and keep the manifest current."""
        table = self.spec.tables[index]
        written = {t.name: t for t in self.progress.tables}
        plan = plan_table(
            self.spec, index, WrittenTables(self.out_dir, written), self.options, self.settings
        )
        earlier = self.done.get(table.name, {})
        files = {i: f for i, f in earlier.items() if i < len(plan.chunks)}
        jobs = [
            ChunkJob(self.spec, chunk, self.out_dir, written, self.settings)
            for chunk in plan.chunks
            if chunk.index not in files
        ]

        def record(job: ChunkJob, file: ManifestFile) -> None:
            files[job.plan.index] = file
            self._save(table_manifest(table, files))

        self._save(table_manifest(table, files))
        run_jobs(jobs, pool, self.run, self.settings.chunk_retries, record)
        logger.info("generated %s rows of %s in %s chunks", plan.rows, table.name, len(files))

    def _save(self, entry: TableManifest) -> None:
        """Replace the table's entry in the progress manifest and write it."""
        tables = [t for t in self.progress.tables if t.name != entry.name] + [entry]
        self.progress = self.progress.model_copy(update={"tables": tables})
        write_manifest(self.out_dir, self.progress)


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


def table_manifest(table: TableSpec, files: dict[int, ManifestFile]) -> TableManifest:
    """A table's manifest entry from its finished chunks, in part order."""
    ordered = [files[i] for i in sorted(files)]
    counts: Counter[str] = Counter()
    for file in ordered:
        counts.update({c.scenario: c.rows for c in file.scenarios})
    return TableManifest(
        name=table.name,
        rows=sum(f.rows for f in ordered),
        files=ordered,
        columns=output_schema(table),
        scenario_counts=[ScenarioCount(scenario=s, rows=c) for s, c in sorted(counts.items())],
    )


def _run_hash(spec: GenerationSpec, options: RunOptions, settings: GenerationSettings) -> str:
    """What shapes the bytes: the spec, the options and the settings (worker count excluded)."""
    shaping = settings.model_dump(mode="json", exclude={"workers"})
    text = json.dumps(
        [spec.spec_hash(), options.max_root_rows, shaping], sort_keys=True, default=str
    )
    return hashlib.sha256(text.encode()).hexdigest()


def _finished_chunks(out_dir: Path, run_hash: str) -> dict[str, dict[int, ManifestFile]]:
    """Chunks a previous attempt of this run finished; anything else in `out_dir` is cleared."""
    try:
        previous = read_manifest(out_dir) if (out_dir / MANIFEST_FILE).exists() else None
    except (ValidationError, ValueError, OSError):
        previous = None  # an unreadable or older manifest cannot be trusted; start over
    if previous is None or previous.run_hash != run_hash:
        shutil.rmtree(out_dir / DATA_DIR, ignore_errors=True)
        return {}
    return {
        t.name: {
            int(Path(f.path).stem.removeprefix("part-")): f
            for f in t.files
            if (out_dir / f.path).exists()
        }
        for t in previous.tables
    }
