"""Chunk planning (SPEC 8.5): table sizes, chunks, child chunks by parent key range, streams.

Root tables are cut into chunks of consecutive rows. A child table is planned once its parent is
written: children per parent are drawn for every parent row, then chunks are cut at parent
boundaries so one entity's rows always share a chunk. Every chunk has its own random stream and
row offset, so chunks can run in any order, on any worker, and give the same bytes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from synthgen.config import GenerationSettings
from synthgen.engine.keys import plan_children
from synthgen.engine.scenarios import scenario_counts
from synthgen.spec.models import TYPICAL, GenerationSpec, ScenarioSpec, TableSpec

PLAN_SLOT = 0
"""Stream slot of a table's own planning draws; chunk i uses slot i + 1."""
Counts = NDArray[np.int64]


@dataclass(frozen=True)
class ChunkPlan:
    """One chunk: which rows, which parent rows they hang from, and its scenario quotas."""

    table_index: int
    index: int
    row_offset: int
    rows: int
    parent_start: int
    parent_counts: Counts | None
    """Children of each parent row from `parent_start` on; None for a root table."""
    quotas: tuple[int, ...]
    """Rows each of the table's own scenarios gets in this chunk, in spec order."""


@dataclass(frozen=True)
class TablePlan:
    """Every chunk of one table."""

    table: str
    rows: int
    chunks: tuple[ChunkPlan, ...]


def stream(seed: int, table_index: int, slot: int) -> np.random.Generator:
    """Child `slot` of `SeedSequence(seed, spawn_key=(table_index,))`, exactly as `spawn` makes it.

    Building the child directly avoids spawning every earlier child first.
    """
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(table_index, slot)))


def chunk_limit(table: TableSpec, settings: GenerationSettings) -> int:
    """Rows per chunk: `chunk_rows`, fewer for wide tables so a chunk stays under the cell cap."""
    by_cells = settings.max_cells_per_chunk // max(len(table.columns), 1)
    return max(1, min(settings.chunk_rows, by_cells))


def own_scenarios(spec: GenerationSpec, table: TableSpec) -> list[ScenarioSpec]:
    """Scenarios whose entity is this table, in spec order."""
    return [s for s in spec.scenarios if s.entity == table.name]


def plan_root(
    spec: GenerationSpec, table_index: int, rows: int, settings: GenerationSettings
) -> TablePlan:
    """Consecutive chunks of at most `chunk_limit` rows; every row is eligible for scenarios."""
    table = spec.tables[table_index]
    limit = chunk_limit(table, settings)
    sizes = [min(limit, rows - start) for start in range(0, rows, limit)] or [0]
    quotas = _quotas(own_scenarios(spec, table), sizes)
    offsets = np.cumsum([0, *sizes[:-1]])
    chunks = tuple(
        ChunkPlan(table_index, i, int(offsets[i]), sizes[i], 0, None, quotas[i])
        for i in range(len(sizes))
    )
    return TablePlan(table=table.name, rows=rows, chunks=chunks)


def plan_child(
    spec: GenerationSpec,
    table_index: int,
    parent_scenarios: NDArray[np.object_],
    settings: GenerationSettings,
    *,
    use_target: bool,
) -> TablePlan:
    """Children per parent, then chunks cut at parent boundaries (one entity, one chunk)."""
    table = spec.tables[table_index]
    rule = table.row_rule
    if rule.kind != "per_parent":
        raise ValueError(f"{table.name} is not a child table")
    counts = plan_children(
        rule,
        table.name,
        parent_scenarios,
        spec.scenarios,
        stream(spec.seed, table_index, PLAN_SLOT),
        use_target=use_target,
    )
    bounds = _parent_bounds(counts, chunk_limit(table, settings))
    typical = np.where(parent_scenarios == TYPICAL, counts, 0)
    sizes = [int(counts[a:b].sum()) for a, b in bounds]
    eligible = [int(typical[a:b].sum()) for a, b in bounds]
    quotas = _quotas(own_scenarios(spec, table), eligible)
    offsets = np.cumsum([0, *sizes[:-1]])
    chunks = tuple(
        ChunkPlan(table_index, i, int(offsets[i]), sizes[i], a, counts[a:b], quotas[i])
        for i, (a, b) in enumerate(bounds)
    )
    return TablePlan(table=table.name, rows=int(counts.sum()), chunks=chunks)


def _parent_bounds(counts: Counts, limit: int) -> list[tuple[int, int]]:
    """Parent row ranges whose children fit in `limit` rows; a larger family gets its own."""
    if counts.size == 0:
        return [(0, 0)]
    cumulative = np.cumsum(counts)
    bounds: list[tuple[int, int]] = []
    start = 0
    while start < counts.size:
        before = int(cumulative[start - 1]) if start else 0
        stop = int(np.searchsorted(cumulative, before + limit, side="right"))
        stop = max(stop, start + 1)
        bounds.append((start, stop))
        start = stop
    return bounds


def _quotas(scenarios: Sequence[ScenarioSpec], eligible: Sequence[int]) -> list[tuple[int, ...]]:
    """Exact rows per scenario for the table (PLAN A2), split over chunks by largest remainder."""
    total = sum(eligible)
    if not scenarios:
        return [()] * len(eligible)
    wanted = scenario_counts([s.share for s in scenarios], total)
    per_chunk = np.zeros((len(eligible), len(scenarios)), dtype=np.int64)
    weights = np.asarray(eligible, dtype=np.float64) / total if total else np.zeros(len(eligible))
    for j, count in enumerate(wanted):
        exact = weights * count
        base = np.floor(exact).astype(np.int64)
        left = count - int(base.sum())
        order = np.argsort(-(exact - base), kind="stable")
        base[order[:left]] += 1
        per_chunk[:, j] = base
    free = np.asarray(eligible) - per_chunk.sum(axis=1)
    _rebalance(per_chunk, free)
    return [tuple(int(v) for v in row) for row in per_chunk]


def _rebalance(per_chunk: NDArray[np.int64], free: NDArray[np.int64]) -> None:
    """Move rows from chunks that got more scenario rows than they have to chunks with room."""
    for i in np.flatnonzero(free < 0):
        for j in range(per_chunk.shape[1]):
            while free[i] < 0 and per_chunk[i, j] > 0:
                target = int(np.argmax(free))
                per_chunk[i, j] -= 1
                per_chunk[target, j] += 1
                free[i] += 1
                free[target] -= 1
