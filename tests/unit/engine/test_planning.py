from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from synthgen.engine.planning import _parent_bounds, chunk_limit, plan_child, plan_root, stream
from synthgen.engine.runner import RunOptions, generate_dataset
from synthgen.engine.scenarios import scenario_counts
from synthgen.engine.writer import read_manifest, read_table
from synthgen.spec.models import TYPICAL, GenerationSpec

from .test_runner import BIG, GENERATION, _col, make_spec

SMALL = GENERATION.model_copy(update={"chunk_rows": 97})


def test_streams_are_the_spawned_children() -> None:
    spawned = np.random.SeedSequence(11, spawn_key=(2,)).spawn(4)[3]
    assert stream(11, 2, 3).random(5).tolist() == np.random.default_rng(spawned).random(5).tolist()
    assert stream(11, 2, 3).random(3).tolist() != stream(11, 2, 4).random(3).tolist()


def test_wide_tables_get_shorter_chunks() -> None:
    spec = make_spec()
    capped = GENERATION.model_copy(update={"max_cells_per_chunk": 1_000})
    assert chunk_limit(spec.tables[0], capped) == 1_000 // 3
    assert chunk_limit(spec.tables[0], GENERATION) == GENERATION.chunk_rows


def test_root_chunks_are_consecutive_with_exact_quotas() -> None:
    spec = make_spec(customers=1_000, scenarios=[BIG])
    plan = plan_root(spec, 0, 1_000, SMALL)
    assert [c.rows for c in plan.chunks] == [97] * 10 + [30]
    assert [c.row_offset for c in plan.chunks] == [97 * i for i in range(11)]
    assert sum(c.quotas[0] for c in plan.chunks) == scenario_counts([0.1], 1_000)[0]
    assert all(c.quotas[0] <= c.rows for c in plan.chunks)


def test_empty_tables_still_get_one_chunk() -> None:
    assert [c.rows for c in plan_root(make_spec(), 0, 0, SMALL).chunks] == [0]
    assert _parent_bounds(np.zeros(0, dtype=np.int64), 10) == [(0, 0)]


def test_families_are_never_split() -> None:
    counts = np.array([3, 3, 3, 25, 1, 0, 2], dtype=np.int64)
    bounds = _parent_bounds(counts, 7)
    assert bounds == [(0, 2), (2, 3), (3, 4), (4, 7)]
    assert sum(b - a for a, b in bounds) == counts.size


def test_child_quotas_only_use_typical_rows() -> None:
    orders_big = {**BIG, "entity": "orders", "labels": [], "overrides": []}
    spec = make_spec(customers=500, scenarios=[orders_big])
    parents = np.array([TYPICAL] * 400 + ["inherited"] * 100, dtype=object)
    plan = plan_child(spec, 1, parents, SMALL, use_target=True)
    assert plan.rows == 1_000
    assert sum(c.rows for c in plan.chunks) == 1_000
    starts = [c.parent_start for c in plan.chunks]
    assert starts == sorted(starts) and starts[0] == 0
    typical_rows = sum(
        int(c.parent_counts[: max(0, 400 - c.parent_start)].sum())
        for c in plan.chunks
        if c.parent_counts is not None
    )
    assert sum(c.quotas[0] for c in plan.chunks) == scenario_counts([0.1], typical_rows)[0]


def _digests(path: Path) -> dict[str, str]:
    return {
        p.relative_to(path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((path / "data").rglob("*.parquet"))
    }


def test_chunked_runs_keep_keys_scenarios_and_families_whole(tmp_path: Path) -> None:
    data = make_spec(customers=1_000, scenarios=[BIG]).model_dump()
    data["tables"][1]["columns"].append(
        _col("code_copy", "text", {"type": "derived", "expression": "customers.code"})
    )
    spec = GenerationSpec.model_validate(data)
    manifest = generate_dataset(spec, tmp_path / "a", RunOptions(), SMALL)
    customers, orders = manifest.tables
    assert len(customers.files) == 11 and len(orders.files) > 11
    ids = read_table(tmp_path / "a", customers, ["id"]).column("id").to_pylist()
    assert ids == list(range(1, 1_001))
    order_ids = read_table(tmp_path / "a", orders, ["id"]).column("id").to_pylist()
    assert order_ids == list(range(1, orders.rows + 1))
    assert {c.scenario: c.rows for c in customers.scenario_counts}["big"] == 100
    codes = dict(
        zip(
            ids,
            read_table(tmp_path / "a", customers, ["code"]).column("code").to_pylist(),
            strict=True,
        )
    )
    owner_part: dict[int, int] = {}
    for part, file in enumerate(orders.files):
        for row in pq.read_table(tmp_path / "a" / file.path).to_pylist():
            assert row["code_copy"] == codes[row["customer_id"]]
            assert owner_part.setdefault(row["customer_id"], part) == part  # one family, one chunk
    rerun = generate_dataset(spec, tmp_path / "b", RunOptions(), SMALL)
    assert rerun == manifest
    assert _digests(tmp_path / "a") == _digests(tmp_path / "b")


def test_inherited_scenarios_cross_chunk_boundaries(tmp_path: Path) -> None:
    generate_dataset(make_spec(customers=1_000, scenarios=[BIG]), tmp_path, RunOptions(), SMALL)
    manifest = read_manifest(tmp_path)
    customers = read_table(tmp_path, manifest.tables[0], ["id", "_scenario"]).to_pylist()
    big = {r["id"] for r in customers if r["_scenario"] == "big"}
    orders = read_table(tmp_path, manifest.tables[1], ["customer_id", "_scenario"]).to_pylist()
    assert Counter(r["_scenario"] for r in orders if r["customer_id"] in big) == Counter(
        {"big": sum(1 for r in orders if r["customer_id"] in big)}
    )


@pytest.mark.parametrize("rows", [1, 96, 97, 98])
def test_chunk_edges(tmp_path: Path, rows: int) -> None:
    manifest = generate_dataset(make_spec(customers=rows), tmp_path, RunOptions(), SMALL)
    assert manifest.tables[0].rows == rows
    assert sum(f.rows for f in manifest.tables[0].files) == rows
