from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import pytest

from synthgen.domain.errors import EngineError
from synthgen.engine.chunk import ChunkJob, run_chunk
from synthgen.engine.runner import RunOptions, generate_dataset
from synthgen.engine.writer import MANIFEST_FILE, ManifestFile, read_manifest
from synthgen.spec.models import GenerationSpec

from .test_runner import BIG, GENERATION, make_spec

SMALL = GENERATION.model_copy(update={"chunk_rows": 150})


def _spec() -> GenerationSpec:
    return make_spec(customers=600, scenarios=[BIG])


def _digests(path: Path) -> dict[str, str]:
    found = sorted((path / "data").rglob("*"))
    return {
        p.relative_to(path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in found
        if p.is_file()
    }


def _key(job: ChunkJob) -> tuple[int, int]:
    return job.plan.table_index, job.plan.index


def test_two_workers_give_the_same_bytes_as_one(tmp_path: Path) -> None:
    one = generate_dataset(_spec(), tmp_path / "one", RunOptions(), SMALL, workers=1)
    two = generate_dataset(_spec(), tmp_path / "two", RunOptions(), SMALL, workers=2)
    assert len(one.tables[1].files) > 1
    assert one == two
    assert _digests(tmp_path / "one") == _digests(tmp_path / "two")
    assert (tmp_path / "one" / MANIFEST_FILE).read_bytes() == (
        tmp_path / "two" / MANIFEST_FILE
    ).read_bytes()


def test_a_failed_chunk_is_retried_with_the_same_seed(tmp_path: Path) -> None:
    clean = generate_dataset(_spec(), tmp_path / "clean", RunOptions(), SMALL)
    calls: Counter[tuple[int, int]] = Counter()

    def flaky(job: ChunkJob) -> ManifestFile:
        calls[_key(job)] += 1
        if _key(job) == (1, 2) and calls[_key(job)] < 3:
            raise OSError("disk hiccup")
        return run_chunk(job)

    retried = generate_dataset(_spec(), tmp_path / "retried", RunOptions(), SMALL, run=flaky)
    assert calls[(1, 2)] == 3
    assert retried == clean
    assert _digests(tmp_path / "retried") == _digests(tmp_path / "clean")


def test_a_chunk_that_keeps_failing_fails_the_run_and_leaves_no_partial_files(
    tmp_path: Path,
) -> None:
    def broken(job: ChunkJob) -> ManifestFile:
        if _key(job) == (1, 2):
            run_chunk(job)  # writes the part, then fails as if the process died afterwards
            raise OSError("lost")
        return run_chunk(job)

    with pytest.raises(EngineError, match="chunk 2 of orders failed 3 times"):
        generate_dataset(_spec(), tmp_path, RunOptions(), SMALL, run=broken)
    orders = tmp_path / "data" / "orders"
    assert not (orders / "part-00002.parquet").exists()
    assert not list(orders.glob("*.tmp"))
    progress = read_manifest(tmp_path)
    assert not progress.complete
    assert [f.path.rsplit("/", 1)[1] for f in progress.tables[1].files] == [
        "part-00000.parquet",
        "part-00001.parquet",
    ]


def test_a_restart_skips_chunks_in_the_manifest(tmp_path: Path) -> None:
    clean = generate_dataset(_spec(), tmp_path / "clean", RunOptions(), SMALL)

    def stop_at_orders_2(job: ChunkJob) -> ManifestFile:
        if _key(job) == (1, 2):
            raise OSError("interrupted")
        return run_chunk(job)

    out = tmp_path / "resumed"
    with pytest.raises(EngineError):
        generate_dataset(_spec(), out, RunOptions(), SMALL, run=stop_at_orders_2)
    ran: list[tuple[int, int]] = []

    def counting(job: ChunkJob) -> ManifestFile:
        ran.append(_key(job))
        return run_chunk(job)

    resumed = generate_dataset(_spec(), out, RunOptions(), SMALL, run=counting)
    assert ran and all(key[0] == 1 and key[1] >= 2 for key in ran)
    assert resumed == clean
    assert _digests(out) == _digests(tmp_path / "clean")
    ran.clear()
    generate_dataset(_spec(), out, RunOptions(), SMALL, run=counting)
    assert ran == []  # a finished run reruns nothing


def test_other_runs_in_the_folder_are_replaced(tmp_path: Path) -> None:
    generate_dataset(_spec(), tmp_path, RunOptions(), SMALL)
    ran: list[tuple[int, int]] = []

    def counting(job: ChunkJob) -> ManifestFile:
        ran.append(_key(job))
        return run_chunk(job)

    generate_dataset(_spec(), tmp_path, RunOptions(max_root_rows=50), SMALL, run=counting)
    assert (0, 0) in ran
    assert read_manifest(tmp_path).tables[0].rows == 50
    (tmp_path / MANIFEST_FILE).write_text("{not json", encoding="utf-8")
    ran.clear()
    generate_dataset(_spec(), tmp_path, RunOptions(), SMALL, run=counting)
    assert (0, 0) in ran
    assert read_manifest(tmp_path).complete
