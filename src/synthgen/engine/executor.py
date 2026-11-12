"""Run chunk jobs in this process or a process pool, retrying a failed chunk with the same seed.

A chunk that still fails after `retries` attempts fails the run; its partial files are removed,
while finished chunks stay on disk and in the manifest for a restart (SPEC 10).
"""

from __future__ import annotations

import logging
import multiprocessing
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import contextmanager

from synthgen.domain.errors import EngineError
from synthgen.engine.chunk import ChunkJob
from synthgen.engine.writer import ManifestFile, remove_part

logger = logging.getLogger(__name__)
ChunkRunner = Callable[[ChunkJob], ManifestFile]
Record = Callable[[ChunkJob, ManifestFile], None]
"""Called in the parent process as each chunk finishes, in completion order."""


@contextmanager
def chunk_pool(workers: int) -> Iterator[ProcessPoolExecutor | None]:
    """A spawn-based process pool for several workers; None means run chunks in this process.

    Spawn (not fork) keeps worker processes free of the parent's threads and open connections.
    """
    if workers <= 1:
        yield None
        return
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        yield pool


def run_jobs(
    jobs: Sequence[ChunkJob],
    pool: ProcessPoolExecutor | None,
    run: ChunkRunner,
    retries: int,
    record: Record,
) -> None:
    """Run every job, at most `retries` attempts each; raise EngineError when one keeps failing."""
    if pool is None:
        for job in jobs:
            record(job, _inline(job, run, retries))
        return
    attempts = dict.fromkeys(range(len(jobs)), 0)
    pending: dict[Future[ManifestFile], int] = {pool.submit(run, j): i for i, j in enumerate(jobs)}
    while pending:
        finished, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in finished:
            index = pending.pop(future)
            job = jobs[index]
            error = future.exception()
            if error is None:
                record(job, future.result())
                continue
            attempts[index] += 1
            _failed(job, error, attempts[index], retries)
            pending[pool.submit(run, job)] = index


def _inline(job: ChunkJob, run: ChunkRunner, retries: int) -> ManifestFile:
    attempt = 0
    while True:
        try:
            return run(job)
        except Exception as error:
            attempt += 1
            _failed(job, error, attempt, retries)


def _failed(job: ChunkJob, error: BaseException, attempt: int, retries: int) -> None:
    """Remove the chunk's partial files; raise once the chunk has used all its attempts."""
    table = job.spec.tables[job.plan.table_index].name
    remove_part(job.out_dir, table, job.plan.index)
    logger.warning("chunk %s of %s failed (attempt %s): %s", job.plan.index, table, attempt, error)
    if attempt >= max(retries, 1):
        raise EngineError(
            f"chunk {job.plan.index} of {table} failed {attempt} times: {error}"
        ) from error
