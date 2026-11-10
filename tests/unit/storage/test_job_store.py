from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from synthgen.domain.errors import NotFoundError, StorageError
from synthgen.domain.job import Approval, ApprovalDecision, JobStatus, UserRequest
from synthgen.storage.db import StateDatabase
from synthgen.storage.job_store import JobStore

START = datetime(2026, 11, 10, 9, 0, tzinfo=UTC)
REQUEST = UserRequest(source="finance", tables=("customers",), text="test churn")


def _ticking_clock() -> Iterator[datetime]:
    moment = START
    while True:
        yield moment
        moment += timedelta(seconds=1)


@pytest.fixture
def store(tmp_path: Path) -> JobStore:
    ticks = _ticking_clock()
    return JobStore(StateDatabase(tmp_path / "nested" / "state.db"), clock=lambda: next(ticks))


def test_create_and_get(store: JobStore) -> None:
    created = store.create("job-1", REQUEST)
    assert created.status is JobStatus.QUEUED
    assert created.request == REQUEST
    assert created.error is None
    assert created.created_at == START
    assert store.get("job-1") == created


def test_duplicate_job_id_is_rejected(store: JobStore) -> None:
    store.create("job-1", REQUEST)
    with pytest.raises(StorageError):
        store.create("job-1", REQUEST)


def test_set_status_updates_time_and_error(store: JobStore) -> None:
    store.create("job-1", REQUEST)
    failed = store.set_status("job-1", JobStatus.FAILED, error="FK did not resolve")
    assert failed.status is JobStatus.FAILED
    assert failed.error == "FK did not resolve"
    assert failed.updated_at > failed.created_at


def test_missing_job_raises_not_found(store: JobStore) -> None:
    with pytest.raises(NotFoundError):
        store.get("absent")
    with pytest.raises(NotFoundError):
        store.set_status("absent", JobStatus.FAILED)
    with pytest.raises(NotFoundError):
        store.record_approval(
            "absent", 1, Approval(decision=ApprovalDecision.APPROVED, decided_at=START)
        )


def test_approvals_are_kept_in_order(store: JobStore) -> None:
    store.create("job-1", REQUEST)
    rejected = Approval(decision=ApprovalDecision.REJECTED, feedback="more", decided_at=START)
    approved = Approval(decision=ApprovalDecision.APPROVED, decided_at=START)
    store.record_approval("job-1", 1, rejected)
    store.record_approval("job-1", 2, approved)
    assert store.approvals("job-1") == [(1, rejected), (2, approved)]
    assert store.approvals("other") == []


def test_state_survives_reopening(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    JobStore(StateDatabase(path)).create("job-1", REQUEST)
    assert JobStore(StateDatabase(path)).get("job-1").request == REQUEST


def test_unopenable_path_raises_storage_error(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("not a folder", encoding="utf-8")
    with pytest.raises(StorageError):
        StateDatabase(blocker / "state.db")
