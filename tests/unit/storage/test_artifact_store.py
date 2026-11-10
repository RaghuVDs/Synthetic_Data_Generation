from __future__ import annotations

from pathlib import Path

import pytest

from synthgen.domain.errors import NotFoundError, StorageError
from synthgen.domain.job import ArtifactKind, UserRequest
from synthgen.domain.summary import Summary
from synthgen.storage.artifact_store import ArtifactStore
from synthgen.storage.db import StateDatabase
from synthgen.storage.job_store import JobStore


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    db = StateDatabase(tmp_path / "state.db")
    JobStore(db).create("job-1", UserRequest(source="s", tables=("t",), text="x"))
    return ArtifactStore(db)


def test_versions_increment_and_latest_is_default(store: ArtifactStore) -> None:
    first = Summary(text="first", caveats=[])
    second = Summary(text="second", caveats=["c"])
    assert store.put("job-1", ArtifactKind.SUMMARY, first) == 1
    assert store.put("job-1", ArtifactKind.SUMMARY, second) == 2
    assert store.get("job-1", ArtifactKind.SUMMARY, Summary) == second
    assert store.get("job-1", ArtifactKind.SUMMARY, Summary, version=1) == first
    assert store.versions("job-1", ArtifactKind.SUMMARY) == [1, 2]


def test_kinds_are_versioned_independently(store: ArtifactStore) -> None:
    store.put("job-1", ArtifactKind.SUMMARY, Summary(text="a", caveats=[]))
    assert store.versions("job-1", ArtifactKind.PLAN) == []


def test_missing_artifact_raises_not_found(store: ArtifactStore) -> None:
    with pytest.raises(NotFoundError):
        store.get("job-1", ArtifactKind.PLAN, Summary)
    store.put("job-1", ArtifactKind.SUMMARY, Summary(text="a", caveats=[]))
    with pytest.raises(NotFoundError):
        store.get("job-1", ArtifactKind.SUMMARY, Summary, version=9)


def test_artifact_for_unknown_job_is_rejected(store: ArtifactStore) -> None:
    with pytest.raises(StorageError):
        store.put("no-such-job", ArtifactKind.SUMMARY, Summary(text="a", caveats=[]))
