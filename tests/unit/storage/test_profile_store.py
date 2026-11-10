from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from synthgen.domain.errors import NotFoundError
from synthgen.domain.summary import Summary
from synthgen.storage.db import StateDatabase
from synthgen.storage.profile_store import ProfileStore


@pytest.fixture
def store(tmp_path: Path) -> ProfileStore:
    moments = iter(datetime(2026, 11, 1, tzinfo=UTC) + timedelta(hours=h) for h in range(100))
    return ProfileStore(StateDatabase(tmp_path / "state.db"), clock=lambda: next(moments))


def test_version_is_content_hash(store: ProfileStore) -> None:
    first = store.put("s", "t", "h1", Summary(text="one", caveats=[]))
    again = store.put("s", "t", "h1", Summary(text="one", caveats=[]))
    other = store.put("s", "t", "h1", Summary(text="two", caveats=[]))
    assert first == again
    assert first != other
    assert store.get("s", "t", first, Summary).text == "one"


def test_latest_is_newest_for_schema(store: ProfileStore) -> None:
    store.put("s", "t", "h1", Summary(text="old", caveats=[]))
    newest = store.put("s", "t", "h1", Summary(text="new", caveats=[]))
    store.put("s", "t", "h2", Summary(text="other schema", caveats=[]))
    cached = store.latest("s", "t", "h1")
    assert cached is not None
    assert cached.version == newest
    assert store.latest("s", "t", "unknown") is None


def test_missing_version_raises(store: ProfileStore) -> None:
    with pytest.raises(NotFoundError):
        store.get("s", "t", "nope", Summary)
