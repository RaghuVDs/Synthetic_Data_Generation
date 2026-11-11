"""Profile reuse (SPEC 8.3): same schema hash, younger than 7 days, and no `--refresh`."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import BaseModel

from synthgen.config import ProfilingSettings
from synthgen.connectors.sqlite import SqliteConnector
from synthgen.domain.metadata import Sensitivity
from synthgen.metadata.provider import ColumnDoc, SourceDoc, TableDoc, YamlMetadataProvider
from synthgen.profiling.cache import CachedEntry, profile_request, profile_version
from synthgen.profiling.models import TableProfile
from synthgen.storage.db import StateDatabase
from synthgen.storage.profile_store import ProfileStore

T0 = datetime(2026, 11, 1, tzinfo=UTC)


class CountingCache:
    """A ProfileStore that counts how often a profile is computed and stored."""

    def __init__(self, store: ProfileStore) -> None:
        self.store = store
        self.puts = 0

    def put(self, source: str, table: str, schema_hash: str, profile: BaseModel) -> str:
        self.puts += 1
        return self.store.put(source, table, schema_hash, profile)

    def latest(self, source: str, table: str, schema_hash: str) -> CachedEntry | None:
        return self.store.latest(source, table, schema_hash)


class Clock:
    """A settable clock shared by the store and the requests."""

    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def cache(tmp_path: Path, clock: Clock) -> CountingCache:
    return CountingCache(ProfileStore(StateDatabase(tmp_path / "state.db"), clock=clock))


@pytest.fixture
def connector(shop_db: Path) -> Iterator[SqliteConnector]:
    opened = SqliteConnector(shop_db)
    yield opened
    opened.close()


def _version(  # noqa: PLR0913 - the cache inputs plus what each test varies
    connector: SqliteConnector,
    cache: CountingCache,
    settings: ProfilingSettings,
    *,
    now: datetime,
    refresh: bool = False,
    doc: SourceDoc | None = None,
) -> str:
    provider = YamlMetadataProvider(doc or SourceDoc(source="shop"), connector)
    request = profile_request("shop", provider, "orders", refresh=refresh, now=now)
    return profile_version(request, connector, cache, settings)


def test_fresh_profiles_are_reused(
    connector: SqliteConnector, cache: CountingCache, profiling_settings: ProfilingSettings
) -> None:
    first = _version(connector, cache, profiling_settings, now=T0)
    again = _version(connector, cache, profiling_settings, now=T0 + timedelta(days=6))
    assert first == again and cache.puts == 1
    stored = cache.store.get("shop", "orders", first, TableProfile)
    assert stored.table == "orders" and stored.relationships


def test_old_profiles_are_recomputed_once(
    connector: SqliteConnector,
    cache: CountingCache,
    clock: Clock,
    profiling_settings: ProfilingSettings,
) -> None:
    first = _version(connector, cache, profiling_settings, now=T0)
    clock.now = T0 + timedelta(days=8)
    again = _version(connector, cache, profiling_settings, now=clock.now)
    assert again == first and cache.puts == 2  # same data, same content-derived version
    _version(connector, cache, profiling_settings, now=clock.now + timedelta(days=1))
    assert cache.puts == 2  # the recomputation refreshed the age


def test_refresh_recomputes(
    connector: SqliteConnector, cache: CountingCache, profiling_settings: ProfilingSettings
) -> None:
    _version(connector, cache, profiling_settings, now=T0)
    _version(connector, cache, profiling_settings, now=T0, refresh=True)
    assert cache.puts == 2


def test_schema_or_tag_changes_miss_the_cache(
    connector: SqliteConnector, cache: CountingCache, profiling_settings: ProfilingSettings
) -> None:
    _version(connector, cache, profiling_settings, now=T0)
    tagged = SourceDoc(
        source="shop",
        tables={
            "orders": TableDoc(columns={"amount": ColumnDoc(sensitivity=Sensitivity.INTERNAL)})
        },
    )
    _version(connector, cache, profiling_settings, now=T0, doc=tagged)
    assert cache.puts == 2
