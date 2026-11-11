"""Profile reuse (SPEC 8.3).

A cached profile serves until its schema changes, it is older than `cache_max_age_days`, or the
caller asks for a refresh.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from pydantic import BaseModel

from synthgen.config import ProfilingSettings
from synthgen.domain.metadata import Relationship, TableMeta
from synthgen.domain.ports import Connector
from synthgen.profiling.profiler import profile_table, schema_hash


class CachedEntry(Protocol):
    """A cached profile's version and age."""

    @property
    def version(self) -> str:
        """Content-derived version."""
        ...

    @property
    def created_at(self) -> datetime:
        """When it was computed."""
        ...


class ProfileCache(Protocol):
    """Where profiles are kept."""

    def put(self, source: str, table: str, schema_hash: str, profile: BaseModel) -> str:
        """Store a profile and return its version."""
        ...

    def latest(self, source: str, table: str, schema_hash: str) -> CachedEntry | None:
        """The newest profile for this schema, or None on a miss."""
        ...


@dataclass(frozen=True)
class ProfileRequest:
    """One table to profile and how fresh its profile must be."""

    source: str
    meta: TableMeta
    relationships: Sequence[Relationship]
    refresh: bool
    now: datetime


def profile_version(
    request: ProfileRequest, connector: Connector, cache: ProfileCache, settings: ProfilingSettings
) -> str:
    """The version of a fresh-enough cached profile, computing and storing one when needed."""
    digest = schema_hash(request.meta)
    cached = cache.latest(request.source, request.meta.name, digest)
    max_age = timedelta(days=settings.cache_max_age_days)
    if cached is not None and not request.refresh and request.now - cached.created_at <= max_age:
        return cached.version
    profile = profile_table(
        request.source,
        connector,
        request.meta,
        request.relationships,
        settings=settings,
        seed=settings.sample_seed,
        now=request.now,
    )
    return cache.put(request.source, request.meta.name, digest, profile)
