"""Versioned profiles in the `profile_cache` table. A version is a hash of the profile content."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import TypeVar

from pydantic import BaseModel

from synthgen.domain.errors import NotFoundError
from synthgen.storage.db import Clock, StateDatabase, utc_now

ModelT = TypeVar("ModelT", bound=BaseModel)
VERSION_CHARS = 16


class CachedProfile(BaseModel):
    """A stored profile with its version and age."""

    version: str
    created_at: datetime
    content_json: str


class ProfileStore:
    """Keeps every profile version so a job can always reload the exact one it used."""

    def __init__(self, db: StateDatabase, clock: Clock = utc_now) -> None:
        """Use the given database; the clock is injectable for tests."""
        self._db = db
        self._clock = clock

    def put(self, source: str, table: str, schema_hash: str, profile: BaseModel) -> str:
        """Store a profile and return its content-derived version; storing twice is harmless."""
        content = profile.model_dump_json()
        version = hashlib.sha256(content.encode()).hexdigest()[:VERSION_CHARS]
        with self._db.connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO profile_cache (source, table_name, schema_hash,"
                " profile_version, content_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (source, table, schema_hash, version, content, self._clock().isoformat()),
            )
        return version

    def get(self, source: str, table: str, version: str, model: type[ModelT]) -> ModelT:
        """Load one exact version; raises NotFoundError when absent."""
        with self._db.connect() as conn:
            row = conn.execute(
                "SELECT content_json FROM profile_cache"
                " WHERE source = ? AND table_name = ? AND profile_version = ?",
                (source, table, version),
            ).fetchone()
        if row is None:
            raise NotFoundError(f"profile {version} of {source}.{table} not found")
        return model.model_validate_json(row["content_json"])

    def latest(self, source: str, table: str, schema_hash: str) -> CachedProfile | None:
        """The newest profile for this schema; None is a cache miss, not an error."""
        with self._db.connect() as conn:
            row = conn.execute(
                "SELECT profile_version, content_json, created_at FROM profile_cache"
                " WHERE source = ? AND table_name = ? AND schema_hash = ?"
                " ORDER BY created_at DESC, rowid DESC LIMIT 1",
                (source, table, schema_hash),
            ).fetchone()
        if row is None:
            return None
        return CachedProfile(
            version=row["profile_version"],
            created_at=datetime.fromisoformat(row["created_at"]),
            content_json=row["content_json"],
        )
