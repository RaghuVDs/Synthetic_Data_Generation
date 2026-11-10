"""Versioned JSON artifacts per job: intent, plan, spec, reports (SPEC 10)."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from synthgen.domain.errors import NotFoundError
from synthgen.domain.job import ArtifactKind
from synthgen.storage.db import Clock, StateDatabase, utc_now

ModelT = TypeVar("ModelT", bound=BaseModel)


class ArtifactStore:
    """Stores every version of each artifact so a job's history can be inspected."""

    def __init__(self, db: StateDatabase, clock: Clock = utc_now) -> None:
        """Use the given database; the clock is injectable for tests."""
        self._db = db
        self._clock = clock

    def put(self, job_id: str, kind: ArtifactKind, content: BaseModel) -> int:
        """Save a new version and return its number, starting at 1."""
        with self._db.connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 AS next FROM job_artifacts"
                " WHERE job_id = ? AND kind = ?",
                (job_id, kind),
            ).fetchone()
            version = int(row["next"])
            conn.execute(
                "INSERT INTO job_artifacts (job_id, kind, version, content_json, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (job_id, kind, version, content.model_dump_json(), self._clock().isoformat()),
            )
        return version

    def get(
        self,
        job_id: str,
        kind: ArtifactKind,
        model: type[ModelT],
        version: int | None = None,
    ) -> ModelT:
        """Load a version (the latest when omitted); raises NotFoundError when absent."""
        query = "SELECT content_json FROM job_artifacts WHERE job_id = ? AND kind = ?"
        params: tuple[str | int, ...] = (job_id, kind)
        if version is None:
            query += " ORDER BY version DESC LIMIT 1"
        else:
            query += " AND version = ?"
            params = (*params, version)
        with self._db.connect() as conn:
            row = conn.execute(query, params).fetchone()
        if row is None:
            which = "latest" if version is None else f"version {version}"
            raise NotFoundError(f"{kind} ({which}) not found for job {job_id}")
        return model.model_validate_json(row["content_json"])

    def versions(self, job_id: str, kind: ArtifactKind) -> list[int]:
        """Stored version numbers of one artifact kind, ascending."""
        with self._db.connect() as conn:
            rows = conn.execute(
                "SELECT version FROM job_artifacts WHERE job_id = ? AND kind = ? ORDER BY version",
                (job_id, kind),
            ).fetchall()
        return [int(r["version"]) for r in rows]
