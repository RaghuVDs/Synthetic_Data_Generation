"""The state database: schema and short-lived connections that surface StorageError."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from synthgen.domain.errors import StorageError

Clock = Callable[[], datetime]

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    request_json TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS job_artifacts (
    job_id TEXT NOT NULL REFERENCES jobs(job_id),
    kind TEXT NOT NULL,
    version INTEGER NOT NULL,
    content_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (job_id, kind, version)
);
CREATE TABLE IF NOT EXISTS approvals (
    approval_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(job_id),
    plan_version INTEGER NOT NULL,
    decision TEXT NOT NULL,
    approval_json TEXT NOT NULL,
    decided_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS llm_calls (
    call_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    node TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    latency_ms INTEGER NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS profile_cache (
    source TEXT NOT NULL,
    table_name TEXT NOT NULL,
    schema_hash TEXT NOT NULL,
    profile_version TEXT NOT NULL,
    content_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (source, table_name, profile_version)
);
"""


def utc_now() -> datetime:
    """Current time in UTC; the default clock for stores."""
    return datetime.now(UTC)


class StateDatabase:
    """Owns the state database file. Opens one connection per unit of work."""

    def __init__(self, path: Path) -> None:
        """Create the parent folder and schema if missing."""
        self._path = path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StorageError(f"cannot create folder for state database {path}: {exc}") from exc
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    @property
    def path(self) -> Path:
        """Location of the database file."""
        return self._path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection that commits on success, rolls back on error and always closes."""
        try:
            conn = sqlite3.connect(self._path)
        except sqlite3.Error as exc:
            raise StorageError(f"cannot open state database {self._path}: {exc}") from exc
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            raise StorageError(f"state database error: {exc}") from exc
        finally:
            conn.close()
