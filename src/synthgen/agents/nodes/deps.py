"""What nodes depend on, as protocols. Concrete stores and connectors are wired in app.py."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from synthgen.agents.llm.client import LlmClient
from synthgen.config import Settings
from synthgen.domain.job import Approval, ArtifactKind, JobRecord, JobStatus
from synthgen.domain.ports import Connector, MetadataProvider

ModelT = TypeVar("ModelT", bound=BaseModel)
StateUpdate = dict[str, Any]  # Any: a partial JobState; LangGraph merges it into the state


class Jobs(Protocol):
    """Job status and approval records."""

    def set_status(self, job_id: str, status: JobStatus, error: str | None = None) -> JobRecord:
        """Move the job to a new status."""
        ...

    def record_approval(self, job_id: str, plan_version: int, approval: Approval) -> None:
        """Keep the user's decision on a plan version."""
        ...


class Artifacts(Protocol):
    """Versioned JSON artifacts per job."""

    def put(self, job_id: str, kind: ArtifactKind, content: BaseModel) -> int:
        """Save a new version and return its number."""
        ...

    def get(
        self, job_id: str, kind: ArtifactKind, model: type[ModelT], version: int | None = None
    ) -> ModelT:
        """Load a version, the latest by default."""
        ...

    def versions(self, job_id: str, kind: ArtifactKind) -> list[int]:
        """Stored version numbers, ascending."""
        ...


class CachedProfileRecord(Protocol):
    """A cached profile's version and age."""

    @property
    def version(self) -> str:
        """Content-derived version."""
        ...

    @property
    def created_at(self) -> datetime:
        """When it was computed."""
        ...


class Profiles(Protocol):
    """Versioned profile cache."""

    def put(self, source: str, table: str, schema_hash: str, profile: BaseModel) -> str:
        """Store a profile and return its version."""
        ...

    def get(self, source: str, table: str, version: str, model: type[ModelT]) -> ModelT:
        """Load one exact version."""
        ...

    def latest(self, source: str, table: str, schema_hash: str) -> CachedProfileRecord | None:
        """The newest profile for this schema, or None on a cache miss."""
        ...


@dataclass(frozen=True)
class Source:
    """An open source: its read-only connector and merged metadata."""

    connector: Connector
    metadata: MetadataProvider


@dataclass(frozen=True)
class NodeDeps:
    """Everything a node may use. Built once per run in app.py."""

    settings: Settings
    llm: LlmClient
    jobs: Jobs
    artifacts: Artifacts
    profiles: Profiles
    open_source: Callable[[str], AbstractContextManager[Source]]
    clock: Callable[[], datetime]

    def job_dir(self, job_id: str) -> Path:
        """Output folder of one job."""
        return self.settings.paths.output_dir / "jobs" / job_id
