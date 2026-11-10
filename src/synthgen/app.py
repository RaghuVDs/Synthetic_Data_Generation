"""Composition root and use-case functions. Each public method mirrors a future REST endpoint."""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path

import pyarrow as pa

from synthgen.agents.graph import JobRunner, open_checkpointer
from synthgen.agents.llm.client import LlmClient
from synthgen.agents.llm.stub import StubClient
from synthgen.agents.nodes.deps import NodeDeps, Source
from synthgen.config import Settings, load_settings, read_environment
from synthgen.connectors.sqlite import SqliteConnector
from synthgen.domain.base import StrictModel
from synthgen.domain.errors import ConfigError, MetadataError, NotFoundError, SynthgenError
from synthgen.domain.job import IDENTIFIER_PATTERN, Approval, ArtifactKind, JobRecord, UserRequest
from synthgen.domain.metadata import SourceOverview, TableSummary
from synthgen.domain.pending import Pending, PendingApproval, PendingClarification
from synthgen.domain.report import ValidationReport
from synthgen.domain.summary import Summary
from synthgen.engine.writer import MANIFEST_FILE, Manifest, read_manifest, read_table
from synthgen.metadata.provider import SourceDoc, YamlMetadataProvider, load_source_doc
from synthgen.profiling.cache import ProfileRequest, profile_version
from synthgen.profiling.card import LlmSafeProfile, build_card
from synthgen.profiling.models import TableProfile
from synthgen.seed.finance import build_finance
from synthgen.seed.orders import build_orders
from synthgen.storage.artifact_store import ArtifactStore
from synthgen.storage.db import Clock, StateDatabase, utc_now
from synthgen.storage.job_store import JobStore
from synthgen.storage.profile_store import ProfileStore

SeedBuilder = Callable[[Path, int, float], dict[str, int]]
SEED_BUILDERS: dict[str, SeedBuilder] = {"finance": build_finance, "orders": build_orders}
JOB_ID_CHARS = 12


class JobView(StrictModel):
    """A job's record and what it is waiting for, if anything."""

    job: JobRecord
    pending: Pending | None


class JobReport(StrictModel):
    """Everything a finished job produced for the tester."""

    job: JobRecord
    final_report: ValidationReport | None
    summary: Summary | None
    manifest: Manifest | None


class Application:
    """Wires concrete classes from settings and exposes the use cases."""

    def __init__(
        self, settings: Settings, llm: LlmClient | None = None, clock: Clock = utc_now
    ) -> None:
        """Keep the settings; collaborators are built on first use. Tests may inject an LLM."""
        self._settings = settings
        self._llm = llm
        self._clock = clock

    @property
    def settings(self) -> Settings:
        """The settings this application was built with."""
        return self._settings

    def close(self) -> None:
        """Release the checkpoint connection if the graph was used."""
        runner = self.__dict__.get("_runner")
        if isinstance(runner, JobRunner):
            runner.close()

    def __enter__(self) -> Application:
        """Use as `with build_application(...) as app:` so resources are released."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close on leaving the block."""
        self.close()

    @cached_property
    def _db(self) -> StateDatabase:
        return StateDatabase(self._settings.paths.state_db_path)

    @cached_property
    def _jobs(self) -> JobStore:
        return JobStore(self._db, self._clock)

    @cached_property
    def _artifacts(self) -> ArtifactStore:
        return ArtifactStore(self._db, self._clock)

    @cached_property
    def _profiles(self) -> ProfileStore:
        return ProfileStore(self._db, self._clock)

    @cached_property
    def _runner(self) -> JobRunner:
        deps = NodeDeps(
            settings=self._settings,
            llm=self._llm or make_llm(self._settings),
            jobs=self._jobs,
            artifacts=self._artifacts,
            profiles=self._profiles,
            open_source=self._open_source,
            clock=self._clock,
        )
        return JobRunner(deps, open_checkpointer(self._settings.paths.checkpoints_db_path))

    def seed(self) -> dict[str, dict[str, int]]:
        """Build every fake source database; return row counts per source and table."""
        cfg = self._settings.seed
        return {
            name: build(self._source_path(name), cfg.random_seed, cfg.scale)
            for name, build in SEED_BUILDERS.items()
        }

    def list_tables(self, source: str) -> SourceOverview:
        """Tables of a source with row counts, plus every relationship."""
        with self._open_source(source) as src:
            doc_description = self._source_doc(source).description
            return SourceOverview(
                source=source,
                description=doc_description,
                tables=tuple(
                    TableSummary(
                        name=t.name,
                        description=t.description,
                        row_count=src.connector.row_count(t.name),
                        column_count=len(t.columns),
                    )
                    for t in src.metadata.tables()
                ),
                relationships=tuple(src.metadata.relationships()),
            )

    def profile_tables(
        self, source: str, tables: Sequence[str], refresh: bool
    ) -> list[LlmSafeProfile]:
        """Profile (or reuse cached profiles of) the tables and return their cards."""
        with self._open_source(source) as src:
            rels = src.metadata.relationships()
            cards = []
            for name in tables:
                meta = src.metadata.table(name)
                request = ProfileRequest(
                    source=source, meta=meta, relationships=rels, refresh=refresh, now=self._clock()
                )
                version = profile_version(
                    request, src.connector, self._profiles, self._settings.profiling
                )
                profile = self._profiles.get(source, name, version, TableProfile)
                cards.append(build_card(profile, meta, rels, self._settings.profiling))
            return cards

    def start_job(self, request: UserRequest) -> JobView:
        """Create a job and run it until it pauses or ends."""
        return self.run_job(self.create_job(request).job_id)

    def create_job(self, request: UserRequest) -> JobRecord:
        """Register a job for an existing source without running it."""
        self._require_source(request.source)
        return self._jobs.create(uuid.uuid4().hex[:JOB_ID_CHARS], request)

    def run_job(self, job_id: str) -> JobView:
        """Run a job from its start, or from its last checkpoint after an interruption."""
        job = self.get_status(job_id)
        if job.status.is_terminal or self.get_pending(job_id) is not None:
            return self.view(job_id)
        if self._runner.has_run(job_id):
            self._runner.resume(job_id)
        else:
            self._runner.start(job_id, job.request)
        return self.view(job_id)

    def view(self, job_id: str) -> JobView:
        """The job's record and pending question or plan."""
        return JobView(job=self.get_status(job_id), pending=self.get_pending(job_id))

    def get_status(self, job_id: str) -> JobRecord:
        """The job's current record."""
        return self._jobs.get(job_id)

    def get_pending(self, job_id: str) -> Pending | None:
        """What the job is waiting for; None when it is not paused."""
        return self._runner.pending(job_id)

    def answer_clarification(self, job_id: str, answer: str) -> JobView:
        """Give the paused intent analyst its answer and continue."""
        if not isinstance(self.get_pending(job_id), PendingClarification):
            raise SynthgenError(f"job {job_id} is not waiting for an answer")
        self._runner.resume(job_id, answer)
        return self.view(job_id)

    def decide_plan(self, job_id: str, approval: Approval) -> JobView:
        """Approve, edit or reject the pending plan and continue."""
        if not isinstance(self.get_pending(job_id), PendingApproval):
            raise SynthgenError(f"job {job_id} is not waiting for a plan decision")
        self._runner.resume(job_id, approval.model_dump(mode="json"))
        return self.view(job_id)

    def resume_job(self, job_id: str) -> JobView:
        """Continue a paused or interrupted job; a paused job comes back with its question."""
        return self.run_job(job_id)

    def get_report(self, job_id: str) -> JobReport:
        """Final validation report, summary and manifest, as far as the job got."""
        out = self._job_dir(job_id)
        return JobReport(
            job=self.get_status(job_id),
            final_report=self._artifacts.get(job_id, ArtifactKind.FINAL_REPORT, ValidationReport)
            if self._has(job_id, ArtifactKind.FINAL_REPORT)
            else None,
            summary=self._artifacts.get(job_id, ArtifactKind.SUMMARY, Summary)
            if self._has(job_id, ArtifactKind.SUMMARY)
            else None,
            manifest=read_manifest(out) if (out / MANIFEST_FILE).is_file() else None,
        )

    def _has(self, job_id: str, kind: ArtifactKind) -> bool:
        return bool(self._artifacts.versions(job_id, kind))

    def get_preview(self, job_id: str, table: str, rows: int) -> pa.Table:
        """The first rows of one generated table."""
        manifest = self.get_report(job_id).manifest
        entry = manifest.table(table) if manifest else None
        if entry is None:
            raise NotFoundError(f"job {job_id} has no output for table {table!r}")
        return read_table(self._job_dir(job_id), entry).slice(0, rows)

    def _job_dir(self, job_id: str) -> Path:
        return self._settings.paths.output_dir / "jobs" / job_id

    def _source_path(self, source: str) -> Path:
        if not re.fullmatch(IDENTIFIER_PATTERN, source):
            raise NotFoundError(f"invalid source name {source!r}")
        return self._settings.paths.data_dir / f"{source}.db"

    def _require_source(self, source: str) -> Path:
        path = self._source_path(source)
        if not path.is_file():
            raise NotFoundError(
                f"source {source!r} not found at {path}; run `python main.py seed` first"
            )
        return path

    @contextmanager
    def _open_source(self, source: str) -> Iterator[Source]:
        path = self._require_source(source)
        connector = SqliteConnector(path)
        try:
            yield Source(
                connector=connector,
                metadata=YamlMetadataProvider(self._source_doc(source), connector),
            )
        finally:
            connector.close()

    def _source_doc(self, source: str) -> SourceDoc:
        """The source's metadata YAML; without one, relationships come from the database."""
        path = self._settings.paths.metadata_dir / f"{source}.yaml"
        if not path.is_file():
            return SourceDoc(source=source)
        doc = load_source_doc(path)
        if doc.source != source:
            raise MetadataError(f"{path} describes source {doc.source!r}, expected {source!r}")
        return doc


def make_llm(settings: Settings) -> LlmClient:
    """The LLM client named by `LLM_PROVIDER`."""
    if settings.llm.provider == "stub":
        return StubClient(settings.llm.stub_fixtures)
    raise ConfigError("LLM_PROVIDER=openai_compatible is not available yet; use stub")


def build_application(
    environ: Mapping[str, str], settings_file: Path, env_file: Path | None
) -> Application:
    """Load settings from YAML, an optional .env file and the environment, then wire the app."""
    return Application(load_settings(settings_file, read_environment(environ, env_file)))
