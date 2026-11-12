"""Composition root and use-case functions. Each public method mirrors a future REST endpoint."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
from functools import cached_property
from pathlib import Path

import httpx
import pyarrow as pa

from synthgen.agents.graph import JobRunner, open_checkpointer
from synthgen.agents.llm.client import LlmClient
from synthgen.agents.llm.openai_compatible import OpenAiCompatibleClient
from synthgen.agents.llm.structured import StructuredLlm
from synthgen.agents.llm.stub import StubClient
from synthgen.agents.nodes.deps import NodeDeps
from synthgen.agents.tools.cache import ToolResultCache
from synthgen.config import Settings, load_settings, read_environment
from synthgen.domain.base import StrictModel
from synthgen.domain.errors import NotFoundError, SynthgenError
from synthgen.domain.job import (
    Approval,
    ArtifactKind,
    JobRecord,
    JobStatus,
    UserRequest,
)
from synthgen.domain.metadata import SourceOverview
from synthgen.domain.pending import Pending, PendingApproval, PendingClarification
from synthgen.domain.report import ValidationReport
from synthgen.domain.summary import Summary
from synthgen.engine.writer import MANIFEST_FILE, Manifest, read_manifest, read_table
from synthgen.evaluation.golden import load_golden
from synthgen.evaluation.run import EvalDeps, EvalReport, run_golden, save_report
from synthgen.evaluation.scoring import JobScore
from synthgen.profiling.cache import profile_request, profile_version
from synthgen.profiling.card import LlmSafeProfile, build_card
from synthgen.profiling.models import TableProfile
from synthgen.scenarios.templates.loader import BUILT_IN, TemplateLibrary
from synthgen.sources import SourceCatalog
from synthgen.storage.artifact_store import ArtifactStore
from synthgen.storage.db import Clock, StateDatabase, utc_now
from synthgen.storage.job_store import JobStore
from synthgen.storage.llm_call_log import LlmCallLog
from synthgen.storage.profile_store import ProfileStore

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
        """Release the checkpoint and HTTP connections if they were opened."""
        runner = self.__dict__.get("_runner")
        if isinstance(runner, JobRunner):
            runner.close()
        http = self.__dict__.get("_http")
        if isinstance(http, httpx.Client):
            http.close()

    def __enter__(self) -> Application:
        """Use as `with build_application(...) as app:` so resources are released."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close on leaving the block."""
        self.close()

    @cached_property
    def _sources(self) -> SourceCatalog:
        return SourceCatalog(self._settings.paths)

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
    def _llm_log(self) -> LlmCallLog:
        return LlmCallLog(self._db, self._clock)

    @cached_property
    def _http(self) -> httpx.Client:
        return httpx.Client()

    @cached_property
    def _runner(self) -> JobRunner:
        client = self._llm or make_llm(self._settings, self._http)
        llm = StructuredLlm(client, self._llm_log, self._settings.agent.caps.llm_reprompts)
        deps = NodeDeps(
            settings=self._settings,
            llm=llm,
            jobs=self._jobs,
            artifacts=self._artifacts,
            profiles=self._profiles,
            open_source=self._sources.open,
            clock=self._clock,
            templates=TemplateLibrary(self._settings.paths.templates_dir or BUILT_IN),
            tool_cache=ToolResultCache(),
        )
        return JobRunner(deps, open_checkpointer(self._settings.paths.checkpoints_db_path))

    def seed(self) -> dict[str, dict[str, int]]:
        """Build every fake source database; return row counts per source and table."""
        return self._sources.seed(self._settings.seed)

    def list_tables(self, source: str) -> SourceOverview:
        """Tables of a source with row counts, plus every relationship."""
        return self._sources.overview(source)

    def profile_tables(
        self, source: str, tables: Sequence[str], refresh: bool
    ) -> list[LlmSafeProfile]:
        """Profile (or reuse cached profiles of) the tables and return their cards."""
        with self._sources.open(source) as src:
            rels = src.metadata.relationships()
            cards = []
            for name in tables:
                meta = src.metadata.table(name)
                request = profile_request(
                    source, src.metadata, name, refresh=refresh, now=self._clock()
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
        self._sources.require(request.source)
        return self._jobs.create(uuid.uuid4().hex[:JOB_ID_CHARS], request)

    def run_job(self, job_id: str) -> JobView:
        """Run a job from its start, or from its last checkpoint after an interruption."""
        self._expire_stale_plan(job_id)
        job = self.get_status(job_id)
        if job.status.is_terminal or self.get_pending(job_id) is not None:
            return self.view(job_id)
        if self._runner.has_run(job_id):
            self._runner.resume(job_id)
        else:
            self._runner.start(job_id, job.request)
        return self.view(job_id)

    def view(self, job_id: str) -> JobView:
        """The job's record and pending question or plan; a stale plan expires first."""
        self._expire_stale_plan(job_id)
        job = self.get_status(job_id)
        return JobView(
            job=job, pending=None if job.status.is_terminal else self.get_pending(job_id)
        )

    def _expire_stale_plan(self, job_id: str) -> None:
        """Cancel a job whose plan has waited longer than `plan_expiry_days` (SPEC 10)."""
        pending = self.get_pending(job_id)
        if not isinstance(pending, PendingApproval) or self.get_status(job_id).status.is_terminal:
            return
        created = self._artifacts.created_at(job_id, ArtifactKind.PLAN, pending.plan_version)
        days = self._settings.agent.plan_expiry_days
        if self._clock() - created > timedelta(days=days):
            reason = f"plan expired after {days} days without a decision"
            self._jobs.set_status(job_id, JobStatus.CANCELLED, reason)

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
        if self.view(job_id).job.status.is_terminal:
            raise SynthgenError(f"job {job_id} has ended: {self.get_status(job_id).error}")
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
            final_report=self._artifacts.latest(
                job_id, ArtifactKind.FINAL_REPORT, ValidationReport
            ),
            summary=self._artifacts.latest(job_id, ArtifactKind.SUMMARY, Summary),
            manifest=read_manifest(out) if (out / MANIFEST_FILE).is_file() else None,
        )

    def get_preview(self, job_id: str, table: str, rows: int) -> pa.Table:
        """The first rows of one generated table."""
        manifest = self.get_report(job_id).manifest
        entry = manifest.table(table) if manifest else None
        if entry is None:
            raise NotFoundError(f"job {job_id} has no output for table {table!r}")
        return read_table(self._job_dir(job_id), entry).slice(0, rows)

    def run_eval(
        self,
        volumes: Sequence[int],
        only: Sequence[str],
        sources: Sequence[str],
        on_score: Callable[[JobScore], None] | None = None,
    ) -> tuple[EvalReport, Path]:
        """Run the golden set (or part of it) at each volume; return the report and its file."""
        conf = self._settings.evaluation
        deps = EvalDeps(
            jobs=self,
            artifacts=self._artifacts,
            calls=self._llm_log,
            counters=self._runner.counters,
            clock=self._clock,
            default_answer=conf.default_answer,
            max_pauses=conf.max_pauses,
        )
        golden = load_golden(conf.golden_dir).select(only, sources)
        report = run_golden(golden, volumes or conf.volumes, deps, on_score)
        return report, save_report(report, self._settings.paths.output_dir / conf.report_dir)

    def _job_dir(self, job_id: str) -> Path:
        return self._settings.paths.output_dir / "jobs" / job_id


def make_llm(settings: Settings, http: httpx.Client) -> LlmClient:
    """The LLM client named by `LLM_PROVIDER`."""
    if settings.llm.provider == "stub":
        return StubClient(settings.llm.stub_fixtures)
    return OpenAiCompatibleClient(settings.llm, http)


def build_application(
    environ: Mapping[str, str], settings_file: Path, env_file: Path | None
) -> Application:
    """Load settings from YAML, an optional .env file and the environment, then wire the app."""
    return Application(load_settings(settings_file, read_environment(environ, env_file)))
