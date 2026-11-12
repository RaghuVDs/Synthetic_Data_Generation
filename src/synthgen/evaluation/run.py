"""Run the golden set: one job per request and volume, driven through every pause, then scored.

Clarifying questions get the request's `answer` (or the configured default) and a plan still
waiting at the gate is approved as proposed, so no job needs a person. Jobs run one after
another; each request's row target is set on its `target_table`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel

from synthgen.domain.base import StrictModel
from synthgen.domain.intent import TestIntent
from synthgen.domain.job import (
    Approval,
    ApprovalDecision,
    ArtifactKind,
    JobRecord,
    LoopCounters,
    RowTarget,
    UserRequest,
)
from synthgen.domain.llm_call import LlmCallRecord
from synthgen.domain.pending import Pending, PendingApproval, PendingClarification
from synthgen.domain.plan import ScenarioPlan
from synthgen.domain.report import ValidationReport
from synthgen.domain.semantic import SemanticMap
from synthgen.evaluation.golden import GoldenRequest, GoldenSet
from synthgen.evaluation.scoring import EvalSummary, JobFacts, JobScore, score, summarise

ModelT = TypeVar("ModelT", bound=BaseModel)
_LOG = logging.getLogger(__name__)


class EvalJobs(Protocol):
    """The use cases an eval run drives (the application implements them)."""

    def create_job(self, request: UserRequest) -> JobRecord:
        """Register a job."""
        ...

    def run_job(self, job_id: str) -> object:
        """Run it until it pauses or ends."""
        ...

    def get_status(self, job_id: str) -> JobRecord:
        """Its current record."""
        ...

    def get_pending(self, job_id: str) -> Pending | None:
        """What it waits for, if anything."""
        ...

    def answer_clarification(self, job_id: str, answer: str) -> object:
        """Answer its questions and continue."""
        ...

    def decide_plan(self, job_id: str, approval: Approval) -> object:
        """Decide its plan and continue."""
        ...


class ArtifactReader(Protocol):
    """Versioned job artifacts."""

    def get(
        self, job_id: str, kind: ArtifactKind, model: type[ModelT], version: int | None = None
    ) -> ModelT:
        """One version, the latest when omitted."""
        ...

    def latest(self, job_id: str, kind: ArtifactKind, model: type[ModelT]) -> ModelT | None:
        """The latest version, or None."""
        ...

    def versions(self, job_id: str, kind: ArtifactKind) -> list[int]:
        """Stored versions, ascending."""
        ...


class CallLog(Protocol):
    """The audited LLM calls."""

    def for_job(self, job_id: str) -> list[LlmCallRecord]:
        """Every call of a job."""
        ...


@dataclass(frozen=True)
class EvalDeps:
    """Everything an eval run reads and drives; `counters` gives a job's loop counters."""

    jobs: EvalJobs
    artifacts: ArtifactReader
    calls: CallLog
    counters: Callable[[str], LoopCounters]
    clock: Callable[[], datetime]
    default_answer: str
    max_pauses: int


class EvalReport(StrictModel):
    """Scores of every job, with totals overall and per volume."""

    started_at: datetime
    scores: list[JobScore]
    summary: EvalSummary
    by_volume: dict[int, EvalSummary]


def run_golden(
    golden: GoldenSet,
    volumes: Sequence[int],
    deps: EvalDeps,
    on_score: Callable[[JobScore], None] | None = None,
) -> EvalReport:
    """Every request at every volume, scored in order; `on_score` sees each score as it lands."""
    started_at = deps.clock()
    scores: list[JobScore] = []
    for volume in volumes:
        for request in golden.requests:
            facts = run_one(request, volume, deps)
            found = score(request, volume, facts, golden.roles.get(request.source, {}))
            _LOG.info("eval %s at %d rows: %s", request.name, volume, found.status)
            scores.append(found)
            if on_score is not None:
                on_score(found)
    return EvalReport(
        started_at=started_at,
        scores=scores,
        summary=summarise(scores),
        by_volume={v: summarise([s for s in scores if s.volume == v]) for v in volumes},
    )


def run_one(golden: GoldenRequest, volume: int, deps: EvalDeps) -> JobFacts:
    """Run one golden request at one volume to its end and collect what it produced."""
    request = UserRequest(
        source=golden.source,
        tables=tuple(golden.tables),
        text=golden.request,
        row_targets=(RowTarget(table=golden.target_table, rows=volume),),
        auto_approve=True,
    )
    start = time.perf_counter()
    job_id = deps.jobs.create_job(request).job_id
    deps.jobs.run_job(job_id)
    for _ in range(deps.max_pauses):
        pending = deps.jobs.get_pending(job_id)
        if isinstance(pending, PendingClarification):
            deps.jobs.answer_clarification(job_id, golden.answer or deps.default_answer)
        elif isinstance(pending, PendingApproval):
            approval = Approval(decision=ApprovalDecision.APPROVED, decided_at=deps.clock())
            deps.jobs.decide_plan(job_id, approval)
        else:
            break
    return _facts(job_id, deps, time.perf_counter() - start)


def _facts(job_id: str, deps: EvalDeps, seconds: float) -> JobFacts:
    record = deps.jobs.get_status(job_id)
    calls = deps.calls.for_job(job_id)
    dry_runs = deps.artifacts.versions(job_id, ArtifactKind.DRY_RUN_REPORT)
    return JobFacts(
        job_id=job_id,
        status=record.status,
        error=record.error,
        intent=deps.artifacts.latest(job_id, ArtifactKind.INTENT, TestIntent),
        semantic_map=deps.artifacts.latest(job_id, ArtifactKind.SEMANTIC_MAP, SemanticMap),
        plan=deps.artifacts.latest(job_id, ArtifactKind.PLAN, ScenarioPlan),
        counters=deps.counters(job_id),
        has_spec=bool(deps.artifacts.versions(job_id, ArtifactKind.SPEC)),
        first_dry_run_passed=deps.artifacts.get(
            job_id, ArtifactKind.DRY_RUN_REPORT, ValidationReport, dry_runs[0]
        ).passed
        if dry_runs
        else None,
        final_passed=report.passed
        if (report := deps.artifacts.latest(job_id, ArtifactKind.FINAL_REPORT, ValidationReport))
        else None,
        tokens=sum((c.prompt_tokens or 0) + (c.completion_tokens or 0) for c in calls),
        llm_latency_ms=sum(c.latency_ms for c in calls),
        seconds=seconds,
    )


def save_report(report: EvalReport, directory: Path) -> Path:
    """Write the report as JSON named after its start time; return the file."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"eval-{report.started_at:%Y%m%dT%H%M%S}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path
