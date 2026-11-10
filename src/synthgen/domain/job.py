"""Job-level models: the user's request, status, loop counters, answers and approvals (SPEC 3)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from synthgen.domain.base import StrictModel

IDENTIFIER_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*$"


class JobStatus(StrEnum):
    """Lifecycle of a job (SPEC 3.6)."""

    QUEUED = "queued"
    ANALYSING = "analysing"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    PROFILING = "profiling"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPILING = "compiling"
    DRY_RUN = "dry_run"
    REPAIRING = "repairing"
    GENERATING = "generating"
    VALIDATING = "validating"
    COMPLETED = "completed"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """True when the job can make no further progress."""
        return self in _TERMINAL


_TERMINAL = frozenset(
    {JobStatus.COMPLETED, JobStatus.NEEDS_REVIEW, JobStatus.FAILED, JobStatus.CANCELLED}
)


class ArtifactKind(StrEnum):
    """Kinds of versioned JSON artifacts kept per job in the state database (SPEC 10)."""

    INTENT = "intent"
    SEMANTIC_MAP = "semantic_map"
    PLAN = "plan"
    SPEC = "spec"
    BASELINE_SPEC = "baseline_spec"
    DRY_RUN_REPORT = "dry_run_report"
    FINAL_REPORT = "final_report"
    SUMMARY = "summary"


class RowTarget(StrictModel):
    """Requested number of rows for one table (`--rows table=N`)."""

    table: str = Field(pattern=IDENTIFIER_PATTERN)
    rows: int = Field(ge=1)


class UserRequest(StrictModel):
    """What the tester asked for: source, tables, request text, row targets and options."""

    source: str = Field(pattern=IDENTIFIER_PATTERN)
    tables: tuple[str, ...] = Field(min_length=1)
    text: str = Field(min_length=1)
    row_targets: tuple[RowTarget, ...] = ()
    seed: int = Field(default=0, ge=0)
    include_invalid: bool = False
    auto_approve: bool = False
    refresh_profiles: bool = False
    workers: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _check_names(self) -> Self:
        if len(set(self.tables)) != len(self.tables):
            raise ValueError("tables must not repeat")
        targets = [t.table for t in self.row_targets]
        if len(set(targets)) != len(targets):
            raise ValueError("row targets must not repeat a table")
        return self


class QuestionAnswer(StrictModel):
    """One clarification question put to the user and the answer they gave."""

    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class LoopCounters(StrictModel):
    """How many times each capped loop has run (SPEC 3.4)."""

    clarify: int = Field(default=0, ge=0)
    plan_rounds: int = Field(default=0, ge=0)
    spec_repair: int = Field(default=0, ge=0)
    validation_repair: int = Field(default=0, ge=0)


class ApprovalDecision(StrEnum):
    """What the user did at the approval gate (SPEC 9)."""

    APPROVED = "approved"
    EDITED = "edited"
    REJECTED = "rejected"


class ShareEdit(StrictModel):
    """A new share for one scenario, set by the user at approval."""

    scenario: str = Field(min_length=1)
    share: float = Field(ge=0.0, le=1.0)


class Approval(StrictModel):
    """The user's decision on a plan, with any edits or feedback."""

    decision: ApprovalDecision
    feedback: str | None = None
    share_edits: tuple[ShareEdit, ...] = ()
    removed_scenarios: tuple[str, ...] = ()
    decided_at: datetime

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        has_edits = bool(self.share_edits or self.removed_scenarios)
        if self.decision is ApprovalDecision.EDITED and not has_edits:
            raise ValueError("an edited approval needs at least one edit")
        if self.decision is not ApprovalDecision.EDITED and has_edits:
            raise ValueError("only an edited approval may carry edits")
        return self

    @property
    def has_feedback(self) -> bool:
        """True when a rejection carries text the planner can act on."""
        return bool(self.feedback and self.feedback.strip())


class JobRecord(StrictModel):
    """A job as stored: its request, current status and any terminal error."""

    job_id: str
    request: UserRequest
    status: JobStatus
    error: str | None
    created_at: datetime
    updated_at: datetime
