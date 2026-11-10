"""Graph state (SPEC 3.2): small structures and references, checkpointed after every node."""

from __future__ import annotations

import enum
import types
import typing
from typing import Any, TypedDict

from pydantic import BaseModel

from synthgen.domain.base import StrictModel
from synthgen.domain.intent import TestIntent
from synthgen.domain.job import Approval, JobStatus, LoopCounters, QuestionAnswer, UserRequest
from synthgen.domain.metadata import TableSet
from synthgen.domain.patch import SpecPatch
from synthgen.domain.plan import ScenarioPlan
from synthgen.domain.report import ValidationReport
from synthgen.domain.semantic import SemanticMap
from synthgen.spec.models import ScenarioSpec, SpecError


class Route(StrictModel):
    """Where the graph goes next, the status the job shows meanwhile, and why it stopped."""

    next: str
    status: JobStatus
    reason: str | None


class JobState(TypedDict):
    """The SPEC 3.2 fields plus `approval_errors`, `patch` and `route` (PLAN decisions)."""

    job_id: str
    request: UserRequest
    intent: TestIntent | None
    clarifications: list[QuestionAnswer]
    table_set: TableSet | None
    profile_refs: dict[str, str]
    semantic_map: SemanticMap | None
    plan: ScenarioPlan | None
    approval: Approval | None
    approval_errors: list[str]
    overlays: list[ScenarioSpec]
    spec_ref: str | None
    spec_errors: list[SpecError]
    patch: SpecPatch | None
    dry_run_report: ValidationReport | None
    final_report: ValidationReport | None
    counters: LoopCounters
    status: JobStatus
    error: str | None
    route: Route | None


def initial_state(job_id: str, request: UserRequest) -> JobState:
    """A fresh state for a new job."""
    return JobState(
        job_id=job_id,
        request=request,
        intent=None,
        clarifications=[],
        table_set=None,
        profile_refs={},
        semantic_map=None,
        plan=None,
        approval=None,
        approval_errors=[],
        overlays=[],
        spec_ref=None,
        spec_errors=[],
        patch=None,
        dry_run_report=None,
        final_report=None,
        counters=LoopCounters(),
        status=JobStatus.ANALYSING,
        error=None,
        route=None,
    )


def checkpoint_types() -> list[type]:
    """Model and enum classes the state can hold: the checkpointer's allow-list."""
    found: dict[type, None] = {}
    pending: list[Any] = list(typing.get_type_hints(JobState).values())  # Any: type expressions
    while pending:
        hint = pending.pop()
        if isinstance(hint, type) and issubclass(hint, BaseModel | enum.Enum):
            if hint in found:
                continue
            found[hint] = None
            if issubclass(hint, BaseModel):
                pending.extend(f.annotation for f in hint.model_fields.values())
        elif isinstance(hint, types.UnionType) or typing.get_origin(hint) is not None:
            pending.extend(typing.get_args(hint))
    return list(found)
