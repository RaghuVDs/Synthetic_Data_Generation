"""What a paused job is waiting for: an answer to questions, or a decision on a plan (SPEC 3.5)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from synthgen.domain.base import StrictModel
from synthgen.domain.intent import Ambiguity
from synthgen.domain.plan import ScenarioPlan


class PendingClarification(StrictModel):
    """Blocking questions the intent analyst could not settle from the request."""

    kind: Literal["clarification"]
    questions: list[Ambiguity]


class PendingApproval(StrictModel):
    """A plan to approve, edit or reject, with any problems from a previous edit or repair."""

    kind: Literal["approval"]
    plan: ScenarioPlan
    plan_version: int
    errors: list[str]
    explanation: str | None


Pending = Annotated[PendingClarification | PendingApproval, Field(discriminator="kind")]
