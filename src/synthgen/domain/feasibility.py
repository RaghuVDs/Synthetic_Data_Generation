"""Feasibility of a planned scenario (SPEC 7.4). Nothing is dropped silently."""

from __future__ import annotations

from enum import StrEnum

from synthgen.domain.base import StrictModel


class FeasibilityStatus(StrEnum):
    """Whether the system can generate a scenario as planned."""

    EXPRESSIBLE = "expressible"
    PARTIAL = "partial"
    NOT_EXPRESSIBLE = "not_expressible"


class FeasibilityResult(StrictModel):
    """What can and cannot be generated for one scenario, and why."""

    scenario: str
    status: FeasibilityStatus
    covered: list[str]
    not_covered: list[str]
    reason: str | None
    alternative: str | None
