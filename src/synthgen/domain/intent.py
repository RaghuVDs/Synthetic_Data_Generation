"""TestIntent: what the intent_analyst extracts from the request text (SPEC 5)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from synthgen.domain.base import StrictModel


class TestType(StrEnum):
    """Kind of test the data is for."""

    __test__ = False  # not a pytest test class

    MODEL_VALIDATION = "model_validation"
    RULE_TESTING = "rule_testing"
    PIPELINE_TESTING = "pipeline_testing"
    PERFORMANCE_TESTING = "performance_testing"
    UI_TESTING = "ui_testing"


class Ambiguity(StrictModel):
    """Something the request leaves open. A blocking one pauses the job for an answer."""

    question: str = Field(min_length=1)
    options: list[str]
    blocking: bool


class TestIntent(StrictModel):
    """Structured reading of the tester's request. Every field is required for strict mode."""

    __test__ = False  # not a pytest test class

    domain: str
    objective: str = Field(min_length=1)
    test_type: TestType
    entities_of_interest: list[str]
    must_include: list[str]
    must_exclude: list[str]
    volume_hint: int | None = Field(ge=1)
    include_invalid: bool
    ambiguities: list[Ambiguity]
    confidence: float = Field(ge=0.0, le=1.0)

    @property
    def blocking_ambiguities(self) -> list[Ambiguity]:
        """Ambiguities that must be answered before the job continues."""
        return [a for a in self.ambiguities if a.blocking]
