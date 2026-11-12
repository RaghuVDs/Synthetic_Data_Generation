"""Event shapes a scenario can impose: children per parent, sequences and state paths (SPEC 8.4)."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from synthgen.domain.base import StrictModel
from synthgen.spec.generators import Generator


class ChildCount(StrictModel):
    """Share of parents that have this many children."""

    children: int = Field(ge=0)
    weight: float = Field(ge=0.0)


class CardinalityOverride(StrictModel):
    """Children per parent row in `table` for the scenario's parents (relationship_shape).

    `[{"children": 0, "weight": 1}]` gives the scenario's parents no children (absence).
    """

    table: str
    counts: list[ChildCount] = Field(min_length=1)

    @model_validator(mode="after")
    def _weights(self) -> Self:
        if sum(c.weight for c in self.counts) <= 0:
            raise ValueError("cardinality weights must not all be zero")
        return self


class StepColumn(StrictModel):
    """A column a sequence step sets on its events, e.g. a status along a state path."""

    column: str
    generator: Generator


class SequenceStep(StrictModel):
    """`count` events, each `gap_min_seconds` to `gap_max_seconds` after the event before it."""

    count: int = Field(ge=1)
    gap_min_seconds: float = Field(gt=0.0)
    gap_max_seconds: float = Field(gt=0.0)
    columns: list[StepColumn]

    @model_validator(mode="after")
    def _gaps(self) -> Self:
        if self.gap_max_seconds < self.gap_min_seconds:
            raise ValueError("sequence gap maximum is below its minimum")
        return self


class SequenceSpec(StrictModel):
    """Ordered events per entity: the scenario's parents get exactly the steps' events in `table`.

    `entity_key` is the table's driving key column; `time_column` orders and spaces the events.
    """

    table: str
    entity_key: str
    time_column: str
    steps: list[SequenceStep] = Field(min_length=1)

    @property
    def events(self) -> int:
        """Events per entity across all steps."""
        return sum(s.count for s in self.steps)
