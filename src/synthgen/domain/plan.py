"""ScenarioPlan: the scenarios the planner proposes and the user approves (SPEC 5, 7)."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from synthgen.domain.base import StrictModel

_SHARE_EPSILON = 1e-9
"""Tolerance for floating-point rounding when shares sum to exactly 1."""


class ScenarioCategory(StrEnum):
    """Why a scenario exists in the data."""

    TYPICAL = "typical"
    TARGET_SCENARIO = "target_scenario"
    NEAR_MISS = "near_miss"
    BOUNDARY = "boundary"
    RARE = "rare"
    TEMPORAL = "temporal"
    VOLUME_SKEW = "volume_skew"
    INVALID = "invalid"


class ScenarioUnit(StrEnum):
    """What a scenario's share counts: whole entities or individual events."""

    ENTITY = "entity"
    EVENT = "event"


class Primitive(StrEnum):
    """The fourteen domain-neutral building blocks of every scenario (SPEC 7.2)."""

    VALUE_SHIFT = "value_shift"
    MIX_SHIFT = "mix_shift"
    CONDITIONAL_RULE = "conditional_rule"
    SEQUENCE = "sequence"
    TIMING_SHIFT = "timing_shift"
    GAP_OR_ABSENCE = "gap_or_absence"
    RELATIONSHIP_SHAPE = "relationship_shape"
    CROSS_ENTITY_LINK = "cross_entity_link"
    STATE_PATH = "state_path"
    CORRELATION_CHANGE = "correlation_change"
    BOUNDARY_VALUE = "boundary_value"
    VOLUME_SKEW = "volume_skew"
    TEXT_CONTENT = "text_content"
    RULE_BREAK = "rule_break"


class AssertionMetric(StrEnum):
    """What a signature assertion measures: the row value, or an aggregate per entity."""

    VALUE = "value"
    COUNT = "count"
    SUM = "sum"
    MEAN = "mean"
    DISTINCT_COUNT = "distinct_count"
    INTER_ARRIVAL_HOURS = "inter_arrival_hours"


class AssertionOperator(StrEnum):
    """Comparison used by a signature assertion."""

    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"
    IN = "in"
    NOT_IN = "not_in"


class Assertion(StrictModel):
    """One checkable statement about a scenario's rows (SPEC 5 `signature`)."""

    table: str
    column: str
    metric: AssertionMetric
    operator: AssertionOperator
    value: float | str | bool | list[str]
    window_hours: float | None = Field(gt=0)


class Lever(StrictModel):
    """One change a scenario makes: which column, through which primitive, how."""

    table: str
    column: str
    primitive: Primitive
    change: str = Field(min_length=1)


class PlannedScenario(StrictModel):
    """One scenario in the plan."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    category: ScenarioCategory
    description: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    share: float = Field(ge=0.0, le=1.0)
    unit: ScenarioUnit
    tables: list[str] = Field(min_length=1)
    primitives: list[Primitive] = Field(min_length=1)
    levers: list[Lever]
    signature: list[Assertion]
    template_ref: str | None


class ScenarioPlan(StrictModel):
    """Scenarios with shares summing to at most 1; the remainder is typical data (SPEC 7.3)."""

    scenarios: list[PlannedScenario]
    assumptions: list[str]

    @model_validator(mode="after")
    def _check_names_and_shares(self) -> Self:
        names = [s.name for s in self.scenarios]
        if len(set(names)) != len(names):
            raise ValueError("scenario names must be unique")
        if self.total_share > 1.0 + _SHARE_EPSILON:
            raise ValueError(f"scenario shares sum to {self.total_share:.4f}, above 1")
        return self

    @property
    def total_share(self) -> float:
        """Sum of all scenario shares."""
        return sum(s.share for s in self.scenarios)
