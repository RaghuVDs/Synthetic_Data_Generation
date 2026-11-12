"""Column generators, a tagged union on `type` (SPEC 8.4).

Every field is required so the union works as a strict-mode JSON schema for the overlay writer.
A `conditional` generator holds leaf generators only, so the schema never recurses.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from synthgen.domain.base import StrictModel
from synthgen.domain.errors import ExpressionError
from synthgen.spec.distributions import DistributionGen
from synthgen.spec.expressions import parse

Scalar = str | float | bool
"""A single value. Whole numbers are numbers; the engine casts to each column's type."""
WEIGHT_TOLERANCE = 1e-6
WEEKDAYS = 7
HOURS = 24
MAX_TEXT_LENGTH = 1000
"""Longest model-written text value accepted in a text pool."""


def check_weights(weights: list[float], expected: int, what: str) -> None:
    """Weights must be non-negative, match their values in number and sum to 1."""
    if len(weights) != expected:
        raise ValueError(f"{what}: {len(weights)} weights for {expected} values")
    if any(w < 0 for w in weights):
        raise ValueError(f"{what}: weights must not be negative")
    if abs(sum(weights) - 1.0) > WEIGHT_TOLERANCE:
        raise ValueError(f"{what}: weights sum to {sum(weights):.6f}, not 1")


class SequenceGen(StrictModel):
    """Consecutive integers: start, start + step, ... (keys)."""

    type: Literal["sequence"]
    start: int
    step: int = Field(ge=1)


class UuidGen(StrictModel):
    """Random version-4 UUID text drawn from the seeded stream."""

    type: Literal["uuid"]


class ForeignKeyGen(StrictModel):
    """A key drawn from an earlier table's generated keys.

    uniform: every key equally likely. zipf: keys ranked by a seeded hash, weight 1/rank^exponent.
    weighted: `hot_share` of rows use `hot_keys` keys (hot keys); the rest are uniform.
    shared_group: every row draws from the same `group_size` keys, so unrelated rows share them.
    Parameters a distribution does not use are null.
    """

    type: Literal["foreign_key"]
    parent_table: str
    parent_column: str
    distribution: Literal["uniform", "zipf", "weighted", "shared_group"]
    exponent: float | None = Field(gt=0.0)
    hot_keys: int | None = Field(ge=1)
    hot_share: float | None = Field(ge=0.0, le=1.0)
    group_size: int | None = Field(ge=1)

    @model_validator(mode="after")
    def _parameters(self) -> Self:
        used: set[str] = {
            "uniform": set[str](),
            "zipf": {"exponent"},
            "weighted": {"hot_keys", "hot_share"},
            "shared_group": {"group_size"},
        }[self.distribution]
        for name in ("exponent", "hot_keys", "hot_share", "group_size"):
            if (getattr(self, name) is not None) != (name in used):
                state = "needs" if name in used else "must not set"
                raise ValueError(f"{self.distribution} foreign key {state} {name}")
        return self


class EmpiricalGen(StrictModel):
    """Inverse-CDF sampling through evenly spaced quantiles, rounded to `decimals` when set."""

    type: Literal["empirical"]
    quantiles: list[float] = Field(min_length=2)
    decimals: int | None = Field(ge=0)

    @model_validator(mode="after")
    def _sorted(self) -> Self:
        if any(b < a for a, b in zip(self.quantiles, self.quantiles[1:], strict=False)):
            raise ValueError("quantiles must not decrease")
        return self


class CategoricalGen(StrictModel):
    """One of the listed values, drawn with the given weights."""

    type: Literal["categorical"]
    values: list[Scalar] = Field(min_length=1)
    weights: list[float]

    @model_validator(mode="after")
    def _weights(self) -> Self:
        check_weights(self.weights, len(self.values), "categorical")
        return self


class DatetimeGen(StrictModel):
    """ISO dates or timestamps between start and end, with optional weekday and hour mixes."""

    type: Literal["datetime"]
    start: str
    end: str
    granularity: Literal["date", "datetime"]
    weekday_weights: list[float] | None
    hour_weights: list[float] | None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.end < self.start:
            raise ValueError("datetime end is before start")
        if self.weekday_weights is not None:
            check_weights(self.weekday_weights, WEEKDAYS, "weekday")
        if self.hour_weights is not None:
            check_weights(self.hour_weights, HOURS, "hour")
        return self


class PatternGen(StrictModel):
    """Text from masks: `A` a letter, `9` a digit, anything else kept as written.

    With `check_digit` "luhn" the last digit of each value becomes its Luhn check digit.
    """

    type: Literal["pattern"]
    masks: list[str] = Field(min_length=1)
    weights: list[float]
    check_digit: Literal["luhn"] | None

    @model_validator(mode="after")
    def _weights(self) -> Self:
        check_weights(self.weights, len(self.masks), "pattern")
        if self.check_digit and any("9" not in m for m in self.masks):
            raise ValueError("a check digit needs a digit (9) in every mask")
        return self


class SemanticKind(StrEnum):
    """Personal fields drawn from seeded pools of fake values."""

    FIRST_NAME = "first_name"
    LAST_NAME = "last_name"
    FULL_NAME = "full_name"
    EMAIL = "email"
    PHONE = "phone"
    STREET_ADDRESS = "street_address"
    CITY = "city"
    POSTCODE = "postcode"
    COMPANY = "company"
    CARD_NUMBER = "card_number"
    DATE_OF_BIRTH = "date_of_birth"


class SemanticGen(StrictModel):
    """A fake personal value of one kind, from a pool seeded by the spec seed and the kind."""

    type: Literal["semantic"]
    kind: SemanticKind


class TextPoolGen(StrictModel):
    """Model-written free text: one of at most a few hundred short values, drawn uniformly."""

    type: Literal["text_pool"]
    values: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _lengths(self) -> Self:
        if any(len(v) > MAX_TEXT_LENGTH for v in self.values):
            raise ValueError(f"text pool values must be at most {MAX_TEXT_LENGTH} characters")
        return self


class BoundaryGen(StrictModel):
    """Values at limits (type, profile or rule edges), drawn uniformly from the listed points."""

    type: Literal["boundary"]
    points: list[Scalar] = Field(min_length=1)


class DerivedGen(StrictModel):
    """A value computed from other columns of the same row (grammar in spec/expressions.py)."""

    type: Literal["derived"]
    expression: str

    @model_validator(mode="after")
    def _parses(self) -> Self:
        try:
            parse(self.expression)
        except ExpressionError as error:
            raise ValueError(f"expression does not parse: {error}") from error
        return self


class ConstantGen(StrictModel):
    """The same value on every row; None makes the column all null."""

    type: Literal["constant"]
    value: Scalar | None


LeafGenerator = Annotated[
    SequenceGen
    | UuidGen
    | ForeignKeyGen
    | EmpiricalGen
    | DistributionGen
    | CategoricalGen
    | DatetimeGen
    | PatternGen
    | SemanticGen
    | TextPoolGen
    | BoundaryGen
    | DerivedGen
    | ConstantGen,
    Field(discriminator="type"),
]
"""Every generator except `conditional`."""


class ConditionCase(StrictModel):
    """Rows whose condition column holds one of `values` use `generator`."""

    values: list[Scalar] = Field(min_length=1)
    generator: LeafGenerator


class ConditionalGen(StrictModel):
    """A generator chosen per row by another column of the same row; other rows use `otherwise`."""

    type: Literal["conditional"]
    column: str
    cases: list[ConditionCase] = Field(min_length=1)
    otherwise: LeafGenerator


Generator = Annotated[
    SequenceGen
    | UuidGen
    | ForeignKeyGen
    | EmpiricalGen
    | DistributionGen
    | CategoricalGen
    | DatetimeGen
    | PatternGen
    | SemanticGen
    | TextPoolGen
    | BoundaryGen
    | DerivedGen
    | ConditionalGen
    | ConstantGen,
    Field(discriminator="type"),
]
