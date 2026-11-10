"""Column generators, a tagged union on `type` (SPEC 8.4).

Every field is required so the union works as a strict-mode JSON schema for the overlay writer.
M2 ships the generators the walking skeleton needs; M5 adds the rest.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from synthgen.domain.base import StrictModel

Scalar = str | int | float | bool
WEIGHT_TOLERANCE = 1e-6
WEEKDAYS = 7
HOURS = 24


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
    """A key drawn from an earlier table's generated keys."""

    type: Literal["foreign_key"]
    parent_table: str
    parent_column: str
    distribution: Literal["uniform"]


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
    """Text from masks: `A` a letter, `9` a digit, anything else kept as written."""

    type: Literal["pattern"]
    masks: list[str] = Field(min_length=1)
    weights: list[float]

    @model_validator(mode="after")
    def _weights(self) -> Self:
        check_weights(self.weights, len(self.masks), "pattern")
        return self


class ConstantGen(StrictModel):
    """The same value on every row; None makes the column all null."""

    type: Literal["constant"]
    value: Scalar | None


Generator = Annotated[
    SequenceGen
    | UuidGen
    | ForeignKeyGen
    | EmpiricalGen
    | CategoricalGen
    | DatetimeGen
    | PatternGen
    | ConstantGen,
    Field(discriminator="type"),
]
