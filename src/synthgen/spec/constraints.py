"""Table constraints and deliberate defects (SPEC 8.4 `Constraint`, 7.2 rule_break)."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from synthgen.domain.base import StrictModel
from synthgen.domain.errors import ExpressionError
from synthgen.spec.expressions import parse
from synthgen.spec.generators import Scalar


class OrderConstraint(StrictModel):
    """`lower <= upper` wherever both are present. `lower` may be `parent.column`."""

    kind: Literal["order"]
    lower: str
    upper: str


class RangeConstraint(StrictModel):
    """Values of `column` lie inside [minimum, maximum]; a missing bound is open."""

    kind: Literal["range"]
    column: str
    minimum: Scalar | None
    maximum: Scalar | None


class UniqueConstraint(StrictModel):
    """No two rows share the same values in all of `columns`."""

    kind: Literal["unique"]
    columns: list[str] = Field(min_length=1)


class NotNullConstraint(StrictModel):
    """`column` is never null."""

    kind: Literal["not_null"]
    column: str


class ExpressionConstraint(StrictModel):
    """A boolean expression (spec/expressions.py grammar) that holds on every row."""

    kind: Literal["expression"]
    expression: str

    @model_validator(mode="after")
    def _parses(self) -> Self:
        try:
            parse(self.expression)
        except ExpressionError as error:
            raise ValueError(f"expression does not parse: {error}") from error
        return self


Constraint = Annotated[
    OrderConstraint | RangeConstraint | UniqueConstraint | NotNullConstraint | ExpressionConstraint,
    Field(discriminator="kind"),
]


class DefectKind(StrEnum):
    """Ways a value can be deliberately wrong (rule_break)."""

    NULL = "null"
    """Null in a column that is never null."""
    OUT_OF_RANGE = "out_of_range"
    """A number or date far beyond the column's values."""
    BAD_FORMAT = "bad_format"
    """Text or a date that does not follow the column's format."""
    ORPHAN_KEY = "orphan_key"
    """A foreign key that matches no parent row."""
    DUPLICATE_KEY = "duplicate_key"
    """A key or unique value copied from another row."""


class DefectSpec(StrictModel):
    """Break `column` on a share (`rate`) of the scenario's rows in `table`; see `_defect`."""

    table: str
    column: str
    kind: DefectKind
    rate: float = Field(gt=0.0, le=1.0)
