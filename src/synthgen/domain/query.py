"""Read-only query vocabulary shared by connectors and their callers (SPEC 8.2)."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from synthgen.domain.base import StrictModel
from synthgen.domain.job import IDENTIFIER_PATTERN

Scalar = int | float | str


class ColumnInfo(StrictModel):
    """A column as the database declares it."""

    name: str
    declared_type: str
    nullable: bool
    primary_key_position: int = Field(ge=0)
    """1-based position in the primary key, 0 when not part of it."""


class ForeignKeyInfo(StrictModel):
    """A foreign key as the database declares it."""

    column: str
    parent_table: str
    parent_column: str


class AggFunc(StrEnum):
    """Whitelisted aggregate functions. SQLite lacks stddev, so callers combine sum and sum_sq."""

    COUNT = "count"
    COUNT_DISTINCT = "count_distinct"
    NULL_COUNT = "null_count"
    SUM = "sum"
    SUM_SQ = "sum_sq"
    MIN = "min"
    MAX = "max"
    AVG = "avg"


class AggExpr(StrictModel):
    """One aggregate over one column, or COUNT(*) when column is None."""

    func: AggFunc
    column: str | None
    alias: str = Field(pattern=IDENTIFIER_PATTERN)

    @model_validator(mode="after")
    def _column_required(self) -> Self:
        if self.column is None and self.func is not AggFunc.COUNT:
            raise ValueError(f"{self.func} needs a column")
        return self


class Operator(StrEnum):
    """Comparison operators allowed in filters."""

    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"
    IN = "in"
    IS_NULL = "is_null"
    NOT_NULL = "not_null"


class Condition(StrictModel):
    """One predicate on one column. Values are always passed as SQL parameters."""

    column: str
    op: Operator
    value: Scalar | tuple[Scalar, ...] | None = None

    @model_validator(mode="after")
    def _value_matches_op(self) -> Self:
        if self.op in (Operator.IS_NULL, Operator.NOT_NULL):
            if self.value is not None:
                raise ValueError(f"{self.op} takes no value")
        elif self.op is Operator.IN:
            if not isinstance(self.value, tuple) or not self.value:
                raise ValueError("in needs a non-empty tuple of values")
        elif self.value is None or isinstance(self.value, tuple):
            raise ValueError(f"{self.op} needs one scalar value")
        return self


class Filter(StrictModel):
    """Conditions combined with AND."""

    conditions: tuple[Condition, ...] = Field(min_length=1)
