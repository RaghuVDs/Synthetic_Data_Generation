"""The generation spec (SPEC 8.4): a code-compiled baseline plus LLM-written scenario overlays."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from synthgen.domain.base import StrictModel
from synthgen.domain.plan import ScenarioCategory, ScenarioUnit
from synthgen.spec.constraints import Constraint, DefectSpec
from synthgen.spec.expressions import ExprType
from synthgen.spec.generators import Generator, Scalar
from synthgen.spec.shapes import CardinalityOverride, ChildCount, SequenceSpec

SPEC_FORMAT_VERSION = 2
MATRIX_TOLERANCE = 1e-6
"""Allowed rounding error in a correlation matrix's symmetry and unit diagonal."""
TYPICAL = "typical"
"""`_scenario` and `_category` value of rows no scenario touched."""


class ColumnType(StrEnum):
    """Logical column type in the output."""

    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    TEXT = "text"
    DATE = "date"
    DATETIME = "datetime"


EXPR_TYPES = {
    ColumnType.INTEGER: ExprType.NUMBER,
    ColumnType.FLOAT: ExprType.NUMBER,
    ColumnType.BOOLEAN: ExprType.NUMBER,
    ColumnType.TEXT: ExprType.TEXT,
    ColumnType.DATE: ExprType.DATE,
    ColumnType.DATETIME: ExprType.DATETIME,
}
"""How expressions see each column type; booleans are stored as 0 and 1, so they are numbers."""


class ColumnSpec(StrictModel):
    """How one column is generated."""

    name: str
    type: ColumnType
    nullable: bool
    null_rate: float = Field(ge=0.0, le=1.0)
    primary_key: bool
    unique: bool
    sensitive: bool
    generator: Generator


class FixedRows(StrictModel):
    """A root table with a set number of rows."""

    kind: Literal["fixed"]
    rows: int = Field(ge=1)


class PerParentRows(StrictModel):
    """A child table sized by its driving parent: children per parent drawn from `counts`.

    The total is `target_rows` when set, else parents x mean children, rounded.
    """

    kind: Literal["per_parent"]
    parent_table: str
    parent_column: str
    child_column: str
    counts: list[ChildCount] = Field(min_length=1)
    target_rows: int | None = Field(ge=0)

    @property
    def mean_children(self) -> float:
        """Expected children per parent."""
        total = sum(c.weight for c in self.counts)
        return sum(c.children * c.weight for c in self.counts) / total if total else 0.0


RowRule = Annotated[FixedRows | PerParentRows, Field(discriminator="kind")]


class CorrelationSpec(StrictModel):
    """Spearman correlations between numeric columns, generated together by a Gaussian copula."""

    columns: list[str] = Field(min_length=2)
    spearman: list[list[float]]

    @model_validator(mode="after")
    def _shape(self) -> Self:
        size = len(self.columns)
        if len(set(self.columns)) != size:
            raise ValueError("correlation columns repeat")
        if len(self.spearman) != size or any(len(row) != size for row in self.spearman):
            raise ValueError(f"correlation matrix must be {size} x {size}")
        for i, row in enumerate(self.spearman):
            if abs(row[i] - 1.0) > MATRIX_TOLERANCE:
                raise ValueError("correlation matrix diagonal must be 1")
            for j, value in enumerate(row):
                if not -1.0 <= value <= 1.0:
                    raise ValueError("correlations must lie between -1 and 1")
                if abs(value - self.spearman[j][i]) > MATRIX_TOLERANCE:
                    raise ValueError("correlation matrix must be symmetric")
        return self


class TableSpec(StrictModel):
    """One table: how many rows, how each column is made, and which columns move together."""

    name: str
    row_rule: RowRule
    columns: list[ColumnSpec] = Field(min_length=1)
    correlation: CorrelationSpec | None
    constraints: list[Constraint]

    def column(self, name: str) -> ColumnSpec | None:
        """One column's spec, or None when the table has no such column."""
        return next((c for c in self.columns if c.name == name), None)

    @property
    def primary_key(self) -> ColumnSpec | None:
        """The single primary-key column, if any."""
        return next((c for c in self.columns if c.primary_key), None)


class LabelSetting(StrictModel):
    """A label column value carried by a scenario's rows."""

    table: str
    column: str
    value: Scalar


class ColumnOverride(StrictModel):
    """A generator that replaces the baseline one for a scenario's rows."""

    table: str
    column: str
    generator: Generator


class CorrelationOverride(CorrelationSpec):
    """Correlations a scenario's rows follow instead of the table's (correlation_change)."""

    table: str


class ScenarioSpec(StrictModel):
    """A scenario overlay. Rows of `entity` are assigned; descendant rows inherit the scenario."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    category: ScenarioCategory
    share: float = Field(ge=0.0, le=1.0)
    unit: ScenarioUnit
    entity: str
    labels: list[LabelSetting]
    overrides: list[ColumnOverride]
    correlations: list[CorrelationOverride]
    cardinality: list[CardinalityOverride]
    sequences: list[SequenceSpec]
    defects: list[DefectSpec]


class OverlaySet(StrictModel):
    """What the overlay writer returns: one overlay per approved scenario."""

    scenarios: list[ScenarioSpec]


class ProfileRef(StrictModel):
    """The profile version a table's baseline was compiled from."""

    table: str
    version: str


class OutputSpec(StrictModel):
    """How Parquet is written."""

    compression: Literal["zstd"]
    chunk_rows: int = Field(ge=1)


class GenerationSpec(StrictModel):
    """Everything the engine needs. Same spec and seed give byte-identical output."""

    version: int
    seed: int = Field(ge=0)
    profile_versions: list[ProfileRef]
    tables: list[TableSpec] = Field(min_length=1)
    scenarios: list[ScenarioSpec]
    output: OutputSpec

    def table(self, name: str) -> TableSpec | None:
        """One table's spec, or None when the spec has no such table."""
        return next((t for t in self.tables if t.name == name), None)

    def spec_hash(self) -> str:
        """Hash of the canonical JSON form; recorded in the manifest."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


class SpecError(StrictModel):
    """One spec validation failure, with where it came from so routing can act on it."""

    origin: Literal["baseline", "overlay"]
    scenario: str | None
    path: str
    message: str
