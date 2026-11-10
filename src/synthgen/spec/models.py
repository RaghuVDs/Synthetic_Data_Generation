"""The generation spec (SPEC 8.4): a code-compiled baseline plus LLM-written scenario overlays."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field

from synthgen.domain.base import StrictModel
from synthgen.domain.plan import ScenarioCategory, ScenarioUnit
from synthgen.spec.generators import Generator, Scalar

SPEC_FORMAT_VERSION = 1
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


class ChildCount(StrictModel):
    """Share of parents that have this many children."""

    children: int = Field(ge=0)
    weight: float = Field(ge=0.0)


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


class TableSpec(StrictModel):
    """One table: how many rows and how each column is made."""

    name: str
    row_rule: RowRule
    columns: list[ColumnSpec] = Field(min_length=1)

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


class ScenarioSpec(StrictModel):
    """A scenario overlay. Rows of `entity` are assigned; descendant rows inherit the scenario."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    category: ScenarioCategory
    share: float = Field(ge=0.0, le=1.0)
    unit: ScenarioUnit
    entity: str
    labels: list[LabelSetting]
    overrides: list[ColumnOverride]


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
