"""SemanticMap: column roles and business rules from the semantic_mapper (SPEC 5)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from synthgen.domain.base import StrictModel


class ColumnRole(StrEnum):
    """What a column means, independent of domain."""

    IDENTIFIER = "identifier"
    ENTITY_KEY = "entity_key"
    FOREIGN_KEY = "foreign_key"
    EVENT_TIME = "event_time"
    MEASURE = "measure"
    CATEGORY = "category"
    STATUS = "status"
    LOCATION = "location"
    FREE_TEXT = "free_text"
    LABEL = "label"
    SENSITIVE_IDENTIFIER = "sensitive_identifier"
    OTHER = "other"


class Grain(StrEnum):
    """What one row of a table represents."""

    ENTITY = "entity"
    EVENT = "event"
    REFERENCE = "reference"


class ColumnSemantics(StrictModel):
    """Role of one column, with an optional short note."""

    name: str
    role: ColumnRole
    note: str | None


class TableSemantics(StrictModel):
    """Grain, key columns and column roles of one table."""

    table: str
    grain: Grain
    entity_key: str | None
    event_time: str | None
    label_column: str | None
    columns: list[ColumnSemantics]


class BusinessRule(StrictModel):
    """A rule the data appears to follow, stated in plain language."""

    rule: str = Field(min_length=1)
    columns: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class SemanticMap(StrictModel):
    """Roles and rules for every table in the job."""

    tables: list[TableSemantics]
    business_rules: list[BusinessRule]

    def for_table(self, name: str) -> TableSemantics | None:
        """Semantics of one table, or None when the map does not cover it."""
        return next((t for t in self.tables if t.table == name), None)
