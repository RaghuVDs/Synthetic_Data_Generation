"""What planner tools can see: the job's tables, profiles, source and templates (SPEC 6)."""

from __future__ import annotations

from dataclasses import dataclass

from synthgen.config import ProfilingSettings
from synthgen.domain.errors import ToolArgumentError
from synthgen.domain.metadata import Sensitivity
from synthgen.domain.ports import Source
from synthgen.profiling.models import ColumnProfile, TableProfile
from synthgen.scenarios.templates.loader import TemplateLibrary


@dataclass(frozen=True)
class ToolContext:
    """One job's view for tools. Only tables in the job's table set are visible."""

    job_id: str
    source: Source
    profiles: dict[str, TableProfile]
    templates: TemplateLibrary
    settings: ProfilingSettings

    def table(self, name: str) -> TableProfile:
        """A job table's profile; unknown names are argument errors the model can read."""
        if name not in self.profiles:
            raise ToolArgumentError(
                f"unknown table {name!r}; tables: {', '.join(sorted(self.profiles))}"
            )
        return self.profiles[name]

    def column(self, table: str, name: str, *, values: bool = False) -> ColumnProfile:
        """A column's profile. With `values`, a sensitive column is refused (masks only)."""
        profile = self.table(table)
        column = profile.column(name)
        if column is None:
            known = ", ".join(c.name for c in profile.columns)
            raise ToolArgumentError(f"unknown column {table}.{name}; columns: {known}")
        if values and column.sensitivity is Sensitivity.SENSITIVE:
            raise ToolArgumentError(
                f"{table}.{name} is sensitive; only its mask and lengths may be shown"
            )
        return column
