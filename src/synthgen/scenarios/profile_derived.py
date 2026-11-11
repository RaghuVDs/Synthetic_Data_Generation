"""Profile-derived scenario candidates (SPEC 7.1 source 2): no domain knowledge needed.

Candidates are suggestions for the planner, described without any column values.
"""

from __future__ import annotations

from collections.abc import Mapping

from synthgen.domain.base import StrictModel
from synthgen.domain.metadata import Sensitivity
from synthgen.domain.plan import Primitive, ScenarioCategory
from synthgen.domain.semantic import ColumnRole, SemanticMap
from synthgen.profiling.models import ColumnProfile, TableProfile

MAX_CANDIDATES = 12
SKEW_TOP_SHARE = 0.05
"""A key is hot when its most loaded parent holds more than this share of children."""


class Candidate(StrictModel):
    """One scenario the profile suggests."""

    name: str
    category: ScenarioCategory
    primitive: Primitive
    table: str
    column: str
    description: str


def profile_candidates(
    profiles: Mapping[str, TableProfile], semantic: SemanticMap, include_invalid: bool
) -> list[Candidate]:
    """Boundaries, rare values, peak times, hot keys and (when asked) rule breaks."""
    found: list[Candidate] = []
    for table, profile in profiles.items():
        semantics = semantic.for_table(table)
        roles = {c.name: c.role for c in semantics.columns} if semantics else {}
        for column in profile.columns:
            if column.sensitivity is not Sensitivity.SENSITIVE:
                found += _column_candidates(table, column, roles.get(column.name), include_invalid)
        found += _skew_candidates(table, profile)
    return found[:MAX_CANDIDATES]


def _column_candidates(
    table: str, column: ColumnProfile, role: ColumnRole | None, include_invalid: bool
) -> list[Candidate]:
    rules = [
        (
            bool(column.numeric) and role == ColumnRole.MEASURE,
            "boundary",
            Primitive.BOUNDARY_VALUE,
            f"{column.name} at its observed minimum and maximum",
        ),
        (
            bool(column.categorical and column.categorical.rare_count.value)
            and role in (ColumnRole.CATEGORY, ColumnRole.STATUS),
            "rare",
            Primitive.MIX_SHIFT,
            f"values of {column.name} that are rare in the source",
        ),
        (
            bool(column.temporal) and role == ColumnRole.EVENT_TIME,
            "temporal",
            Primitive.TIMING_SHIFT,
            f"{column.name} concentrated in its busiest hours",
        ),
        (
            include_invalid and not column.null_rate.value,
            "invalid",
            Primitive.RULE_BREAK,
            f"missing values in {column.name}, which is never empty in the source",
        ),
    ]
    return [
        _candidate(table, column.name, category, primitive, description)
        for applies, category, primitive, description in rules
        if applies
    ]


def _skew_candidates(table: str, profile: TableProfile) -> list[Candidate]:
    found = []
    for shape in profile.relationships:
        total = sum(children * parents for children, parents in shape.counts.value)
        top = max((children for children, _ in shape.counts.value), default=0)
        if total and top / total > SKEW_TOP_SHARE:
            description = f"a few {shape.parent_table} rows owning most {table} rows"
            found.append(
                _candidate(
                    table, shape.child_column, "volume_skew", Primitive.VOLUME_SKEW, description
                )
            )
    return found


def _candidate(
    table: str, column: str, category: str, primitive: Primitive, description: str
) -> Candidate:
    return Candidate(
        name=f"{category}_{table}_{column}".lower(),
        category=ScenarioCategory(category),
        primitive=primitive,
        table=table,
        column=column,
        description=description,
    )


def candidates_text(candidates: list[Candidate]) -> str:
    """One compact line per candidate for prompts."""
    return (
        "\n".join(
            f"- {c.name} | {c.category} | {c.primitive} on {c.table}.{c.column}: {c.description}"
            for c in candidates
        )
        or "none"
    )
