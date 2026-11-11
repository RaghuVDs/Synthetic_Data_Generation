"""Helpers shared by nodes: spec references, profile cards and JSON blocks for prompts."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import TypeVar

from pydantic import BaseModel

from synthgen.agents.llm.structured import CallContext, no_meaning_errors
from synthgen.agents.nodes.deps import NodeDeps
from synthgen.agents.prompts.loader import render
from synthgen.agents.state import JobState
from synthgen.domain.errors import SynthgenError
from synthgen.domain.job import ArtifactKind
from synthgen.domain.semantic import SemanticMap
from synthgen.profiling.card import LlmSafeProfile, build_card, card_text
from synthgen.profiling.models import TableProfile
from synthgen.spec.models import GenerationSpec

SPEC_REF_PREFIX = "spec/"
PLANNER_ROLE = "scenario_planner"
ModelT = TypeVar("ModelT", bound=BaseModel)


def spec_ref(version: int) -> str:
    """Reference to a stored merged spec."""
    return f"{SPEC_REF_PREFIX}{version}"


def load_spec(state: JobState, deps: NodeDeps) -> GenerationSpec:
    """The merged spec the state points at."""
    ref = state["spec_ref"]
    if ref is None or not ref.startswith(SPEC_REF_PREFIX):
        raise SynthgenError(f"no merged spec to use (spec_ref={ref!r})")
    version = int(ref.removeprefix(SPEC_REF_PREFIX))
    return deps.artifacts.get(state["job_id"], ArtifactKind.SPEC, GenerationSpec, version)


def load_profile(state: JobState, deps: NodeDeps, table: str) -> TableProfile:
    """The exact profile version this job uses for a table."""
    return deps.profiles.get(
        state["request"].source, table, state["profile_refs"][table], TableProfile
    )


def cards(state: JobState, deps: NodeDeps) -> list[LlmSafeProfile]:
    """LLM-safe profile cards for every table in the job; the only profile view agents get."""
    table_set = state["table_set"]
    if table_set is None:
        raise SynthgenError("profile cards requested before the table set is known")
    with deps.open_source(state["request"].source) as source:
        return [
            build_card(
                load_profile(state, deps, name),
                source.metadata.table(name),
                table_set.relationships,
                deps.settings.profiling,
            )
            for name in table_set.order
        ]


def job_columns(state: JobState, deps: NodeDeps) -> dict[str, list[str]]:
    """Column names of every table in the job, for meaning checks."""
    table_set = state["table_set"]
    if table_set is None:
        raise SynthgenError("columns requested before the table set is known")
    with deps.open_source(state["request"].source) as source:
        return {t: [c.name for c in source.metadata.table(t).columns] for t in table_set.order}


def ask(  # noqa: PLR0913 - the full description of one structured call
    state: JobState,
    deps: NodeDeps,
    role: str,
    blocks: dict[str, str],
    schema: type[ModelT],
    *,
    meaning: Callable[[ModelT], list[str]] = no_meaning_errors,
) -> ModelT:
    """Render the role's prompt and get an answer that passed all three gates."""
    version, messages = render(role, blocks)
    llm = deps.settings.llm
    temperature = llm.planner_temperature if role == PLANNER_ROLE else llm.temperature
    ctx = CallContext(job_id=state["job_id"], node=role, prompt_version=version)
    return deps.llm.ask(ctx, messages, schema, temperature, meaning)


def as_json(value: BaseModel | Sequence[BaseModel]) -> str:
    """Compact JSON for a prompt block (no spaces, no null fields)."""
    if isinstance(value, BaseModel):
        return value.model_dump_json(exclude_none=True)
    return json.dumps(
        [v.model_dump(mode="json", exclude_none=True) for v in value], separators=(",", ":")
    )


def cards_text(state: JobState, deps: NodeDeps, roles: SemanticMap | None = None) -> str:
    """Every table's card in the compact text form, with column roles when a map is given."""
    return "\n\n".join(card_text(c, column_roles(roles, c.table)) for c in cards(state, deps))


def column_roles(semantic: SemanticMap | None, table: str) -> dict[str, str] | None:
    """Column name -> role for one table, when the map covers it."""
    found = semantic.for_table(table) if semantic else None
    return {c.name: str(c.role) for c in found.columns} if found else None


def semantic_summary(semantic: SemanticMap) -> str:
    """The map without per-column roles (those ride on the cards): grain, keys, rules."""
    tables = [
        t.model_dump(mode="json", exclude={"columns"}, exclude_none=True) for t in semantic.tables
    ]
    rules = [r.model_dump(mode="json") for r in semantic.business_rules]
    return json.dumps({"tables": tables, "business_rules": rules}, separators=(",", ":"))
