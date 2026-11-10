"""The planner's read-only tools: definitions, validation, caching and size cap (SPEC 6)."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from synthgen.agents.llm.client import ToolDef
from synthgen.agents.llm.schema import strict_schema
from synthgen.agents.tools import profile_tools as pt
from synthgen.agents.tools import stats_tools as st
from synthgen.agents.tools.context import ToolContext
from synthgen.domain.base import StrictModel
from synthgen.domain.errors import ConnectorError, MetadataError, ToolArgumentError

CHARS_PER_TOKEN = 4
TRUNCATED = "...[truncated]"
Result = dict[str, Any] | list[Any]  # Any: JSON-compatible tool output


class TemplateSearchArgs(StrictModel):
    """Arguments of search_scenario_templates."""

    domain: str
    keywords: list[str]


class TemplateArgs(StrictModel):
    """Arguments of get_scenario_template."""

    template_id: str


@dataclass(frozen=True)
class ToolSpec:
    """One tool: its description, argument model and implementation."""

    name: str
    description: str
    args: type[BaseModel]
    run: Callable[[ToolContext, Any], Result]  # Any: the tool's own argument model


def _search(ctx: ToolContext, args: TemplateSearchArgs) -> Result:
    return [s.model_dump() for s in ctx.templates.search(args.domain, args.keywords)]


def _template(ctx: ToolContext, args: TemplateArgs) -> Result:
    template = ctx.templates.get(args.template_id)
    if template is None:
        raise ToolArgumentError(f"unknown template {args.template_id!r}")
    return template.model_dump(mode="json")


TOOLS = (
    ToolSpec(
        "get_column_profile",
        "Full statistics of one column (masks only if sensitive).",
        pt.ColumnArgs,
        pt.get_column_profile,
    ),
    ToolSpec(
        "get_conditional_stats",
        "A target column summarised per category of another column.",
        st.ConditionalArgs,
        st.get_conditional_stats,
    ),
    ToolSpec(
        "get_relationship_profile",
        "Children per parent, key skew and orphan rate.",
        pt.RelationshipArgs,
        pt.get_relationship_profile,
    ),
    ToolSpec(
        "get_behaviour_profile",
        "Events per entity, inter-arrival times and bursts.",
        st.BehaviourArgs,
        st.get_behaviour_profile,
    ),
    ToolSpec(
        "get_class_separation",
        "Columns ranked by how well they separate a label's classes.",
        st.SeparationArgs,
        st.get_class_separation,
    ),
    ToolSpec(
        "search_scenario_templates",
        "Find scenario templates by domain and keywords.",
        TemplateSearchArgs,
        _search,
    ),
    ToolSpec(
        "get_scenario_template", "Fetch one scenario template in full.", TemplateArgs, _template
    ),
)


class ToolResultCache:
    """Tool results per job, so repeated calls in later plan rounds cost nothing."""

    def __init__(self) -> None:
        """Start empty."""
        self._results: dict[tuple[str, str, str], str] = {}

    def get(self, key: tuple[str, str, str]) -> str | None:
        """A cached result, or None."""
        return self._results.get(key)

    def put(self, key: tuple[str, str, str], result: str) -> None:
        """Remember a result."""
        self._results[key] = result


class ToolRegistry:
    """Runs tools for one job. Every problem comes back as an error string, never an exception."""

    def __init__(self, ctx: ToolContext, cache: ToolResultCache, max_tokens: int) -> None:
        """Bind the job context, the shared cache and the result size cap."""
        self._ctx = ctx
        self._cache = cache
        self._max_chars = max_tokens * CHARS_PER_TOKEN
        self._tools = {t.name: t for t in TOOLS}

    def definitions(self) -> list[ToolDef]:
        """Tool definitions for the model."""
        return [
            ToolDef(name=t.name, description=t.description, parameters=strict_schema(t.args))
            for t in TOOLS
        ]

    def call(self, name: str, arguments: str) -> str:
        """Run one tool call and return its (capped) JSON result or an error string."""
        tool = self._tools.get(name)
        if tool is None:
            return f"error: unknown tool {name!r}"
        try:
            args = tool.args.model_validate_json(arguments)
        except ValidationError as exc:
            return f"error: invalid arguments: {exc.errors()[0]['msg']} at {exc.errors()[0]['loc']}"
        key = (self._ctx.job_id, name, args.model_dump_json())
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            text = json.dumps(tool.run(self._ctx, args), default=str)
        except (ToolArgumentError, MetadataError, ConnectorError) as exc:
            return f"error: {exc}"
        result = (
            text
            if len(text) <= self._max_chars
            else text[: self._max_chars - len(TRUNCATED)] + TRUNCATED
        )
        self._cache.put(key, result)
        return result
