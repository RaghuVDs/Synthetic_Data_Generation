"""Node 5, LLM with tools: build the ScenarioPlan, then code runs the feasibility check.

Groq cannot combine tools with structured output (SPEC 4.2), so the planner first explores with
tool-calling turns, capped at `planner_tool_calls` (PLAN Q5), and then makes one structured call
that receives the tool results as a data block.
"""

from __future__ import annotations

import json

from synthgen.agents.llm.client import Message
from synthgen.agents.llm.structured import CallContext
from synthgen.agents.meaning import check_plan
from synthgen.agents.nodes.common import (
    PLANNER_ROLE,
    as_json,
    ask,
    cards,
    column_roles,
    job_columns,
    load_profile,
    semantic_summary,
)
from synthgen.agents.nodes.deps import NodeDeps, StateUpdate
from synthgen.agents.prompts.loader import render
from synthgen.agents.state import JobState
from synthgen.agents.tools.context import ToolContext
from synthgen.agents.tools.registry import ToolRegistry
from synthgen.domain.errors import SynthgenError
from synthgen.domain.job import ApprovalDecision, ArtifactKind
from synthgen.domain.plan import ScenarioPlan
from synthgen.domain.semantic import ColumnRole, SemanticMap
from synthgen.profiling.card import LlmSafeProfile, card_text
from synthgen.scenarios.feasibility import check_feasibility
from synthgen.scenarios.profile_derived import candidates_text, profile_candidates

ROLE = PLANNER_ROLE
EXPLORE_INSTRUCTION = (
    "Investigate with the tools where it helps you quantify scenarios. "
    "When you know enough, reply with a short note and no tool calls."
)


def role_bearing(
    all_cards: list[LlmSafeProfile], semantic: SemanticMap, limit: int
) -> list[LlmSafeProfile]:
    """Above `limit` tables, keep only columns with a role other than `other` (SPEC 4.4)."""
    if len(all_cards) <= limit:
        return all_cards
    keep = {
        (t.table, c.name)
        for t in semantic.tables
        for c in t.columns
        if c.role is not ColumnRole.OTHER
    }
    return [
        card.model_copy(
            update={"columns": [c for c in card.columns if (card.table, c.name) in keep]}
        )
        for card in all_cards
    ]


def explore(
    state: JobState, deps: NodeDeps, registry: ToolRegistry, blocks: dict[str, str]
) -> list[dict[str, str]]:
    """Tool-calling turns until the model stops or the call cap is reached."""
    version, messages = render(ROLE, blocks)
    conversation = [*messages, Message(role="user", content=EXPLORE_INSTRUCTION)]
    ctx = CallContext(job_id=state["job_id"], node=ROLE, prompt_version=version)
    cap = deps.settings.agent.caps.planner_tool_calls
    findings: list[dict[str, str]] = []
    while len(findings) < cap:
        turn = deps.llm.tool_turn(
            ctx, conversation, registry.definitions(), deps.settings.llm.planner_temperature
        )
        calls = turn.tool_calls[: cap - len(findings)]
        if not calls:
            break
        conversation.append(
            Message(role="assistant", content=turn.content, tool_calls=tuple(calls))
        )
        for call in calls:
            result = registry.call(call.name, call.arguments)
            conversation.append(
                Message(role="tool", content=result, tool_call_id=call.id, name=call.name)
            )
            findings.append({"tool": call.name, "arguments": call.arguments, "result": result})
    return findings


def run(state: JobState, deps: NodeDeps) -> StateUpdate:
    """Plan scenarios; a rejected plan's feedback starts a new revision round."""
    intent, semantic_map, table_set = state["intent"], state["semantic_map"], state["table_set"]
    if intent is None or semantic_map is None or table_set is None:
        raise SynthgenError("planner ran before intent, semantic map and table set were ready")
    approval = state["approval"]
    replanning = approval is not None and approval.decision is ApprovalDecision.REJECTED
    counters = state["counters"]
    if replanning:
        counters = counters.model_copy(update={"plan_rounds": counters.plan_rounds + 1})
    columns = job_columns(state, deps)
    profiles = {name: load_profile(state, deps, name) for name in table_set.order}
    limit = deps.settings.profiling.mapper_batch_tables
    blocks = {
        "INTENT": as_json(intent),
        "SEMANTIC_MAP": semantic_summary(semantic_map),
        "PROFILE_CARDS": "\n\n".join(
            card_text(c, column_roles(semantic_map, c.table))
            for c in role_bearing(cards(state, deps), semantic_map, limit)
        ),
        "PROFILE_CANDIDATES": candidates_text(
            profile_candidates(profiles, semantic_map, intent.include_invalid)
        ),
        "TEMPLATES": as_json(deps.templates.search(intent.domain, intent.must_include)),
        "PREVIOUS_PLAN": as_json(state["plan"]) if replanning and state["plan"] else "none",
        "FEEDBACK": approval.feedback if replanning and approval and approval.feedback else "none",
    }
    with deps.open_source(state["request"].source) as source:
        context = ToolContext(
            state["job_id"], source, profiles, deps.templates, deps.settings.profiling
        )
        registry = ToolRegistry(
            context, deps.tool_cache, deps.settings.agent.tool_result_max_tokens
        )
        blocks["TOOL_RESULTS"] = json.dumps(explore(state, deps, registry, blocks)) or "[]"
    plan = ask(
        state,
        deps,
        ROLE,
        blocks,
        ScenarioPlan,
        meaning=lambda value: check_plan(value, columns, deps.templates.ids),
    )
    deps.artifacts.put(state["job_id"], ArtifactKind.PLAN, plan)
    feasibility = check_feasibility(plan, semantic_map, columns)
    return {"plan": plan, "feasibility": feasibility, "approval": None, "counters": counters}
