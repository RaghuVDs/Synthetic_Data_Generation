"""Node 5, LLM: build the ScenarioPlan. M3 adds planner tools and the feasibility check."""

from __future__ import annotations

from synthgen.agents.nodes.common import as_json, cards
from synthgen.agents.nodes.deps import NodeDeps, StateUpdate
from synthgen.agents.prompts.loader import render
from synthgen.agents.state import JobState
from synthgen.domain.errors import SynthgenError
from synthgen.domain.job import ApprovalDecision, ArtifactKind
from synthgen.domain.plan import ScenarioPlan

ROLE = "scenario_planner"


def run(state: JobState, deps: NodeDeps) -> StateUpdate:
    """Plan scenarios; a rejected plan's feedback starts a new revision round."""
    intent, semantic_map = state["intent"], state["semantic_map"]
    if intent is None or semantic_map is None:
        raise SynthgenError("planner ran before intent and semantic map were ready")
    approval = state["approval"]
    replanning = approval is not None and approval.decision is ApprovalDecision.REJECTED
    counters = state["counters"]
    if replanning:
        counters = counters.model_copy(update={"plan_rounds": counters.plan_rounds + 1})
    feedback = approval.feedback if replanning and approval and approval.feedback else "none"
    _, messages = render(
        ROLE,
        {
            "INTENT": as_json(intent),
            "SEMANTIC_MAP": as_json(semantic_map),
            "PROFILE_CARDS": as_json(cards(state, deps)),
            "PREVIOUS_PLAN": as_json(state["plan"]) if replanning and state["plan"] else "none",
            "FEEDBACK": feedback,
        },
    )
    plan = deps.llm.complete(messages, ScenarioPlan)
    deps.artifacts.put(state["job_id"], ArtifactKind.PLAN, plan)
    return {"plan": plan, "approval": None, "counters": counters}
