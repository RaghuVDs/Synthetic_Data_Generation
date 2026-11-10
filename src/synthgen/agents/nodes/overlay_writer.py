"""Node 8, LLM: approved scenarios to ScenarioSpec overlays; repairs listed spec errors."""

from __future__ import annotations

import json

from synthgen.agents.nodes.common import as_json, cards
from synthgen.agents.nodes.deps import NodeDeps, StateUpdate
from synthgen.agents.prompts.loader import render
from synthgen.agents.state import JobState
from synthgen.domain.errors import SynthgenError
from synthgen.domain.job import ArtifactKind
from synthgen.domain.plan import ScenarioPlan
from synthgen.spec.models import GenerationSpec, OverlaySet, ScenarioSpec

ROLE = "overlay_writer"


def column_list(baseline: GenerationSpec) -> str:
    """Table, column, type and sensitivity only: no profile-derived values reach the model."""
    return json.dumps(
        [
            {"table": t.name, "column": c.name, "type": c.type, "sensitive": c.sensitive}
            for t in baseline.tables
            for c in t.columns
        ]
    )


def run(state: JobState, deps: NodeDeps) -> StateUpdate:
    """Write one overlay per approved scenario; a repair pass counts against the cap."""
    plan = state["plan"]
    if plan is None:
        raise SynthgenError("overlay writer ran without an approved plan")
    job_id = state["job_id"]
    baseline = deps.artifacts.get(job_id, ArtifactKind.BASELINE_SPEC, GenerationSpec)
    counters = state["counters"]
    if state["spec_errors"]:
        counters = counters.model_copy(update={"spec_repair": counters.spec_repair + 1})
    _, messages = render(
        ROLE,
        {
            "APPROVED_PLAN": as_json(plan),
            "COLUMNS": column_list(baseline),
            "PROFILE_CARDS": as_json(cards(state, deps)),
            "SPEC_ERRORS": as_json(state["spec_errors"]) if state["spec_errors"] else "none",
        },
    )
    overlays = deps.llm.complete(messages, OverlaySet)
    return {"overlays": approved_shares(overlays, plan), "spec_ref": None, "counters": counters}


def approved_shares(overlays: OverlaySet, plan: ScenarioPlan) -> list[ScenarioSpec]:
    """Overlays carry the approved share and category; the approval gate's numbers are binding."""
    approved = {s.name: s for s in plan.scenarios}
    return [
        o.model_copy(
            update={"share": approved[o.name].share, "category": approved[o.name].category}
        )
        if o.name in approved
        else o
        for o in overlays.scenarios
    ]
