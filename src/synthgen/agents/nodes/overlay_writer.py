"""Node 8, LLM: approved scenarios to ScenarioSpec overlays; repairs listed spec errors."""

from __future__ import annotations

from synthgen.agents.meaning import check_overlays
from synthgen.agents.nodes.common import as_json, ask, cards_text, job_columns
from synthgen.agents.nodes.deps import NodeDeps, StateUpdate
from synthgen.agents.state import JobState
from synthgen.domain.errors import SynthgenError
from synthgen.domain.feasibility import FeasibilityStatus
from synthgen.domain.plan import ScenarioPlan
from synthgen.spec.models import OverlaySet, ScenarioSpec

ROLE = "overlay_writer"


def feasibility_notes(state: JobState, names: list[str]) -> str:
    """What each partly expressible scenario cannot do, so the writer leaves it out."""
    notes = [
        f"- {f.scenario}: leave out {', '.join(f.not_covered)} ({f.reason})"
        for f in state["feasibility"]
        if f.scenario in names and f.status is FeasibilityStatus.PARTIAL
    ]
    return "\n".join(notes) or "none"


def run(state: JobState, deps: NodeDeps) -> StateUpdate:
    """Write one overlay per approved scenario; a repair pass counts against the cap."""
    plan = state["plan"]
    if plan is None:
        raise SynthgenError("overlay writer ran without an approved plan")
    counters = state["counters"]
    if state["spec_errors"]:
        counters = counters.model_copy(update={"spec_repair": counters.spec_repair + 1})
    plan = generable(plan, state)
    expected = [s.name for s in plan.scenarios]
    columns = job_columns(state, deps)
    overlays = ask(
        state,
        deps,
        ROLE,
        {
            "APPROVED_PLAN": as_json(plan),
            "NOT_EXPRESSIBLE_PARTS": feasibility_notes(state, expected),
            "PROFILE_CARDS": cards_text(state, deps, state["semantic_map"], plan_columns(plan)),
            "SPEC_ERRORS": as_json(state["spec_errors"]) if state["spec_errors"] else "none",
        },
        OverlaySet,
        meaning=lambda value: check_overlays(value, expected, columns),
    )
    return {"overlays": approved_shares(overlays, plan), "spec_ref": None, "counters": counters}


def plan_columns(plan: ScenarioPlan) -> set[tuple[str, str]]:
    """Every (table, column) a plan's levers or signatures touch."""
    found = {(lv.table, lv.column) for s in plan.scenarios for lv in s.levers}
    return found | {(a.table, a.column) for s in plan.scenarios for a in s.signature}


def generable(plan: ScenarioPlan, state: JobState) -> ScenarioPlan:
    """The approved plan without scenarios the feasibility check found not expressible."""
    blocked = {
        f.scenario for f in state["feasibility"] if f.status is FeasibilityStatus.NOT_EXPRESSIBLE
    }
    return plan.model_copy(
        update={"scenarios": [s for s in plan.scenarios if s.name not in blocked]}
    )


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
