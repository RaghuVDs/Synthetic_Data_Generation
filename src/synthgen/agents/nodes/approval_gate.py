"""Node 6, human: approve, edit or reject the plan. Only this gate can start generation."""

from __future__ import annotations

from langgraph.types import interrupt
from pydantic import ValidationError

from synthgen.agents.nodes.deps import NodeDeps, StateUpdate
from synthgen.agents.state import JobState
from synthgen.domain.errors import SynthgenError
from synthgen.domain.job import Approval, ApprovalDecision, ArtifactKind
from synthgen.domain.pending import PendingApproval
from synthgen.domain.plan import ScenarioPlan


def apply_edits(plan: ScenarioPlan, approval: Approval) -> ScenarioPlan:
    """The plan with scenarios removed and shares changed; raises ValueError when invalid."""
    names = {s.name for s in plan.scenarios}
    unknown = sorted(
        ({e.scenario for e in approval.share_edits} | set(approval.removed_scenarios)) - names
    )
    if unknown:
        raise ValueError(f"unknown scenarios: {', '.join(unknown)}")
    shares = {e.scenario: e.share for e in approval.share_edits}
    kept = [
        s.model_copy(update={"share": shares.get(s.name, s.share)})
        for s in plan.scenarios
        if s.name not in approval.removed_scenarios
    ]
    return ScenarioPlan.model_validate({"scenarios": kept, "assumptions": plan.assumptions})


def run(state: JobState, deps: NodeDeps) -> StateUpdate:
    """Pause for the user's decision unless auto-approve applies; validate any edits."""
    plan = state["plan"]
    if plan is None:
        raise SynthgenError("approval gate reached without a plan")
    job_id = state["job_id"]
    counters = state["counters"]
    patch = state["patch"]
    review_again = bool(state["approval_errors"]) or bool(patch and patch.plan_change_required)
    if review_again:
        counters = counters.model_copy(update={"plan_rounds": counters.plan_rounds + 1})
    version = deps.artifacts.versions(job_id, ArtifactKind.PLAN)[-1]
    if state["request"].auto_approve and not state["approval_errors"]:
        approval = Approval(decision=ApprovalDecision.APPROVED, decided_at=deps.clock())
    else:
        pending = PendingApproval(
            kind="approval",
            plan=plan,
            plan_version=version,
            errors=state["approval_errors"],
            explanation=patch.explanation if patch and patch.plan_change_required else None,
        )
        approval = Approval.model_validate(interrupt(pending.model_dump(mode="json")))
    deps.jobs.record_approval(job_id, version, approval)
    update: StateUpdate = {
        "approval": approval,
        "approval_errors": [],
        "counters": counters,
        "patch": None,
    }
    if approval.decision is ApprovalDecision.EDITED:
        try:
            edited = apply_edits(plan, approval)
        except (ValueError, ValidationError) as exc:
            return {**update, "approval_errors": [str(exc)]}
        deps.artifacts.put(job_id, ArtifactKind.PLAN, edited)
        update["plan"] = edited
    return update
