"""Every edge of the graph as a pure function (SPEC 3.3). Code routes; the LLM never does.

A router sees the state after its node ran and returns the next node, the status the job shows
until that node finishes, and a reason when the job stops. Each loop compares its counter with
its cap before going round again; the node that runs the loop body increments the counter.
"""

from __future__ import annotations

from collections.abc import Callable

from synthgen.agents.state import JobState, Route
from synthgen.config import LoopCaps
from synthgen.domain.job import ApprovalDecision, JobStatus
from synthgen.domain.report import CheckFamily

INTENT = "intent_analyst"
CONTEXT = "context_builder"
PROFILER = "profiler"
MAPPER = "semantic_mapper"
PLANNER = "scenario_planner"
APPROVAL = "approval_gate"
BASELINE = "baseline_compiler"
OVERLAY = "overlay_writer"
SPEC_VALIDATOR = "spec_validator"
DRY_RUN = "dry_run"
VALIDATOR = "validator"
REPAIR = "repair_analyst"
FULL = "full_generation"
FINAL = "final_validation"
REPORTER = "reporter"
FINISH = "finish"

NODE_STATUS = {
    INTENT: JobStatus.ANALYSING,
    CONTEXT: JobStatus.ANALYSING,
    PROFILER: JobStatus.PROFILING,
    MAPPER: JobStatus.PLANNING,
    PLANNER: JobStatus.PLANNING,
    APPROVAL: JobStatus.AWAITING_APPROVAL,
    BASELINE: JobStatus.COMPILING,
    OVERLAY: JobStatus.COMPILING,
    SPEC_VALIDATOR: JobStatus.COMPILING,
    DRY_RUN: JobStatus.DRY_RUN,
    VALIDATOR: JobStatus.DRY_RUN,
    REPAIR: JobStatus.REPAIRING,
    FULL: JobStatus.GENERATING,
    FINAL: JobStatus.VALIDATING,
    REPORTER: JobStatus.VALIDATING,
}
HARD_GATES = (CheckFamily.INTEGRITY, CheckFamily.PRIVACY)


def go(node: str, status: JobStatus | None = None) -> Route:
    """Continue to `node`, showing its working status unless another is given."""
    return Route(next=node, status=status or NODE_STATUS[node], reason=None)


def stop(status: JobStatus, reason: str | None) -> Route:
    """End the job with a terminal status."""
    return Route(next=FINISH, status=status, reason=reason)


def after_intent(state: JobState, caps: LoopCaps) -> Route:
    """Ask again while blocking questions remain and rounds are left (SPEC 3.3 rows 1-2)."""
    intent = state["intent"]
    if intent is None or not intent.blocking_ambiguities:
        return go(CONTEXT)
    if state["counters"].clarify < caps.clarify:
        return go(INTENT, JobStatus.AWAITING_CLARIFICATION)
    return stop(JobStatus.NEEDS_REVIEW, f"questions still open after {caps.clarify} rounds")


def after_approval(state: JobState, caps: LoopCaps) -> Route:
    """Approve, re-plan on feedback, re-show an invalid edit, or cancel (rows 4-6)."""
    rounds_left = state["counters"].plan_rounds < caps.plan_rounds
    if state["approval_errors"]:
        return (
            go(APPROVAL)
            if rounds_left
            else stop(JobStatus.NEEDS_REVIEW, "plan edits still invalid")
        )
    approval = state["approval"]
    if approval is None or approval.decision is not ApprovalDecision.REJECTED:
        return go(BASELINE)
    if not approval.has_feedback:
        return stop(JobStatus.CANCELLED, "plan rejected without feedback")
    if rounds_left:
        return go(PLANNER)
    return stop(JobStatus.NEEDS_REVIEW, f"plan rejected after {caps.plan_rounds} revisions")


def after_spec_validator(state: JobState, caps: LoopCaps) -> Route:
    """Dry-run a valid spec; send overlay errors back; fail on baseline errors (rows 7-8)."""
    errors = state["spec_errors"]
    if not errors:
        return go(DRY_RUN)
    baseline = [e for e in errors if e.origin == "baseline"]
    if baseline:
        return stop(JobStatus.FAILED, f"baseline spec invalid: {baseline[0].message}")
    if state["counters"].spec_repair < caps.spec_repair:
        return go(OVERLAY)
    return stop(JobStatus.NEEDS_REVIEW, f"spec still invalid after {caps.spec_repair} repairs")


def after_validator(state: JobState, caps: LoopCaps) -> Route:
    """Generate when the dry run passes; repair what an overlay can fix; fail otherwise (9-12)."""
    report = state["dry_run_report"]
    if report is None or report.passed:
        return go(FULL)
    untraced = [f for f in report.failures() if f.family in HARD_GATES and f.scenario is None]
    if untraced:
        f = untraced[0]
        return stop(JobStatus.FAILED, f"{f.family} check {f.check} failed on {f.table}.{f.column}")
    if state["counters"].validation_repair < caps.validation_repair:
        return go(REPAIR)
    return stop(
        JobStatus.NEEDS_REVIEW, f"dry run still failing after {caps.validation_repair} repairs"
    )


def after_repair(state: JobState, caps: LoopCaps) -> Route:
    """Re-validate a patched spec, or return to approval when the plan itself must change."""
    patch = state["patch"]
    if patch is None or not patch.plan_change_required:
        return go(SPEC_VALIDATOR)
    if state["counters"].plan_rounds < caps.plan_rounds:
        return go(APPROVAL)
    return stop(JobStatus.NEEDS_REVIEW, "repair needs a plan change but no plan rounds are left")


def after_final_validation(state: JobState, caps: LoopCaps) -> Route:
    """Report on clean output; anything else needs review (rows 14-15)."""
    del caps
    report = state["final_report"]
    if report is not None and report.passed:
        return go(REPORTER)
    return stop(JobStatus.NEEDS_REVIEW, "final validation failed")


def _then(node: str) -> Callable[[JobState, LoopCaps], Route]:
    return lambda _state, _caps: go(node)


ROUTERS: dict[str, Callable[[JobState, LoopCaps], Route]] = {
    INTENT: after_intent,
    CONTEXT: _then(PROFILER),
    PROFILER: _then(MAPPER),
    MAPPER: _then(PLANNER),
    PLANNER: _then(APPROVAL),
    APPROVAL: after_approval,
    BASELINE: _then(OVERLAY),
    OVERLAY: _then(SPEC_VALIDATOR),
    SPEC_VALIDATOR: after_spec_validator,
    DRY_RUN: _then(VALIDATOR),
    VALIDATOR: after_validator,
    REPAIR: after_repair,
    FULL: _then(FINAL),
    FINAL: after_final_validation,
    REPORTER: lambda _state, _caps: stop(JobStatus.COMPLETED, None),
}


def route_after(node: str, state: JobState, caps: LoopCaps) -> Route:
    """The route after `node`. A node that already ended the job goes straight to finish."""
    if state["status"].is_terminal:
        return stop(state["status"], state["error"])
    return ROUTERS[node](state, caps)
