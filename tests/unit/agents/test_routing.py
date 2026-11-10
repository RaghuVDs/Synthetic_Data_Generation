from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from synthgen.agents import routing as r
from synthgen.agents.state import JobState, checkpoint_types, initial_state
from synthgen.config import LoopCaps, load_settings
from synthgen.domain.intent import TestIntent
from synthgen.domain.job import (
    Approval,
    ApprovalDecision,
    JobStatus,
    LoopCounters,
    ShareEdit,
    UserRequest,
)
from synthgen.domain.patch import SpecPatch
from synthgen.domain.report import CheckFamily, CheckResult, CheckStatus, ValidationReport
from synthgen.spec.models import SpecError

ROOT = Path(__file__).resolve().parents[3]
CAPS: LoopCaps = load_settings(ROOT / "config" / "settings.yaml", {}).agent.caps
NOW = datetime(2026, 11, 10, tzinfo=UTC)


def state(**changes: Any) -> JobState:
    base = initial_state("job", UserRequest(source="s", tables=("t",), text="x"))
    base.update(changes)  # type: ignore[typeddict-item]
    return base


def counters(**values: int) -> LoopCounters:
    return LoopCounters(**values)


def intent(blocking: bool) -> TestIntent:
    return TestIntent.model_validate(
        {
            "domain": "d",
            "objective": "o",
            "test_type": "rule_testing",
            "entities_of_interest": [],
            "must_include": [],
            "must_exclude": [],
            "volume_hint": None,
            "include_invalid": False,
            "ambiguities": [{"question": "q?", "options": [], "blocking": blocking}],
            "confidence": 0.5,
        }
    )


def approval(decision: ApprovalDecision, feedback: str | None = None) -> Approval:
    edits = (ShareEdit(scenario="s", share=0.1),) if decision is ApprovalDecision.EDITED else ()
    return Approval(decision=decision, feedback=feedback, share_edits=edits, decided_at=NOW)


def report(*rows: tuple[CheckFamily, CheckStatus, str | None]) -> ValidationReport:
    return ValidationReport(
        results=tuple(
            CheckResult(family=f, check="c", table="t", metric="m", status=s, scenario=sc)
            for f, s, sc in rows
        )
    )


def spec_error(origin: str) -> SpecError:
    return SpecError.model_validate(
        {"origin": origin, "scenario": None, "path": "/", "message": "bad"}
    )


def test_intent_routes() -> None:
    assert r.after_intent(state(intent=intent(False)), CAPS).next == r.CONTEXT
    paused = r.after_intent(state(intent=intent(True)), CAPS)
    assert (paused.next, paused.status) == (r.INTENT, JobStatus.AWAITING_CLARIFICATION)
    capped = r.after_intent(
        state(intent=intent(True), counters=counters(clarify=CAPS.clarify)), CAPS
    )
    assert (capped.next, capped.status) == (r.FINISH, JobStatus.NEEDS_REVIEW)


def test_planner_always_goes_to_approval() -> None:
    route = r.route_after(r.PLANNER, state(), CAPS)
    assert (route.next, route.status) == (r.APPROVAL, JobStatus.AWAITING_APPROVAL)


@pytest.mark.parametrize(
    ("decision", "feedback", "rounds", "expected"),
    [
        (ApprovalDecision.APPROVED, None, 0, (r.BASELINE, JobStatus.COMPILING)),
        (ApprovalDecision.EDITED, None, 0, (r.BASELINE, JobStatus.COMPILING)),
        (ApprovalDecision.REJECTED, "more fraud", 0, (r.PLANNER, JobStatus.PLANNING)),
        (
            ApprovalDecision.REJECTED,
            "more fraud",
            CAPS.plan_rounds,
            (r.FINISH, JobStatus.NEEDS_REVIEW),
        ),
        (ApprovalDecision.REJECTED, None, 0, (r.FINISH, JobStatus.CANCELLED)),
    ],
)
def test_approval_routes(
    decision: ApprovalDecision, feedback: str | None, rounds: int, expected: tuple[str, JobStatus]
) -> None:
    route = r.after_approval(
        state(approval=approval(decision, feedback), counters=counters(plan_rounds=rounds)), CAPS
    )
    assert (route.next, route.status) == expected


def test_invalid_edit_is_shown_again_until_cap() -> None:
    assert r.after_approval(state(approval_errors=["shares above 1"]), CAPS).next == r.APPROVAL
    capped = state(approval_errors=["x"], counters=counters(plan_rounds=CAPS.plan_rounds))
    assert r.after_approval(capped, CAPS).status is JobStatus.NEEDS_REVIEW


def test_spec_validator_routes() -> None:
    assert r.after_spec_validator(state(), CAPS).next == r.DRY_RUN
    assert (
        r.after_spec_validator(state(spec_errors=[spec_error("overlay")]), CAPS).next == r.OVERLAY
    )
    failed = r.after_spec_validator(state(spec_errors=[spec_error("baseline")]), CAPS)
    assert failed.status is JobStatus.FAILED
    capped = state(
        spec_errors=[spec_error("overlay")], counters=counters(spec_repair=CAPS.spec_repair)
    )
    assert r.after_spec_validator(capped, CAPS).status is JobStatus.NEEDS_REVIEW


@pytest.mark.parametrize(
    ("rows", "repairs", "expected"),
    [
        ([(CheckFamily.INTEGRITY, CheckStatus.PASS, None)], 0, (r.FULL, JobStatus.GENERATING)),
        ([(CheckFamily.FIDELITY, CheckStatus.FAIL, None)], 0, (r.REPAIR, JobStatus.REPAIRING)),
        ([(CheckFamily.SCENARIO, CheckStatus.FAIL, None)], 0, (r.REPAIR, JobStatus.REPAIRING)),
        ([(CheckFamily.INTEGRITY, CheckStatus.FAIL, "big")], 0, (r.REPAIR, JobStatus.REPAIRING)),
        ([(CheckFamily.PRIVACY, CheckStatus.FAIL, "big")], 0, (r.REPAIR, JobStatus.REPAIRING)),
        ([(CheckFamily.INTEGRITY, CheckStatus.FAIL, None)], 0, (r.FINISH, JobStatus.FAILED)),
        ([(CheckFamily.PRIVACY, CheckStatus.FAIL, None)], 0, (r.FINISH, JobStatus.FAILED)),
        (
            [(CheckFamily.FIDELITY, CheckStatus.FAIL, None)],
            CAPS.validation_repair,
            (r.FINISH, JobStatus.NEEDS_REVIEW),
        ),
        ([(CheckFamily.SCENARIO, CheckStatus.SKIPPED, None)], 0, (r.FULL, JobStatus.GENERATING)),
    ],
)
def test_validator_routes(
    rows: list[tuple[CheckFamily, CheckStatus, str | None]],
    repairs: int,
    expected: tuple[str, JobStatus],
) -> None:
    route = r.after_validator(
        state(dry_run_report=report(*rows), counters=counters(validation_repair=repairs)), CAPS
    )
    assert (route.next, route.status) == expected


def test_repair_routes() -> None:
    patch = SpecPatch(patches=[], plan_change_required=False, explanation="tune")
    assert r.after_repair(state(patch=patch), CAPS).next == r.SPEC_VALIDATOR
    replan = patch.model_copy(update={"plan_change_required": True})
    assert r.after_repair(state(patch=replan), CAPS).next == r.APPROVAL
    capped = state(patch=replan, counters=counters(plan_rounds=CAPS.plan_rounds))
    assert r.after_repair(capped, CAPS).status is JobStatus.NEEDS_REVIEW


def test_final_validation_routes() -> None:
    passed = report((CheckFamily.INTEGRITY, CheckStatus.PASS, None))
    failed = report((CheckFamily.FIDELITY, CheckStatus.FAIL, None))
    assert r.after_final_validation(state(final_report=passed), CAPS).next == r.REPORTER
    assert (
        r.after_final_validation(state(final_report=failed), CAPS).status is JobStatus.NEEDS_REVIEW
    )
    assert r.route_after(r.REPORTER, state(), CAPS).status is JobStatus.COMPLETED


def test_terminal_status_set_by_a_node_goes_to_finish() -> None:
    route = r.route_after(r.PROFILER, state(status=JobStatus.FAILED, error="disk full"), CAPS)
    assert (route.next, route.status, route.reason) == (r.FINISH, JobStatus.FAILED, "disk full")


def test_every_work_node_has_a_router_and_status() -> None:
    assert set(r.ROUTERS) == set(r.NODE_STATUS)
    assert len(r.ROUTERS) == 15


def test_checkpoint_types_cover_nested_models() -> None:
    names = {t.__name__ for t in checkpoint_types()}
    assert {"UserRequest", "RowTarget", "TestIntent", "Ambiguity", "TestType", "JobStatus"} <= names
    assert {"ScenarioSpec", "ColumnOverride", "CategoricalGen", "Route", "CheckResult"} <= names
