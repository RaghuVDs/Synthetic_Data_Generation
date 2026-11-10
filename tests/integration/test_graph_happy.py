from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synthgen.app import Application, JobView
from synthgen.domain.job import Approval, ApprovalDecision, JobStatus, ShareEdit, UserRequest
from synthgen.domain.pending import PendingApproval, PendingClarification

from .helpers import fixture_set, skeleton

NOW = datetime(2026, 11, 10, tzinfo=UTC)
EnvFactory = Callable[..., dict[str, str]]
AppFactory = Callable[[dict[str, str]], Application]


def request(**changes: Any) -> UserRequest:
    fields: dict[str, Any] = {
        "source": "finance",
        "tables": ("customers", "accounts"),
        "text": "customers with unusually high balances",
        "auto_approve": True,
    }
    fields.update(changes)
    return UserRequest.model_validate(fields)


def start(make_app: AppFactory, env: dict[str, str], **changes: Any) -> tuple[Application, JobView]:
    app = make_app(env)
    return app, app.start_job(request(**changes))


def decide(decision: ApprovalDecision, **changes: Any) -> Approval:
    return Approval(decision=decision, decided_at=NOW, **changes)


def test_auto_approved_job_completes(make_app: AppFactory, job_env: EnvFactory) -> None:
    env = job_env()
    app, view = start(make_app, env)
    assert view.job.status is JobStatus.COMPLETED, view.job.error
    report = app.get_report(view.job.job_id)
    assert report.final_report is not None and report.final_report.passed
    assert report.manifest is not None
    rows = {t.name: t.rows for t in report.manifest.tables}
    assert rows["customers"] == 1_000
    assert report.summary is not None
    assert "including 50 high-balance customers" in report.summary.text
    out = Path(env["SYNTHGEN_PATHS__OUTPUT_DIR"]) / "jobs" / view.job.job_id
    for name in ("manifest.json", "spec.json", "validation_report.json"):
        assert (out / name).is_file()
    preview = app.get_preview(view.job.job_id, "accounts", 5)
    assert preview.num_rows == 5


def _parts(env: dict[str, str], job_id: str) -> dict[str, str]:
    data = Path(env["SYNTHGEN_PATHS__OUTPUT_DIR"]) / "jobs" / job_id / "data"
    return {
        p.relative_to(data).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in data.rglob("*.parquet")
    }


def test_same_seed_gives_identical_parquet(make_app: AppFactory, job_env: EnvFactory) -> None:
    env = job_env()
    _, first = start(make_app, env, seed=5)
    _, second = start(make_app, env, seed=5)
    _, other = start(make_app, env, seed=6)
    assert _parts(env, first.job.job_id) == _parts(env, second.job.job_id)
    assert _parts(env, first.job.job_id) != _parts(env, other.job.job_id)


def test_job_pauses_for_approval(make_app: AppFactory, job_env: EnvFactory) -> None:
    app, view = start(make_app, job_env(), auto_approve=False)
    assert view.job.status is JobStatus.AWAITING_APPROVAL
    assert isinstance(view.pending, PendingApproval)
    assert view.pending.plan.scenarios[0].name == "high_balance"
    done = app.decide_plan(view.job.job_id, decide(ApprovalDecision.APPROVED))
    assert done.job.status is JobStatus.COMPLETED
    assert done.pending is None


def test_edited_share_is_what_gets_generated(make_app: AppFactory, job_env: EnvFactory) -> None:
    app, view = start(make_app, job_env(), auto_approve=False)
    edit = decide(
        ApprovalDecision.EDITED, share_edits=(ShareEdit(scenario="high_balance", share=0.1),)
    )
    done = app.decide_plan(view.job.job_id, edit)
    assert done.job.status is JobStatus.COMPLETED
    manifest = app.get_report(view.job.job_id).manifest
    assert manifest is not None
    counts = {c.scenario: c.rows for c in manifest.tables[0].scenario_counts}
    assert counts["high_balance"] == 100


def test_invalid_edit_is_shown_again(make_app: AppFactory, job_env: EnvFactory) -> None:
    app, view = start(make_app, job_env(), auto_approve=False)
    again = app.decide_plan(
        view.job.job_id, decide(ApprovalDecision.EDITED, removed_scenarios=("ghost",))
    )
    assert isinstance(again.pending, PendingApproval)
    assert again.pending.errors == ["unknown scenarios: ghost"]
    done = app.decide_plan(view.job.job_id, decide(ApprovalDecision.APPROVED))
    assert done.job.status is JobStatus.COMPLETED


def test_rejection_without_feedback_cancels(make_app: AppFactory, job_env: EnvFactory) -> None:
    app, view = start(make_app, job_env(), auto_approve=False)
    done = app.decide_plan(view.job.job_id, decide(ApprovalDecision.REJECTED))
    assert done.job.status is JobStatus.CANCELLED


def test_rejections_replan_until_the_cap(make_app: AppFactory, job_env: EnvFactory) -> None:
    app, view = start(make_app, job_env(), auto_approve=False)
    job_id = view.job.job_id
    for _ in range(3):
        view = app.decide_plan(job_id, decide(ApprovalDecision.REJECTED, feedback="add more"))
        assert isinstance(view.pending, PendingApproval)
    view = app.decide_plan(job_id, decide(ApprovalDecision.REJECTED, feedback="add more"))
    assert view.job.status is JobStatus.NEEDS_REVIEW
    assert view.job.error == "plan rejected after 3 revisions"


def _blocking_intent() -> dict[str, Any]:
    intent = skeleton("TestIntent")
    intent["ambiguities"] = [
        {"question": "Which balance counts as high?", "options": ["p95", "p99"], "blocking": True}
    ]
    return intent


def test_clarification_round_trip(
    make_app: AppFactory, job_env: EnvFactory, tmp_path: Path
) -> None:
    fixtures = fixture_set(
        tmp_path / "fx",
        TestIntent=[(None, _blocking_intent()), ("ANSWER:", skeleton("TestIntent"))],
    )
    app, view = start(make_app, job_env(fixtures))
    assert view.job.status is JobStatus.AWAITING_CLARIFICATION
    assert isinstance(view.pending, PendingClarification)
    assert view.pending.questions[0].question == "Which balance counts as high?"
    done = app.answer_clarification(view.job.job_id, "p99")
    assert done.job.status is JobStatus.COMPLETED


def test_clarification_cap(make_app: AppFactory, job_env: EnvFactory, tmp_path: Path) -> None:
    fixtures = fixture_set(tmp_path / "fx", TestIntent=[(None, _blocking_intent())])
    app, view = start(make_app, job_env(fixtures))
    view = app.answer_clarification(view.job.job_id, "p99")
    assert isinstance(view.pending, PendingClarification)
    view = app.answer_clarification(view.job.job_id, "really p99")
    assert view.job.status is JobStatus.NEEDS_REVIEW
