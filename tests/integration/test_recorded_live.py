from __future__ import annotations

from collections.abc import Callable

from synthgen.app import Application
from synthgen.domain.feasibility import FeasibilityStatus
from synthgen.domain.job import ArtifactKind, JobStatus, UserRequest
from synthgen.domain.plan import ScenarioPlan

from .helpers import FIXTURES

EnvFactory = Callable[..., dict[str, str]]
AppFactory = Callable[[dict[str, str]], Application]
REQUEST = UserRequest(
    source="finance",
    tables=("customers", "accounts", "cards", "transactions"),
    text=(
        "Card fraud test data for our fraud model: include card-testing bursts of small purchases,"
        " large purchases at night, and legitimate large daytime purchases that should not be"
        " flagged."
    ),
    auto_approve=True,
)


def test_recorded_live_answers_pass_every_gate(make_app: AppFactory, job_env: EnvFactory) -> None:
    env = job_env(FIXTURES / "live_finance", SYNTHGEN_AGENT__CAPS__PLANNER_TOOL_CALLS="1")
    app = make_app(env)
    view = app.start_job(REQUEST)
    assert view.job.status is JobStatus.COMPLETED, view.job.error
    job_id = view.job.job_id
    plan = app._artifacts.get(job_id, ArtifactKind.PLAN, ScenarioPlan)
    assert {s.name for s in plan.scenarios} >= {
        "card_testing_bursts",
        "large_nighttime_purchases",
        "legitimate_large_daytime",
    }
    report = app.get_report(job_id)
    assert report.final_report is not None and report.final_report.passed
    assert report.manifest is not None
    generated = {c.scenario for t in report.manifest.tables for c in t.scenario_counts}
    assert {"large_nighttime_purchases", "legitimate_large_daytime"} <= generated
    calls = app._llm_log.for_job(job_id)
    assert all(c.status == "ok" for c in calls)


def test_recorded_plan_feasibility(make_app: AppFactory, job_env: EnvFactory) -> None:
    env = job_env(FIXTURES / "live_finance", SYNTHGEN_AGENT__CAPS__PLANNER_TOOL_CALLS="1")
    view = make_app(env).start_job(REQUEST.model_copy(update={"auto_approve": False}))
    assert view.pending is not None and view.pending.kind == "approval"
    status = {f.scenario: f.status for f in view.pending.feasibility}
    assert status["large_nighttime_purchases"] is FeasibilityStatus.EXPRESSIBLE
    assert status["boundary_transactions_amount"] is FeasibilityStatus.NOT_EXPRESSIBLE
