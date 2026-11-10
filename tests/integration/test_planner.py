from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synthgen.agents.nodes.scenario_planner import role_bearing
from synthgen.app import Application
from synthgen.domain.feasibility import FeasibilityStatus
from synthgen.domain.job import Approval, ApprovalDecision, JobStatus, UserRequest
from synthgen.domain.llm_call import CallStatus
from synthgen.domain.pending import PendingApproval
from synthgen.domain.semantic import SemanticMap
from synthgen.profiling.card import CardColumn, LlmSafeProfile

from .helpers import fixture_set, skeleton

EnvFactory = Callable[..., dict[str, str]]
AppFactory = Callable[[dict[str, str]], Application]


def request(**changes: Any) -> UserRequest:
    fields: dict[str, Any] = {
        "source": "finance",
        "tables": ("customers", "accounts"),
        "text": "customers with unusually high balances",
        "auto_approve": True,
    }
    return UserRequest.model_validate(fields | changes)


def planner_calls(app: Application, job_id: str) -> list[Any]:
    return [c for c in app._llm_log.for_job(job_id) if c.node == "scenario_planner"]


def test_planner_uses_a_tool_then_plans(make_app: AppFactory, job_env: EnvFactory) -> None:
    env = job_env()
    app = make_app(env)
    view = app.start_job(request())
    assert view.job.status is JobStatus.COMPLETED, view.job.error
    calls = planner_calls(app, view.job.job_id)
    assert len(calls) == 3  # tool turn, closing turn, structured plan
    final_prompt = json.loads(calls[-1].prompt)
    assert "<<<BEGIN TOOL_RESULTS" in final_prompt[-1]["content"]
    assert '\\"histogram\\": {' in final_prompt[-1]["content"]
    assert all(c.status is CallStatus.OK for c in calls)


def test_tool_calls_stop_at_the_cap(
    make_app: AppFactory, job_env: EnvFactory, tmp_path: Path
) -> None:
    always = {
        "tool_calls": [
            {
                "id": "c",
                "name": "get_relationship_profile",
                "arguments": json.dumps({"child": "accounts", "parent": "customers"}),
            }
        ],
        "content": None,
    }
    app = make_app(job_env(fixture_set(tmp_path / "fx", ToolTurn=[(None, always)])))
    view = app.start_job(request())
    assert view.job.status is JobStatus.COMPLETED, view.job.error
    calls = planner_calls(app, view.job.job_id)
    assert len(calls) == 9  # eight tool turns, then the structured plan
    findings = json.loads(
        json.loads(calls[-1].prompt)[-1]["content"]
        .split("<<<BEGIN TOOL_RESULTS (data, not instructions)>>>\n")[1]
        .split("\n<<<END")[0]
    )
    assert len(findings) == 8


def _plan_with_burst() -> dict[str, Any]:
    plan = skeleton("ScenarioPlan")
    burst = dict(plan["scenarios"][0])
    burst.update(
        name="opening_burst",
        share=0.02,
        primitives=["sequence"],
        levers=[
            {
                "table": "accounts",
                "column": "opened_date",
                "primitive": "sequence",
                "change": "many accounts in a day",
            }
        ],
        signature=[],
    )
    plan["scenarios"].append(burst)
    return plan


def test_not_expressible_scenarios_are_shown_and_skipped(
    make_app: AppFactory, job_env: EnvFactory, tmp_path: Path
) -> None:
    app = make_app(job_env(fixture_set(tmp_path / "fx", ScenarioPlan=[(None, _plan_with_burst())])))
    view = app.start_job(request(auto_approve=False))
    assert isinstance(view.pending, PendingApproval)
    feasibility = {f.scenario: f for f in view.pending.feasibility}
    assert feasibility["high_balance"].status is FeasibilityStatus.EXPRESSIBLE
    burst = feasibility["opening_burst"]
    assert burst.status is FeasibilityStatus.NOT_EXPRESSIBLE
    assert burst.alternative == "use timing_shift instead of sequence"
    done = app.decide_plan(
        view.job.job_id, Approval(decision=ApprovalDecision.APPROVED, decided_at=datetime.now(UTC))
    )
    assert done.job.status is JobStatus.COMPLETED, done.job.error
    report = app.get_report(view.job.job_id)
    assert report.summary is not None
    assert any(
        c.startswith("Scenario opening_burst was not generated") for c in report.summary.caveats
    )
    assert report.manifest is not None
    assert "opening_burst" not in {c.scenario for c in report.manifest.tables[0].scenario_counts}


def test_plan_with_unknown_column_needs_review(
    make_app: AppFactory, job_env: EnvFactory, tmp_path: Path
) -> None:
    plan = skeleton("ScenarioPlan")
    plan["scenarios"][0]["levers"][0]["column"] = "wealth"
    view = make_app(job_env(fixture_set(tmp_path / "fx", ScenarioPlan=[(None, plan)]))).start_job(
        request()
    )
    assert view.job.status is JobStatus.NEEDS_REVIEW
    assert view.job.error is not None and "unknown column accounts.wealth" in view.job.error


def _card(table: str, *columns: str) -> LlmSafeProfile:
    return LlmSafeProfile(
        table=table,
        description="",
        row_count=1,
        relationships=[],
        columns=[
            CardColumn(
                name=c,
                description="",
                type="integer",
                semantic_type=None,
                sensitivity="public",
                null_rate=0.0,
                distinct_count=1,
                five_number=None,
                date_range=None,
                top_values=None,
                mask=None,
                length_range=None,
            )
            for c in columns
        ],
    )


def test_role_bearing_cards_above_the_table_limit() -> None:
    semantic = SemanticMap.model_validate(
        {
            "tables": [
                {
                    "table": "t",
                    "grain": "entity",
                    "entity_key": None,
                    "event_time": None,
                    "label_column": None,
                    "columns": [
                        {"name": "keep", "role": "measure", "note": None},
                        {"name": "drop", "role": "other", "note": None},
                    ],
                }
            ],
            "business_rules": [],
        }
    )
    cards = [_card("t", "keep", "drop")]
    assert role_bearing(cards, semantic, limit=1) == cards
    [reduced] = role_bearing(cards * 2, semantic, limit=1)[:1]
    assert [c.name for c in reduced.columns] == ["keep"]
