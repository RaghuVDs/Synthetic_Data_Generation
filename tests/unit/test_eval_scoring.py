from __future__ import annotations

from typing import Any

import pytest

from synthgen.domain.intent import TestIntent
from synthgen.domain.job import JobStatus, LoopCounters
from synthgen.domain.plan import ScenarioPlan
from synthgen.domain.semantic import ColumnRole, SemanticMap
from synthgen.evaluation.golden import GoldenRequest, RoleLabels
from synthgen.evaluation.scoring import JobFacts, coverage, role_accuracy, score, summarise

GOLDEN = GoldenRequest.model_validate(
    {
        "name": "fraud",
        "source": "finance",
        "tables": ["cards", "transactions"],
        "target_table": "transactions",
        "request": "fraud bursts and look-alikes",
        "answer": None,
        "expected": {
            "test_type": "model_validation",
            "include_invalid": False,
            "must_have": [
                {
                    "what": "bursts",
                    "categories": ["target_scenario"],
                    "tables": ["transactions"],
                    "columns": ["txn_timestamp"],
                },
                {"what": "look-alikes", "categories": ["near_miss"], "tables": [], "columns": []},
            ],
        },
    }
)
ROLES: RoleLabels = {
    "transactions": {"amount": [ColumnRole.MEASURE], "txn_timestamp": [ColumnRole.EVENT_TIME]},
    "merchants": {"name": [ColumnRole.OTHER]},
}


def _scenario(name: str, category: str, lever: tuple[str, str]) -> dict[str, Any]:
    return {
        "name": name,
        "category": category,
        "description": "d",
        "rationale": "r",
        "share": 0.01,
        "unit": "entity",
        "tables": ["cards", "transactions"],
        "primitives": ["value_shift"],
        "levers": [
            {"table": lever[0], "column": lever[1], "primitive": "value_shift", "change": "c"}
        ],
        "signature": [],
        "template_ref": None,
    }


def _plan(*scenarios: dict[str, Any]) -> ScenarioPlan:
    return ScenarioPlan.model_validate({"scenarios": list(scenarios), "assumptions": []})


def _map(roles: dict[str, str]) -> SemanticMap:
    columns = [{"name": n, "role": r, "note": None} for n, r in roles.items()]
    table = {
        "table": "transactions",
        "grain": "event",
        "entity_key": None,
        "event_time": None,
        "label_column": None,
        "columns": columns,
    }
    return SemanticMap.model_validate({"tables": [table], "business_rules": []})


def _intent(test_type: str, include_invalid: bool = False) -> TestIntent:
    return TestIntent.model_validate(
        {
            "domain": "banking",
            "objective": "o",
            "test_type": test_type,
            "entities_of_interest": [],
            "must_include": [],
            "must_exclude": [],
            "volume_hint": None,
            "include_invalid": include_invalid,
            "ambiguities": [],
            "confidence": 0.9,
        }
    )


def test_coverage_needs_category_tables_and_a_touched_column() -> None:
    wrong_column = _plan(_scenario("a", "target_scenario", ("transactions", "amount")))
    assert coverage(wrong_column, GOLDEN.expected.must_have) == 0.0
    burst = _scenario("b", "target_scenario", ("transactions", "txn_timestamp"))
    assert coverage(_plan(burst), GOLDEN.expected.must_have) == 0.5
    near = _scenario("c", "near_miss", ("cards", "status"))
    assert coverage(_plan(burst, near), GOLDEN.expected.must_have) == 1.0


def test_role_accuracy_counts_labelled_columns_of_the_jobs_tables() -> None:
    mapped = _map({"amount": "measure", "txn_timestamp": "measure"})
    assert role_accuracy(mapped, ROLES, ["transactions"]) == 0.5
    assert role_accuracy(mapped, ROLES, ["cards"]) is None  # no labels for these tables
    assert role_accuracy(None, ROLES, ["transactions"]) is None


def _facts(**changes: Any) -> JobFacts:
    fields: dict[str, Any] = {
        "job_id": "j",
        "status": JobStatus.COMPLETED,
        "error": None,
        "intent": _intent("model_validation"),
        "semantic_map": _map({"amount": "measure", "txn_timestamp": "event_time"}),
        "plan": _plan(_scenario("b", "target_scenario", ("transactions", "txn_timestamp"))),
        "counters": LoopCounters(spec_repair=1, validation_repair=2),
        "has_spec": True,
        "first_dry_run_passed": False,
        "final_passed": True,
        "tokens": 1_000,
        "llm_latency_ms": 2_000,
        "seconds": 3.0,
    }
    return JobFacts.model_validate(fields | changes)


def test_score_reads_every_metric() -> None:
    found = score(GOLDEN, 100, _facts(), ROLES)
    assert found.intent_correct is True
    assert found.role_accuracy == 1.0
    assert found.plan_coverage == 0.5
    assert found.first_pass_valid is False
    assert found.repair_loops == 3
    assert (found.dry_run_passed, found.final_passed) == (False, True)


@pytest.mark.parametrize("intent", [_intent("rule_testing"), _intent("model_validation", True)])
def test_wrong_test_type_or_invalid_flag_is_a_miss(intent: TestIntent) -> None:
    assert score(GOLDEN, 100, _facts(intent=intent), ROLES).intent_correct is False


def test_a_job_that_stopped_early_leaves_later_metrics_empty() -> None:
    early = _facts(
        status=JobStatus.NEEDS_REVIEW,
        semantic_map=None,
        plan=None,
        has_spec=False,
        first_dry_run_passed=None,
        final_passed=None,
        counters=LoopCounters(),
    )
    found = score(GOLDEN, 100, early, ROLES)
    assert (found.plan_coverage, found.first_pass_valid, found.dry_run_passed) == (None,) * 3


def test_summary_averages_what_each_job_reached() -> None:
    done = score(GOLDEN, 100, _facts(), ROLES)
    early = score(GOLDEN, 100, _facts(status=JobStatus.FAILED, plan=None, tokens=3_000), ROLES)
    summary = summarise([done, early])
    assert (summary.jobs, summary.completed) == (2, 1)
    assert summary.plan_coverage == 0.5
    assert summary.tokens_per_job == 2_000
    assert summarise([]).intent_accuracy is None
