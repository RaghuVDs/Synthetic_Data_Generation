from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from synthgen.domain.intent import TestIntent, TestType
from synthgen.domain.patch import SpecPatch
from synthgen.domain.plan import ScenarioPlan
from synthgen.domain.report import CheckFamily, CheckResult, CheckStatus, ValidationReport
from synthgen.domain.semantic import SemanticMap
from synthgen.domain.summary import Summary

LLM_SCHEMAS: list[type[BaseModel]] = [TestIntent, SemanticMap, ScenarioPlan, SpecPatch, Summary]


def _objects(node: Any) -> list[dict[str, Any]]:
    """Every JSON-schema node that describes an object."""
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            found.append(node)
        for value in node.values():
            found.extend(_objects(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_objects(item))
    return found


@pytest.mark.parametrize("model", LLM_SCHEMAS, ids=lambda m: m.__name__)
def test_schema_is_strict_mode_compliant(model: type[BaseModel]) -> None:
    schema = model.model_json_schema()
    objects = _objects(schema)
    assert objects
    for obj in objects:
        assert obj.get("additionalProperties") is False, obj.get("title")
        assert set(obj.get("required", [])) == set(obj.get("properties", {})), obj.get("title")
    assert '"default"' not in json.dumps(schema)


def _scenario(name: str, share: float) -> dict[str, Any]:
    return {
        "name": name,
        "category": "target_scenario",
        "description": "large amounts at night",
        "rationale": "requested",
        "share": share,
        "unit": "entity",
        "tables": ["transactions"],
        "primitives": ["value_shift", "timing_shift"],
        "levers": [
            {
                "table": "transactions",
                "column": "amount",
                "primitive": "value_shift",
                "change": "above p97",
            }
        ],
        "signature": [
            {
                "table": "transactions",
                "column": "amount",
                "metric": "value",
                "operator": "gt",
                "value": 950.0,
                "window_hours": None,
            }
        ],
        "template_ref": None,
    }


def test_plan_round_trip_and_total_share() -> None:
    plan = ScenarioPlan.model_validate(
        {"scenarios": [_scenario("a", 0.4), _scenario("b", 0.6)], "assumptions": []}
    )
    assert plan.total_share == pytest.approx(1.0)
    assert ScenarioPlan.model_validate_json(plan.model_dump_json()) == plan


@pytest.mark.parametrize(
    "scenarios",
    [
        [_scenario("a", 0.7), _scenario("b", 0.4)],
        [_scenario("a", 0.1), _scenario("a", 0.1)],
        [_scenario("Bad Name", 0.1)],
    ],
)
def test_plan_rejects_bad_scenarios(scenarios: list[dict[str, Any]]) -> None:
    with pytest.raises(ValidationError):
        ScenarioPlan.model_validate({"scenarios": scenarios, "assumptions": []})


def test_intent_blocking_ambiguities() -> None:
    intent = TestIntent.model_validate(
        {
            "domain": "retail",
            "objective": "test late deliveries",
            "test_type": "rule_testing",
            "entities_of_interest": ["shipments"],
            "must_include": [],
            "must_exclude": [],
            "volume_hint": None,
            "include_invalid": False,
            "ambiguities": [
                {"question": "Which carrier?", "options": ["any"], "blocking": False},
                {"question": "How late?", "options": ["1 day", "1 week"], "blocking": True},
            ],
            "confidence": 0.8,
        }
    )
    assert intent.test_type is TestType.RULE_TESTING
    assert [a.question for a in intent.blocking_ambiguities] == ["How late?"]


def test_intent_requires_every_field() -> None:
    with pytest.raises(ValidationError):
        TestIntent.model_validate({"domain": "x", "objective": "y", "test_type": "ui_testing"})


def test_semantic_map_lookup() -> None:
    semantic = SemanticMap.model_validate(
        {
            "tables": [
                {
                    "table": "orders",
                    "grain": "event",
                    "entity_key": "customer_id",
                    "event_time": "order_date",
                    "label_column": None,
                    "columns": [{"name": "order_id", "role": "identifier", "note": None}],
                }
            ],
            "business_rules": [],
        }
    )
    assert semantic.for_table("orders") is not None
    assert semantic.for_table("missing") is None


def test_patch_value_is_decoded_json() -> None:
    patch = SpecPatch.model_validate(
        {
            "patches": [
                {"path": "/tables/0/columns/2/clip/max", "value_json": "500.5", "reason": "r"}
            ],
            "plan_change_required": False,
            "explanation": "tighten clip",
        }
    )
    assert patch.patches[0].value == 500.5


@pytest.mark.parametrize(("path", "value_json"), [("no-slash", "1"), ("/a", "not json"), ("", "1")])
def test_patch_rejects_bad_ops(path: str, value_json: str) -> None:
    with pytest.raises(ValidationError):
        SpecPatch.model_validate(
            {
                "patches": [{"path": path, "value_json": value_json, "reason": "r"}],
                "plan_change_required": False,
                "explanation": "x",
            }
        )


def test_summary_word_limit() -> None:
    assert Summary(text="word " * 150, caveats=[]).caveats == []
    with pytest.raises(ValidationError):
        Summary(text="word " * 151, caveats=[])
    with pytest.raises(ValidationError):
        Summary(text="   ", caveats=[])


def test_report_failures_by_family() -> None:
    def row(family: CheckFamily, status: CheckStatus) -> CheckResult:
        return CheckResult(family=family, check="c", table="t", metric="m", status=status)

    report = ValidationReport(
        results=(
            row(CheckFamily.FIDELITY, CheckStatus.FAIL),
            row(CheckFamily.INTEGRITY, CheckStatus.PASS),
            row(CheckFamily.PRIVACY, CheckStatus.SKIPPED),
        )
    )
    assert not report.passed
    assert len(report.failures()) == 1
    assert report.failures(CheckFamily.INTEGRITY) == []
    assert ValidationReport(results=()).passed
