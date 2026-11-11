from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from synthgen.agents.meaning import (
    check_intent,
    check_overlays,
    check_patch,
    check_plan,
    check_semantic_map,
    check_summary,
)
from synthgen.domain.intent import TestIntent
from synthgen.domain.patch import SpecPatch
from synthgen.domain.plan import ScenarioPlan
from synthgen.domain.semantic import SemanticMap
from synthgen.domain.summary import Summary
from synthgen.spec.models import OverlaySet

from ..engine.test_runner import BIG, make_spec

COLUMNS = {"customers": ["id", "code", "tier"], "orders": ["id", "customer_id", "amount", "flag"]}


def intent(volume: int | None) -> TestIntent:
    return TestIntent.model_validate(
        {
            "domain": "d",
            "objective": "o",
            "test_type": "performance_testing",
            "entities_of_interest": [],
            "must_include": [],
            "must_exclude": [],
            "volume_hint": volume,
            "include_invalid": False,
            "ambiguities": [],
            "confidence": 1.0,
        }
    )


def test_intent_volume_limit() -> None:
    assert check_intent(intent(None), 100) == []
    assert check_intent(intent(100), 100) == []
    assert check_intent(intent(101), 100) == ["volume_hint 101 exceeds the limit of 100 rows"]


def semantic(
    tables: Sequence[tuple[str, list[str], str | None]], rule_columns: list[str] | None = None
) -> SemanticMap:
    return SemanticMap.model_validate(
        {
            "tables": [
                {
                    "table": name,
                    "grain": "entity",
                    "entity_key": None,
                    "event_time": key,
                    "label_column": None,
                    "columns": [{"name": c, "role": "other", "note": None} for c in cols],
                }
                for name, cols, key in tables
            ],
            "business_rules": [{"rule": "r", "columns": rule_columns or [], "confidence": 0.5}],
        }
    )


def test_semantic_map_must_cover_every_column_once() -> None:
    complete = semantic(
        [("customers", COLUMNS["customers"], None), ("orders", COLUMNS["orders"], None)]
    )
    assert check_semantic_map(complete, COLUMNS) == []
    problems = check_semantic_map(
        semantic([("customers", ["id", "code", "ghost"], "nope")], ["orders.amount", "bogus"]),
        COLUMNS,
    )
    assert "table orders is not mapped" in problems
    assert "column customers.tier is not mapped" in problems
    assert "unknown column customers.ghost" in problems
    assert "unknown key column customers.nope" in problems
    assert "business rule names unknown column 'bogus'" in problems
    twice = semantic(
        [("customers", COLUMNS["customers"], None)] * 2 + [("orders", COLUMNS["orders"], None)]
    )
    assert check_semantic_map(twice, COLUMNS) == ["table customers is mapped more than once"]


def plan(**changes: Any) -> ScenarioPlan:
    scenario: dict[str, Any] = {
        "name": "s",
        "category": "rare",
        "description": "d",
        "rationale": "r",
        "share": 0.1,
        "unit": "entity",
        "tables": ["orders"],
        "primitives": ["value_shift"],
        "levers": [
            {"table": "orders", "column": "amount", "primitive": "value_shift", "change": "x"}
        ],
        "signature": [
            {
                "table": "orders",
                "column": "amount",
                "metric": "value",
                "operator": "gt",
                "value": 1.0,
                "window_hours": None,
            }
        ],
        "template_ref": None,
    }
    scenario.update(changes)
    return ScenarioPlan.model_validate({"scenarios": [scenario], "assumptions": []})


def test_plan_references() -> None:
    assert check_plan(plan(), COLUMNS, frozenset(), set()) == []
    assert check_plan(plan(template_ref="t.x"), COLUMNS, frozenset({"t.x"}), set()) == []
    problems = check_plan(
        plan(
            tables=["ghosts"],
            template_ref="missing",
            levers=[{"table": "orders", "column": "z", "primitive": "value_shift", "change": "x"}],
        ),
        COLUMNS,
        frozenset(),
        set(),
    )
    assert problems == [
        "s: unknown table 'ghosts'",
        "s: unknown column orders.z",
        "s: unknown template 'missing'",
    ]
    assert check_plan(ScenarioPlan(scenarios=[], assumptions=[]), COLUMNS, frozenset(), set()) == [
        "the plan has no scenarios"
    ]


def test_overlays_match_the_approved_plan() -> None:
    overlays = OverlaySet.model_validate({"scenarios": [BIG]})
    assert check_overlays(overlays, ["big"], COLUMNS) == []
    assert check_overlays(overlays, ["other"], COLUMNS) == [
        "missing overlay for scenario 'other'",
        "overlay 'big' is not an approved scenario",
    ]
    bad = dict(
        BIG,
        overrides=[
            {"table": "orders", "column": "z", "generator": {"type": "constant", "value": 1}}
        ],
    )
    assert check_overlays(OverlaySet.model_validate({"scenarios": [bad]}), ["big"], COLUMNS) == [
        "big: unknown column orders.z"
    ]


def patch(path: str, value: str, replan: bool = False) -> SpecPatch:
    ops = [] if replan else [{"path": path, "value_json": value, "reason": "r"}]
    return SpecPatch.model_validate(
        {"patches": ops, "plan_change_required": replan, "explanation": "e"}
    )


def test_patch_allow_list() -> None:
    spec = make_spec(scenarios=[BIG])
    assert check_patch(patch("/scenarios/0/overrides/0/generator/decimals", "1"), spec) == []
    assert check_patch(patch("/tables/1/columns/2/generator/quantiles", "[1, 2]"), spec) == []
    assert check_patch(patch("", "", replan=True), spec) == []
    [blocked] = check_patch(patch("/scenarios/0/share", "0.5"), spec)
    assert "/scenarios/0/share is not patchable" in blocked
    [typed] = check_patch(patch("/scenarios/0/overrides/0/generator/type", '"constant"'), spec)
    assert "not patchable" in typed
    [missing] = check_patch(patch("/scenarios/0/overrides/7/generator/decimals", "1"), spec)
    assert "not an index" in missing
    empty = SpecPatch(patches=[], plan_change_required=False, explanation="e")
    assert check_patch(empty, spec)[0].startswith("no patches given")


def test_summary_numbers_only_through_placeholders() -> None:
    names = {"total_rows"}
    assert check_summary(Summary(text="We made {total_rows} rows.", caveats=[]), names) == []
    assert check_summary(Summary(text="We made 5 rows.", caveats=[]), names) == [
        "write numbers only as {placeholders}: 'We made 5 rows.'"
    ]
    assert check_summary(Summary(text="{made_up}", caveats=["fine"]), names) == [
        "unknown metric {made_up}"
    ]


def test_plan_must_not_plan_typical_data_or_only_set_labels() -> None:
    flag = {"table": "orders", "column": "flag", "primitive": "value_shift", "change": "x"}
    amount = {"table": "orders", "column": "amount", "primitive": "value_shift", "change": "x"}
    labels = {("orders", "flag")}
    assert check_plan(plan(levers=[flag]), COLUMNS, frozenset(), labels) == [
        "s: only label columns change; change the data columns that make the label true"
    ]
    assert check_plan(plan(levers=[amount, flag]), COLUMNS, frozenset(), labels) == []
    assert check_plan(plan(category="typical"), COLUMNS, frozenset(), set()) == [
        "s: typical data is the remainder after all shares; do not plan it"
    ]
