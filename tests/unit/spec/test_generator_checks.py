from __future__ import annotations

from typing import Any

import pytest

from synthgen.spec.generators import SemanticKind
from synthgen.spec.merge import merge_overlays
from synthgen.spec.models import GenerationSpec, ScenarioSpec
from synthgen.spec.semantic_kinds import semantic_kind
from synthgen.spec.validator import validate_spec

from ..engine.test_runner import GENERATION, make_spec


def _override(table: str, column: str, generator: dict[str, Any]) -> GenerationSpec:
    overlay = {
        "name": "s",
        "category": "target_scenario",
        "share": 0.1,
        "unit": "entity",
        "entity": "customers",
        "labels": [],
        "correlations": [],
        "cardinality": [],
        "sequences": [],
        "defects": [],
        "overrides": [{"table": table, "column": column, "generator": generator}],
    }
    return merge_overlays(make_spec(), [ScenarioSpec.model_validate(overlay)])


def _messages(spec: GenerationSpec) -> list[str]:
    return [e.message for e in validate_spec(spec, GENERATION)]


@pytest.mark.parametrize(
    ("table", "column", "generator"),
    [
        ("orders", "amount", {"type": "derived", "expression": "flag * 2 + amount"}),
        ("orders", "amount", {"type": "derived", "expression": "customers.id + 0.5"}),
        ("orders", "amount", {"type": "boundary", "points": [0.0, 99999.0]}),
        ("customers", "tier", {"type": "text_pool", "values": ["a", "b"]}),
        (
            "orders",
            "amount",
            {
                "type": "conditional",
                "column": "flag",
                "cases": [
                    {"values": [1], "generator": {"type": "boundary", "points": [0.0]}},
                ],
                "otherwise": {"type": "derived", "expression": "amount"},
            },
        ),
    ],
)
def test_valid_new_generators_pass(table: str, column: str, generator: dict[str, Any]) -> None:
    assert _messages(_override(table, column, generator)) == []


@pytest.mark.parametrize(
    ("table", "column", "generator", "fragment"),
    [
        ("orders", "amount", {"type": "derived", "expression": "ghost + 1"}, "unknown column"),
        ("orders", "amount", {"type": "derived", "expression": "flag > 0"}, "gives boolean"),
        (
            "orders",
            "amount",
            {"type": "derived", "expression": "accounts.id + 1"},
            "unknown column accounts.id",
        ),
        ("orders", "amount", {"type": "boundary", "points": ["high"]}, "boundary point"),
        ("customers", "tier", {"type": "boundary", "points": [1.0]}, "boundary point"),
        ("customers", "tier", {"type": "text_pool", "values": ["v"] * 501}, "more than 500"),
        ("orders", "amount", {"type": "semantic", "kind": "email"}, "semantic email"),
        ("customers", "tier", {"type": "semantic", "kind": "date_of_birth"}, ""),
        (
            "orders",
            "amount",
            {
                "type": "conditional",
                "column": "amount",
                "cases": [{"values": [1], "generator": {"type": "constant", "value": 1}}],
                "otherwise": {"type": "constant", "value": 0},
            },
            "not another column",
        ),
        (
            "orders",
            "amount",
            {
                "type": "conditional",
                "column": "flag",
                "cases": [{"values": [1], "generator": {"type": "uuid"}}],
                "otherwise": {"type": "constant", "value": 0},
            },
            "conditional case: uuid",
        ),
        (
            "orders",
            "amount",
            {
                "type": "distribution",
                "family": "norm",
                "params": [1.0, 1.0],
                "minimum": None,
                "maximum": None,
                "decimals": None,
            },
            "",
        ),
    ],
)
def test_new_generator_problems_are_reported(
    table: str, column: str, generator: dict[str, Any], fragment: str
) -> None:
    messages = _messages(_override(table, column, generator))
    if fragment:
        assert any(fragment in m for m in messages), messages
    else:
        assert messages == []


def test_columns_reading_each_other_are_a_cycle() -> None:
    data = make_spec().model_dump()
    expressions = {"amount": "flag + 1", "flag": "amount + 1"}
    for column in data["tables"][1]["columns"]:
        if column["name"] in expressions:
            column["generator"] = {"type": "derived", "expression": expressions[column["name"]]}
    assert any("cycle" in m for m in _messages(GenerationSpec.model_validate(data)))


def test_overrides_reading_each_other_are_a_cycle() -> None:
    spec = _override("orders", "amount", {"type": "derived", "expression": "flag + 1"})
    data = spec.model_dump()
    data["scenarios"][0]["overrides"].append(
        {
            "table": "orders",
            "column": "flag",
            "generator": {"type": "derived", "expression": "amount + 1"},
        }
    )
    assert any("cycle" in m for m in _messages(GenerationSpec.model_validate(data)))


@pytest.mark.parametrize(
    ("column", "semantic_type", "kind"),
    [
        ("email", "email", SemanticKind.EMAIL),
        ("contact", "phone", SemanticKind.PHONE),
        ("pan", "card_number", SemanticKind.CARD_NUMBER),
        ("first_name", None, SemanticKind.FIRST_NAME),
        ("LastName", None, SemanticKind.LAST_NAME),
        ("surname", None, SemanticKind.LAST_NAME),
        ("customer_name", None, SemanticKind.FULL_NAME),
        ("date_of_birth", "date", SemanticKind.DATE_OF_BIRTH),
        ("dob", None, SemanticKind.DATE_OF_BIRTH),
        ("home_city", None, SemanticKind.CITY),
        ("zip", None, SemanticKind.POSTCODE),
        ("street_address", None, SemanticKind.STREET_ADDRESS),
        ("employer", None, SemanticKind.COMPANY),
        ("birth_place", "email", SemanticKind.EMAIL),
        ("national_id", None, None),
    ],
)
def test_semantic_kind_from_format_or_name(
    column: str, semantic_type: str | None, kind: SemanticKind | None
) -> None:
    assert semantic_kind(column, semantic_type) is kind
