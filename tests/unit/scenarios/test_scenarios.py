from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from synthgen.config import ProfilingSettings
from synthgen.connectors.sqlite import SqliteConnector
from synthgen.domain.errors import MetadataError
from synthgen.domain.feasibility import FeasibilityStatus
from synthgen.domain.plan import Primitive, ScenarioPlan
from synthgen.domain.semantic import SemanticMap
from synthgen.metadata.provider import SourceDoc, YamlMetadataProvider
from synthgen.profiling.models import TableProfile
from synthgen.profiling.profiler import profile_table
from synthgen.scenarios import feasibility
from synthgen.scenarios.feasibility import check_feasibility
from synthgen.scenarios.primitives import RULES, Construct, available_constructs
from synthgen.scenarios.profile_derived import profile_candidates
from synthgen.scenarios.templates.loader import BUILT_IN, TemplateLibrary

PROFILE_NOW = datetime(2026, 1, 1, tzinfo=UTC)

COLUMNS = {"orders": ["id", "customer_id", "amount", "placed", "paid"], "customers": ["id", "tier"]}
SEMANTIC = SemanticMap.model_validate(
    {
        "tables": [
            {
                "table": "orders",
                "grain": "event",
                "entity_key": "customer_id",
                "event_time": "placed",
                "label_column": "paid",
                "columns": [
                    {"name": "id", "role": "identifier", "note": None},
                    {"name": "customer_id", "role": "foreign_key", "note": None},
                    {"name": "amount", "role": "measure", "note": None},
                    {"name": "placed", "role": "event_time", "note": None},
                    {"name": "paid", "role": "label", "note": None},
                ],
            },
            {
                "table": "customers",
                "grain": "entity",
                "entity_key": "id",
                "event_time": None,
                "label_column": None,
                "columns": [
                    {"name": "id", "role": "identifier", "note": None},
                    {"name": "tier", "role": "category", "note": None},
                ],
            },
        ],
        "business_rules": [],
    }
)


def scenario(
    name: str, levers: list[tuple[str, str, str]], primitives: list[str] | None = None
) -> dict[str, Any]:
    return {
        "name": name,
        "category": "target_scenario",
        "description": "d",
        "rationale": "r",
        "share": 0.01,
        "unit": "entity",
        "tables": ["orders"],
        "primitives": primitives or sorted({p for _, _, p in levers}) or ["value_shift"],
        "levers": [{"table": t, "column": c, "primitive": p, "change": "x"} for t, c, p in levers],
        "signature": [],
        "template_ref": None,
    }


def test_every_primitive_has_a_rule() -> None:
    assert set(RULES) == set(Primitive)
    assert len(RULES) == 14


def test_available_constructs_follow_the_spec_models() -> None:
    assert available_constructs() == frozenset(Construct)


def test_feasibility_statuses(monkeypatch: pytest.MonkeyPatch) -> None:
    without_sequences = available_constructs() - {Construct.SEQUENCE}
    monkeypatch.setattr(feasibility, "available_constructs", lambda: without_sequences)
    plan = ScenarioPlan.model_validate(
        {
            "scenarios": [
                scenario("big", [("orders", "amount", "value_shift")]),
                scenario(
                    "mixed", [("orders", "amount", "value_shift"), ("orders", "placed", "sequence")]
                ),
                scenario("burst", [("orders", "placed", "sequence")]),
                scenario("ghost", [("orders", "nope", "value_shift")]),
                scenario("wrong_role", [("orders", "amount", "timing_shift")]),
                scenario(
                    "labelled",
                    [("orders", "amount", "value_shift"), ("orders", "paid", "value_shift")],
                ),
            ],
            "assumptions": [],
        }
    )
    results = {r.scenario: r for r in check_feasibility(plan, SEMANTIC, COLUMNS)}
    assert results["big"].status is FeasibilityStatus.EXPRESSIBLE
    assert results["big"].covered == ["value_shift on orders.amount"]
    mixed = results["mixed"]
    assert mixed.status is FeasibilityStatus.PARTIAL
    assert mixed.not_covered == ["sequence on orders.placed"]
    assert mixed.reason is not None and "SequenceSpec" in mixed.reason
    burst = results["burst"]
    assert burst.status is FeasibilityStatus.NOT_EXPRESSIBLE
    assert burst.alternative == "use timing_shift instead of sequence"
    assert results["ghost"].reason == "orders.nope does not exist"
    assert "does not apply" in (results["wrong_role"].reason or "")
    assert results["wrong_role"].alternative == "use value_shift instead of timing_shift"
    labelled = results["labelled"]
    assert labelled.status is FeasibilityStatus.EXPRESSIBLE
    assert labelled.covered == [
        "value_shift on orders.amount",
        "label orders.paid set by the overlay",
    ]


@pytest.fixture(scope="module")
def shop_profiles(
    shop_db: Path, profiling_settings: ProfilingSettings
) -> Iterator[dict[str, TableProfile]]:
    connector = SqliteConnector(shop_db)
    provider = YamlMetadataProvider(SourceDoc(source="shop"), connector)
    yield {
        t: profile_table(
            "shop",
            connector,
            provider.table(t),
            provider.relationships(),
            parents={m.name: m for m in provider.tables()},
            settings=profiling_settings,
            seed=0,
            now=PROFILE_NOW,
        )
        for t in ("customers", "orders")
    }
    connector.close()


def test_profile_candidates(shop_profiles: dict[str, TableProfile]) -> None:
    found = {c.name: c for c in profile_candidates(shop_profiles, SEMANTIC, include_invalid=False)}
    assert "boundary_orders_amount" in found
    assert "temporal_orders_placed" in found
    assert not any(c.category == "invalid" for c in found.values())
    with_invalid = profile_candidates(shop_profiles, SEMANTIC, include_invalid=True)
    assert any(c.category == "invalid" for c in with_invalid)
    assert all("person" not in c.description for c in with_invalid)


def test_built_in_templates() -> None:
    library = TemplateLibrary(BUILT_IN)
    assert library.ids == {
        "card_fraud.card_testing",
        "card_fraud.night_high_value",
        "card_fraud.compromised_merchant",
    }
    assert library.search("fraud", ["night"])[0].id == "card_fraud.night_high_value"
    assert library.search("late delivery", []) == []
    template = library.get("card_fraud.card_testing")
    assert template is not None and template.default_share == 0.002
    assert library.get("missing") is None


def test_no_templates_is_fine(tmp_path: Path) -> None:
    assert TemplateLibrary(tmp_path).ids == frozenset()
    assert TemplateLibrary(tmp_path / "absent").search("any", []) == []


def test_invalid_template_fails_loudly(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text("id: x\n", encoding="utf-8")
    with pytest.raises(MetadataError):
        TemplateLibrary(tmp_path)
