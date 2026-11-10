from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from synthgen.agents.nodes.deps import Source
from synthgen.agents.tools.cache import ToolResultCache
from synthgen.agents.tools.context import ToolContext
from synthgen.agents.tools.registry import TRUNCATED, ToolRegistry
from synthgen.config import ProfilingSettings
from synthgen.connectors.sqlite import SqliteConnector
from synthgen.metadata.provider import SourceDoc, YamlMetadataProvider
from synthgen.profiling.profiler import profile_table
from synthgen.scenarios.templates.loader import BUILT_IN, TemplateLibrary


@pytest.fixture(scope="module")
def ctx(shop_db: Path, profiling_settings: ProfilingSettings) -> Iterator[ToolContext]:
    connector = SqliteConnector(shop_db)
    provider = YamlMetadataProvider(SourceDoc(source="shop"), connector)
    profiles = {
        t: profile_table(
            "shop",
            connector,
            provider.table(t),
            provider.relationships(),
            settings=profiling_settings,
            seed=0,
        )
        for t in ("customers", "orders")
    }
    yield ToolContext(
        job_id="job",
        source=Source(connector=connector, metadata=provider),
        profiles=profiles,
        templates=TemplateLibrary(BUILT_IN),
        settings=profiling_settings,
    )
    connector.close()


def call(ctx: ToolContext, name: str, max_tokens: int = 2_000, **args: Any) -> Any:
    text = ToolRegistry(ctx, ToolResultCache(), max_tokens).call(name, json.dumps(args))
    return text if text.startswith("error") else json.loads(text)


def test_definitions_are_strict(ctx: ToolContext) -> None:
    definitions = ToolRegistry(ctx, ToolResultCache(), 2_000).definitions()
    assert len(definitions) == 7
    assert all(d.parameters["additionalProperties"] is False for d in definitions)


@pytest.mark.parametrize(
    ("name", "args", "fragment"),
    [
        ("no_such_tool", {}, "unknown tool"),
        ("get_column_profile", {"table": "orders"}, "invalid arguments"),
        (
            "get_column_profile",
            {"table": "ghosts", "column": "x"},
            "unknown table 'ghosts'; tables: customers, orders",
        ),
        ("get_column_profile", {"table": "orders", "column": "x"}, "unknown column orders.x"),
        (
            "get_conditional_stats",
            {"table": "customers", "target": "email", "by": "tier"},
            "is sensitive",
        ),
        (
            "get_relationship_profile",
            {"child": "customers", "parent": "orders"},
            "has no foreign key",
        ),
        ("get_scenario_template", {"template_id": "nope"}, "unknown template"),
    ],
)
def test_problems_come_back_as_error_strings(
    ctx: ToolContext, name: str, args: dict[str, Any], fragment: str
) -> None:
    result = call(ctx, name, **args)
    assert isinstance(result, str) and result.startswith("error") and fragment in result


def test_column_profiles(ctx: ToolContext) -> None:
    amount = call(ctx, "get_column_profile", table="orders", column="amount")
    assert amount["type"] == "float" and "p50" in amount["percentiles"]
    assert sum(amount["histogram"]["counts"]) == 450
    tier = call(ctx, "get_column_profile", table="customers", column="tier")
    assert {v["value"] for v in tier["top_values"]} == {"gold", "silver", "bronze"}
    email = call(ctx, "get_column_profile", table="customers", column="email")
    assert email["masks"] and "top_values" not in email
    assert "person" not in json.dumps(email)


def test_conditional_stats(ctx: ToolContext) -> None:
    result = call(ctx, "get_conditional_stats", table="orders", target="amount", by="paid")
    assert [g["paid"] for g in result["groups"]] == [0, 1]
    assert sum(g["rows"] for g in result["groups"]) == 450
    assert result["groups"][0]["avg"] > 0


def test_relationship_profile(ctx: ToolContext) -> None:
    result = call(ctx, "get_relationship_profile", child="orders", parent="customers")
    assert result["children_per_parent"]["mean"] == pytest.approx(1.5)
    assert result["parents_without_children"] == pytest.approx(0.25)
    assert result["orphan_rate"] == 0.0
    assert 0.0 < result["gini"] < 1.0


def test_behaviour_profile(ctx: ToolContext) -> None:
    result = call(
        ctx,
        "get_behaviour_profile",
        table="orders",
        entity_key="customer_id",
        time_column="placed",
        measure="amount",
    )
    assert result["entities"] == 225
    assert result["events"] == 450
    assert result["events_per_entity"]["max"] == 3.0
    assert result["burst_24h"]["max"] == 3.0
    assert result["inter_arrival_hours"]["p50"] == pytest.approx(1.0)


def test_class_separation(ctx: ToolContext) -> None:
    result = call(ctx, "get_class_separation", table="orders", label="paid")
    assert result["positive_class"] == 1
    assert result["positive_share"] == pytest.approx(150 / 450)
    assert {s["column"] for s in result["separation"]} == {"amount", "placed (hour)"}


def test_templates(ctx: ToolContext) -> None:
    found = call(ctx, "search_scenario_templates", domain="card fraud", keywords=["night"])
    assert found[0]["id"] == "card_fraud.night_high_value"
    template = call(ctx, "get_scenario_template", template_id="card_fraud.card_testing")
    assert template["primitives"] == ["sequence", "value_shift"]


def test_results_are_cached_per_job_and_capped(ctx: ToolContext) -> None:
    cache = ToolResultCache()
    registry = ToolRegistry(ctx, cache, 2_000)
    args = json.dumps({"table": "orders", "column": "amount"})
    first = registry.call("get_column_profile", args)
    assert cache.get(("job", "get_column_profile", '{"table":"orders","column":"amount"}')) == first
    assert registry.call("get_column_profile", args) == first
    tiny = ToolRegistry(ctx, ToolResultCache(), 10).call("get_column_profile", args)
    assert tiny.endswith(TRUNCATED) and len(tiny) == 40
