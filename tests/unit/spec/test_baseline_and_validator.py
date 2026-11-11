from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from synthgen.config import GenerationSettings, ProfilingSettings, load_settings
from synthgen.connectors.sqlite import SqliteConnector
from synthgen.domain.semantic import SemanticMap
from synthgen.metadata.graph import build_table_set
from synthgen.metadata.provider import SourceDoc, YamlMetadataProvider
from synthgen.profiling.profiler import profile_table
from synthgen.spec.baseline import BaselineInputs, compile_baseline
from synthgen.spec.merge import merge_overlays
from synthgen.spec.models import GenerationSpec, ScenarioSpec
from synthgen.spec.validator import validate_spec

PROFILE_NOW = datetime(2026, 1, 1, tzinfo=UTC)

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def generation_settings() -> GenerationSettings:
    return load_settings(ROOT / "config" / "settings.yaml", {}).generation


def _semantic(label: str | None) -> SemanticMap:
    return SemanticMap.model_validate(
        {
            "tables": [
                {
                    "table": "orders",
                    "grain": "event",
                    "entity_key": "customer_id",
                    "event_time": "placed",
                    "label_column": label,
                    "columns": [],
                }
            ],
            "business_rules": [],
        }
    )


@pytest.fixture(scope="module")
def baseline(
    shop_db: Path, profiling_settings: ProfilingSettings, generation_settings: GenerationSettings
) -> GenerationSpec:
    connector = SqliteConnector(shop_db)
    provider = YamlMetadataProvider(SourceDoc(source="shop"), connector)
    rels = provider.relationships()
    table_set = build_table_set(["orders"], [t.name for t in provider.tables()], rels)
    profiles = {
        t: profile_table(
            "shop",
            connector,
            provider.table(t),
            rels,
            settings=profiling_settings,
            seed=1,
            now=PROFILE_NOW,
        )
        for t in table_set.order
    }
    inputs = BaselineInputs(
        table_set=table_set,
        tables={t: provider.table(t) for t in table_set.order},
        profiles=profiles,
        profile_versions=dict.fromkeys(table_set.order, "v1"),
        semantic_map=_semantic("paid"),
        row_targets={"customers": 50},
        seed=9,
    )
    spec = compile_baseline(inputs, generation_settings)
    connector.close()
    return spec


def _gen(spec: GenerationSpec, table: str, column: str) -> dict[str, Any]:
    found = spec.table(table)
    assert found is not None
    col = found.column(column)
    assert col is not None
    return col.generator.model_dump()


def test_baseline_generators(baseline: GenerationSpec) -> None:
    assert [t.name for t in baseline.tables] == ["customers", "orders"]
    assert _gen(baseline, "customers", "id")["type"] == "sequence"
    assert _gen(baseline, "customers", "email")["type"] == "pattern"
    tier = _gen(baseline, "customers", "tier")
    assert tier["type"] == "categorical"
    assert set(tier["values"]) == {"gold", "silver", "bronze"}
    assert sum(tier["weights"]) == pytest.approx(1.0)
    assert _gen(baseline, "customers", "joined")["granularity"] == "date"
    assert _gen(baseline, "customers", "score")["type"] == "empirical"
    assert _gen(baseline, "orders", "customer_id")["type"] == "foreign_key"
    assert _gen(baseline, "orders", "placed")["granularity"] == "datetime"
    assert _gen(baseline, "orders", "paid") == {"type": "constant", "value": 0}


def test_baseline_row_rules(baseline: GenerationSpec) -> None:
    customers, orders = baseline.tables
    assert customers.row_rule.kind == "fixed"
    assert customers.row_rule.rows == 50
    rule = orders.row_rule
    assert rule.kind == "per_parent"
    assert rule.mean_children == pytest.approx(1.5)
    assert rule.target_rows is None


def test_baseline_marks_sensitive_and_nullable(baseline: GenerationSpec) -> None:
    customers = baseline.table("customers")
    assert customers is not None
    email = customers.column("email")
    tier = customers.column("tier")
    assert email is not None and email.sensitive and email.type == "text"
    assert tier is not None and tier.null_rate == pytest.approx(0.1)


def test_valid_baseline_has_no_errors(
    baseline: GenerationSpec, generation_settings: GenerationSettings
) -> None:
    assert validate_spec(baseline, generation_settings) == []


def _with_overlay(baseline: GenerationSpec, **changes: Any) -> GenerationSpec:
    overlay = {
        "name": "big_spenders",
        "category": "target_scenario",
        "share": 0.1,
        "unit": "entity",
        "entity": "customers",
        "labels": [{"table": "orders", "column": "paid", "value": 1}],
        "overrides": [
            {
                "table": "orders",
                "column": "amount",
                "generator": {"type": "empirical", "quantiles": [500.0, 900.0], "decimals": 2},
            }
        ],
    }
    overlay.update(changes)
    return merge_overlays(baseline, [ScenarioSpec.model_validate(overlay)])


def test_valid_overlay_has_no_errors(
    baseline: GenerationSpec, generation_settings: GenerationSettings
) -> None:
    merged = _with_overlay(baseline)
    assert len(merged.scenarios) == 1
    assert validate_spec(merged, generation_settings) == []


@pytest.mark.parametrize(
    ("changes", "fragment"),
    [
        ({"entity": "ghosts"}, "unknown table ghosts"),
        ({"labels": [{"table": "orders", "column": "nope", "value": 1}]}, "unknown column"),
        (
            {
                "overrides": [
                    {
                        "table": "customers",
                        "column": "email",
                        "generator": {"type": "categorical", "values": ["x"], "weights": [1.0]},
                    }
                ]
            },
            "sensitive column email",
        ),
        (
            {
                "overrides": [
                    {
                        "table": "orders",
                        "column": "placed",
                        "generator": {"type": "pattern", "masks": ["A"], "weights": [1.0]},
                    }
                ]
            },
            "does not suit",
        ),
    ],
)
def test_overlay_errors_are_attributed(
    baseline: GenerationSpec,
    generation_settings: GenerationSettings,
    changes: dict[str, Any],
    fragment: str,
) -> None:
    errors = validate_spec(_with_overlay(baseline, **changes), generation_settings)
    assert errors
    assert all(e.origin == "overlay" and e.scenario == "big_spenders" for e in errors)
    assert any(fragment in e.message for e in errors)


def test_shares_above_one_fail(
    baseline: GenerationSpec, generation_settings: GenerationSettings
) -> None:
    first = _with_overlay(baseline, share=0.7).scenarios[0]
    second = first.model_copy(update={"name": "second"})
    errors = validate_spec(merge_overlays(baseline, [first, second]), generation_settings)
    assert any("above 1" in e.message for e in errors)


def test_baseline_defects_are_attributed_to_baseline(
    baseline: GenerationSpec, generation_settings: GenerationSettings
) -> None:
    reordered = baseline.model_copy(update={"tables": list(reversed(baseline.tables))})
    errors = validate_spec(reordered, generation_settings)
    assert errors
    assert {e.origin for e in errors} == {"baseline"}
    assert any("not earlier" in e.message or "not generated earlier" in e.message for e in errors)


def test_declared_unique_text_uses_masks_and_observed_uniqueness_is_ignored(
    tmp_path: Path, profiling_settings: ProfilingSettings, generation_settings: GenerationSettings
) -> None:
    path = tmp_path / "names.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            "CREATE TABLE shops (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, city TEXT)"
        )
        conn.executemany(
            "INSERT INTO shops VALUES (?, ?, ?)",
            [(i, f"Shop {i}", f"City {i}") for i in range(1, 21)],
        )
        conn.commit()
    with closing(SqliteConnector(path)) as connector:
        provider = YamlMetadataProvider(SourceDoc(source="names"), connector)
        profile = profile_table(
            "names",
            connector,
            provider.table("shops"),
            [],
            settings=profiling_settings,
            seed=1,
            now=PROFILE_NOW,
        )
        table_set = build_table_set(["shops"], ["shops"], [])
        spec = compile_baseline(
            BaselineInputs(
                table_set=table_set,
                tables={"shops": provider.table("shops")},
                profiles={"shops": profile},
                profile_versions={"shops": "v"},
                semantic_map=None,
                row_targets={"shops": 500},
                seed=1,
            ),
            generation_settings,
        )
    name = spec.tables[0].column("name")
    city = spec.tables[0].column("city")
    assert name is not None and name.unique
    assert name.generator.type == "pattern"
    assert city is not None and not city.unique  # distinct in the sample, but not declared unique
    assert city.generator.type == "categorical"
