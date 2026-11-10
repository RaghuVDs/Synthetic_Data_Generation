from __future__ import annotations

import json
from pathlib import Path

import pytest

from synthgen.config import ProfilingSettings
from synthgen.connectors.sqlite import SqliteConnector
from synthgen.domain.metadata import Sensitivity
from synthgen.metadata.provider import SourceDoc, YamlMetadataProvider
from synthgen.profiling.card import build_card
from synthgen.profiling.models import InferredType, StatSource, TableProfile
from synthgen.profiling.profiler import profile_table, schema_hash


@pytest.fixture(scope="module")
def provider(shop_db: Path) -> YamlMetadataProvider:
    doc = SourceDoc.model_validate(
        {
            "source": "shop",
            "tables": {"customers": {"columns": {"score": {"sensitivity": "internal"}}}},
        }
    )
    return YamlMetadataProvider(doc, SqliteConnector(shop_db))


def _profile(
    shop_db: Path, provider: YamlMetadataProvider, table: str, settings: ProfilingSettings
) -> TableProfile:
    return profile_table(
        "shop",
        SqliteConnector(shop_db),
        provider.table(table),
        provider.relationships(),
        settings=settings,
        seed=1,
    )


def test_customer_columns(
    shop_db: Path, provider: YamlMetadataProvider, profiling_settings: ProfilingSettings
) -> None:
    profile = _profile(shop_db, provider, "customers", profiling_settings)
    assert profile.row_count.value == 300
    assert profile.row_count.source is StatSource.SQL
    assert profile.primary_key_unique is not None and profile.primary_key_unique.value
    columns = {c.name: c for c in profile.columns}
    email = columns["email"]
    assert email.semantic_type == "email"
    assert email.sensitivity is Sensitivity.SENSITIVE
    assert email.top_value is None and email.categorical is None and email.text is not None
    tier = columns["tier"]
    assert tier.null_rate.value == pytest.approx(30 / 300)
    assert tier.categorical is not None
    assert tier.categorical.frequencies.source is StatSource.SQL
    assert {v.value for v in tier.categorical.frequencies.value} == {"gold", "silver", "bronze"}
    assert columns["joined"].inferred_type is InferredType.DATE
    assert columns["joined"].temporal is not None
    assert columns["score"].numeric is not None
    assert columns["score"].sensitivity is Sensitivity.INTERNAL


def test_order_columns_and_children(
    shop_db: Path, provider: YamlMetadataProvider, profiling_settings: ProfilingSettings
) -> None:
    profile = _profile(shop_db, provider, "orders", profiling_settings)
    columns = {c.name: c for c in profile.columns}
    assert columns["paid"].inferred_type is InferredType.BOOLEAN
    assert columns["placed"].inferred_type is InferredType.DATETIME
    [shape] = profile.relationships
    assert shape.parent_table == "customers"
    assert dict(shape.counts.value) == {0: 75, 1: 75, 2: 75, 3: 75}
    assert shape.orphan_rows.value == 0


def test_schema_hash_tracks_columns(provider: YamlMetadataProvider) -> None:
    customers = provider.table("customers")
    renamed = customers.model_copy(update={"name": "other"})
    assert schema_hash(customers) == schema_hash(customers)
    assert schema_hash(customers) != schema_hash(renamed)


def test_card_hides_sensitive_values(
    shop_db: Path, provider: YamlMetadataProvider, profiling_settings: ProfilingSettings
) -> None:
    profile = _profile(shop_db, provider, "customers", profiling_settings)
    card = build_card(
        profile, provider.table("customers"), provider.relationships(), profiling_settings
    )
    text = card.model_dump_json()
    emails = [f"person{i}@mail.example" for i in range(1, 301)]
    assert not any(e in text for e in emails)
    assert "person" not in text
    email = next(c for c in card.columns if c.name == "email")
    assert email.mask is not None and "@" in email.mask
    assert email.top_values is None and email.five_number is None
    assert email.length_range is not None
    tier = next(c for c in card.columns if c.name == "tier")
    assert tier.top_values is not None and len(tier.top_values) == 3
    assert card.relationships == ["orders.customer_id -> customers.id"]
    assert len(json.dumps(card.model_dump())) < 4_000


def test_card_folds_rare_internal_values(
    shop_db: Path, provider: YamlMetadataProvider, profiling_settings: ProfilingSettings
) -> None:
    profile = _profile(shop_db, provider, "customers", profiling_settings)
    tier = next(c for c in profile.columns if c.name == "tier")
    assert tier.categorical is not None
    internal = tier.model_copy(update={"sensitivity": Sensitivity.INTERNAL})
    rare = tier.categorical.frequencies.value[0].model_copy(update={"count": 3})
    frequencies = tier.categorical.frequencies.model_copy(update={"value": [rare]})
    internal = internal.model_copy(
        update={"categorical": tier.categorical.model_copy(update={"frequencies": frequencies})}
    )
    folded = profile.model_copy(update={"columns": [internal]})
    card = build_card(folded, provider.table("customers"), [], profiling_settings)
    assert card.columns[0].top_values == []
