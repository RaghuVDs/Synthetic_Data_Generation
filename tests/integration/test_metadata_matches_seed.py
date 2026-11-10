from __future__ import annotations

from contextlib import closing
from pathlib import Path

import pytest

from synthgen.connectors.sqlite import SqliteConnector
from synthgen.domain.metadata import RelationshipOrigin
from synthgen.metadata.graph import build_table_set
from synthgen.metadata.provider import YamlMetadataProvider, load_source_doc

SOURCES = ["finance", "orders"]
METADATA_DIR = Path(__file__).resolve().parents[2] / "config" / "metadata"


def provider(seeded_dir: Path, source: str) -> YamlMetadataProvider:
    doc = load_source_doc(METADATA_DIR / f"{source}.yaml")
    with closing(SqliteConnector(seeded_dir / f"{source}.db")) as connector:
        return YamlMetadataProvider(doc, connector)


@pytest.mark.parametrize("source", SOURCES)
def test_every_column_is_described_and_tagged(seeded_dir: Path, source: str) -> None:
    for table in provider(seeded_dir, source).tables():
        assert table.description, table.name
        for column in table.columns:
            assert column.description, f"{table.name}.{column.name}"
            assert column.sensitivity is not None, f"{table.name}.{column.name}"


@pytest.mark.parametrize("source", SOURCES)
def test_every_child_has_one_cardinality_driver(seeded_dir: Path, source: str) -> None:
    relationships = provider(seeded_dir, source).relationships()
    children = {r.child_table for r in relationships}
    for child in children:
        drivers = [r for r in relationships if r.child_table == child and r.drives_cardinality]
        assert len(drivers) == 1, child


def test_orders_product_link_comes_from_the_database(seeded_dir: Path) -> None:
    by_label = {r.label(): r for r in provider(seeded_dir, "orders").relationships()}
    product = by_label["order_items.product_id -> products.product_id"]
    assert product.origin is RelationshipOrigin.PRAGMA
    assert not product.drives_cardinality
    assert by_label["order_items.order_id -> orders.order_id"].drives_cardinality


def test_finance_sensitive_columns(seeded_dir: Path) -> None:
    meta = provider(seeded_dir, "finance")
    sensitive = {
        f"{t.name}.{c.name}"
        for t in meta.tables()
        for c in t.columns
        if c.sensitivity == "sensitive"
    }
    assert sensitive == {
        "customers.first_name",
        "customers.last_name",
        "customers.email",
        "customers.phone",
        "customers.date_of_birth",
        "cards.card_number",
    }


@pytest.mark.parametrize(
    ("source", "selected", "order"),
    [
        (
            "finance",
            ["transactions"],
            ("customers", "accounts", "cards", "merchants", "transactions"),
        ),
        (
            "orders",
            ["shipments", "order_items"],
            ("customers", "orders", "products", "order_items", "shipments"),
        ),
    ],
)
def test_table_sets_for_seed_sources(
    seeded_dir: Path, source: str, selected: list[str], order: tuple[str, ...]
) -> None:
    meta = provider(seeded_dir, source)
    table_set = build_table_set(selected, [t.name for t in meta.tables()], meta.relationships())
    assert table_set.order == order
    assert table_set.deferred == ()
