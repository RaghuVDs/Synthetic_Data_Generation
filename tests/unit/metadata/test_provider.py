from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from synthgen.connectors.sqlite import SqliteConnector
from synthgen.domain.errors import MetadataError, UnknownIdentifierError
from synthgen.domain.metadata import RelationshipOrigin, Sensitivity
from synthgen.metadata.provider import SourceDoc, YamlMetadataProvider, load_source_doc


@pytest.fixture(scope="module")
def connector(tmp_path_factory: pytest.TempPathFactory) -> SqliteConnector:
    path = tmp_path_factory.mktemp("meta") / "shop.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE customers (id INTEGER PRIMARY KEY, email TEXT NOT NULL);
        CREATE TABLE stores (id INTEGER PRIMARY KEY, city TEXT);
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL REFERENCES customers(id),
            store_id INTEGER REFERENCES stores(id),
            referrer_id INTEGER REFERENCES orders(id)
        );
        """
    )
    conn.close()
    return SqliteConnector(path)


def _doc(**overrides: Any) -> SourceDoc:
    data: dict[str, Any] = {
        "source": "shop",
        "tables": {
            "customers": {
                "description": "People who buy",
                "columns": {"email": {"description": "Contact", "sensitivity": "sensitive"}},
            }
        },
        "relationships": [{"child": "orders.customer_id", "parent": "customers.id"}],
    }
    data.update(overrides)
    return SourceDoc.model_validate(data)


def test_tables_merge_database_and_yaml(connector: SqliteConnector) -> None:
    provider = YamlMetadataProvider(_doc(), connector)
    assert [t.name for t in provider.tables()] == ["customers", "orders", "stores"]
    customers = provider.table("customers")
    assert customers.description == "People who buy"
    assert customers.primary_key == ("id",)
    email = customers.column("email")
    assert email is not None
    assert email.sensitivity is Sensitivity.SENSITIVE
    assert not email.nullable
    assert provider.table("stores").description == ""


def test_pragma_fills_missing_relationships(connector: SqliteConnector) -> None:
    provider = YamlMetadataProvider(_doc(), connector)
    by_column = {r.child_column: r for r in provider.relationships()}
    assert by_column["customer_id"].origin is RelationshipOrigin.METADATA
    assert by_column["store_id"].origin is RelationshipOrigin.PRAGMA
    assert by_column["store_id"].nullable
    assert by_column["referrer_id"].is_self_reference
    assert len(by_column) == 3


def test_default_driver_is_first_non_nullable(connector: SqliteConnector) -> None:
    provider = YamlMetadataProvider(_doc(relationships=[]), connector)
    drivers = [r.child_column for r in provider.relationships() if r.drives_cardinality]
    assert drivers == ["customer_id"]


def test_explicit_driver_wins(connector: SqliteConnector) -> None:
    doc = _doc(
        relationships=[
            {"child": "orders.customer_id", "parent": "customers.id"},
            {"child": "orders.store_id", "parent": "stores.id", "drives_cardinality": True},
        ]
    )
    drivers = [
        r.child_column
        for r in YamlMetadataProvider(doc, connector).relationships()
        if r.drives_cardinality
    ]
    assert drivers == ["store_id"]


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"tables": {"ghost": {}}}, UnknownIdentifierError),
        ({"tables": {"customers": {"columns": {"ghost": {}}}}}, UnknownIdentifierError),
        ({"relationships": [{"child": "orders.ghost", "parent": "customers.id"}]}, MetadataError),
        ({"relationships": [{"child": "orders.customer_id", "parent": "x.id"}]}, MetadataError),
        (
            {
                "relationships": [
                    {
                        "child": "orders.customer_id",
                        "parent": "customers.id",
                        "drives_cardinality": True,
                    },
                    {"child": "orders.store_id", "parent": "stores.id", "drives_cardinality": True},
                ]
            },
            MetadataError,
        ),
        (
            {
                "relationships": [
                    {
                        "child": "orders.referrer_id",
                        "parent": "orders.id",
                        "drives_cardinality": True,
                    }
                ]
            },
            MetadataError,
        ),
    ],
)
def test_disagreements_fail_loudly(
    connector: SqliteConnector, overrides: dict[str, Any], error: type[Exception]
) -> None:
    with pytest.raises(error):
        YamlMetadataProvider(_doc(**overrides), connector)


def test_unknown_table_lookup(connector: SqliteConnector) -> None:
    with pytest.raises(UnknownIdentifierError):
        YamlMetadataProvider(_doc(), connector).table("ghost")


def test_load_source_doc(tmp_path: Path) -> None:
    good = tmp_path / "good.yaml"
    good.write_text("source: shop\ntables:\n  customers:\n    description: x\n", encoding="utf-8")
    assert load_source_doc(good).tables["customers"].description == "x"
    bad = tmp_path / "bad.yaml"
    bad.write_text("source: shop\nunexpected: 1\n", encoding="utf-8")
    with pytest.raises(MetadataError):
        load_source_doc(bad)
    with pytest.raises(MetadataError):
        load_source_doc(tmp_path / "absent.yaml")
