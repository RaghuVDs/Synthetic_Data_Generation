from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from synthgen.engine.runner import RunOptions, generate_dataset
from synthgen.engine.writer import read_manifest
from synthgen.spec.models import GenerationSpec


def _col(name: str, ctype: str, generator: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "name": name,
        "type": ctype,
        "nullable": extra.get("nullable", False),
        "null_rate": extra.get("null_rate", 0.0),
        "primary_key": extra.get("primary_key", False),
        "unique": extra.get("unique", False),
        "sensitive": False,
        "generator": generator,
    }


def make_spec(
    customers: int = 200, scenarios: list[dict[str, Any]] | None = None
) -> GenerationSpec:
    return GenerationSpec.model_validate(
        {
            "version": 1,
            "seed": 11,
            "profile_versions": [],
            "tables": [
                {
                    "name": "customers",
                    "row_rule": {"kind": "fixed", "rows": customers},
                    "columns": [
                        _col(
                            "id",
                            "integer",
                            {"type": "sequence", "start": 1, "step": 1},
                            primary_key=True,
                        ),
                        _col(
                            "code",
                            "text",
                            {"type": "pattern", "masks": ["AAA-9999"], "weights": [1.0]},
                            unique=True,
                        ),
                        _col(
                            "tier",
                            "text",
                            {"type": "categorical", "values": ["a", "b"], "weights": [0.5, 0.5]},
                            nullable=True,
                            null_rate=0.2,
                        ),
                    ],
                },
                {
                    "name": "orders",
                    "row_rule": {
                        "kind": "per_parent",
                        "parent_table": "customers",
                        "parent_column": "id",
                        "child_column": "customer_id",
                        "counts": [{"children": 1, "weight": 1.0}, {"children": 3, "weight": 1.0}],
                        "target_rows": None,
                    },
                    "columns": [
                        _col(
                            "id",
                            "integer",
                            {"type": "sequence", "start": 1, "step": 1},
                            primary_key=True,
                        ),
                        _col(
                            "customer_id",
                            "integer",
                            {
                                "type": "foreign_key",
                                "parent_table": "customers",
                                "parent_column": "id",
                                "distribution": "uniform",
                            },
                        ),
                        _col(
                            "amount",
                            "float",
                            {"type": "empirical", "quantiles": [1.0, 100.0], "decimals": 2},
                        ),
                        _col("flag", "boolean", {"type": "constant", "value": 0}),
                    ],
                },
            ],
            "scenarios": scenarios or [],
            "output": {"compression": "zstd", "chunk_rows": 250_000},
        }
    )


BIG = {
    "name": "big",
    "category": "target_scenario",
    "share": 0.1,
    "unit": "entity",
    "entity": "customers",
    "labels": [{"table": "orders", "column": "flag", "value": 1}],
    "overrides": [
        {
            "table": "orders",
            "column": "amount",
            "generator": {"type": "empirical", "quantiles": [1_000.0, 2_000.0], "decimals": 2},
        }
    ],
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rerun_is_byte_identical(tmp_path: Path) -> None:
    spec = make_spec(scenarios=[BIG])
    generate_dataset(spec, tmp_path / "a", RunOptions())
    generate_dataset(spec, tmp_path / "b", RunOptions())
    for table in ("customers", "orders"):
        part = Path("data") / table / "part-00000.parquet"
        assert _digest(tmp_path / "a" / part) == _digest(tmp_path / "b" / part)


def test_rows_keys_and_nulls(tmp_path: Path) -> None:
    manifest = generate_dataset(make_spec(), tmp_path, RunOptions())
    customers = pq.read_table(tmp_path / "data" / "customers")
    orders = pq.read_table(tmp_path / "data" / "orders")
    assert customers.num_rows == 200
    assert orders.num_rows == 400
    assert set(orders.column("customer_id").to_pylist()) <= set(customers.column("id").to_pylist())
    assert len(set(customers.column("code").to_pylist())) == 200
    assert 0.1 < customers.column("tier").null_count / 200 < 0.3
    assert read_manifest(tmp_path) == manifest
    assert manifest.spec_hash == make_spec().spec_hash()


def test_scenarios_are_assigned_inherited_and_overlaid(tmp_path: Path) -> None:
    manifest = generate_dataset(make_spec(scenarios=[BIG]), tmp_path, RunOptions())
    customers = pq.read_table(tmp_path / "data" / "customers")
    orders = pq.read_table(tmp_path / "data" / "orders")
    assert customers.column("_scenario").to_pylist().count("big") == 20
    big_customers = set(
        customers.filter(pc.equal(customers["_scenario"], pa.scalar("big")))
        .column("id")
        .to_pylist()
    )
    big_orders = orders.filter(pc.equal(orders["_scenario"], pa.scalar("big")))
    assert set(big_orders.column("customer_id").to_pylist()) == big_customers
    assert pc.min(big_orders["amount"]).as_py() >= 1_000.0
    assert set(big_orders.column("flag").to_pylist()) == {1}
    assert set(big_orders.column("_category").to_pylist()) == {"target_scenario"}
    typical = orders.filter(pc.equal(orders["_scenario"], pa.scalar("typical")))
    assert set(typical.column("flag").to_pylist()) == {0}
    counts = {c.scenario: c.rows for c in manifest.tables[0].scenario_counts}
    assert counts == {"big": 20, "typical": 180}


def test_dry_run_caps_root_rows(tmp_path: Path) -> None:
    manifest = generate_dataset(make_spec(customers=10_000), tmp_path, RunOptions(max_root_rows=50))
    assert [t.rows for t in manifest.tables] == [50, 100]


@pytest.mark.parametrize("customers", [1, 3])
def test_tiny_runs_still_carry_every_scenario(tmp_path: Path, customers: int) -> None:
    manifest = generate_dataset(
        make_spec(customers=customers, scenarios=[BIG]), tmp_path, RunOptions()
    )
    counts = {c.scenario: c.rows for c in manifest.tables[0].scenario_counts}
    assert counts["big"] == 1
