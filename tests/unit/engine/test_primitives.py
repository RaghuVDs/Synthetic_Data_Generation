"""One test per primitive (SPEC 7.2): spec construct in, observable effect out.

A single spec carries one scenario per primitive and is generated once for the module.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from itertools import pairwise
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
from scipy import stats

from synthgen.domain.plan import Primitive
from synthgen.engine.runner import RunOptions, generate_dataset
from synthgen.spec.models import GenerationSpec
from synthgen.spec.validator import validate_spec

from .test_runner import GENERATION, _col

Rows = list[dict[str, Any]]
FK_NONE = {"exponent": None, "hot_keys": None, "hot_share": None, "group_size": None}
WINDOW = {
    "type": "datetime",
    "start": "2024-01-01 00:00:00",
    "end": "2024-12-31 23:59:59",
    "granularity": "datetime",
    "weekday_weights": None,
}


def _fk(parent: str, distribution: str = "uniform", **params: Any) -> dict[str, Any]:
    return {
        "type": "foreign_key",
        "parent_table": parent,
        "parent_column": "id",
        "distribution": distribution,
        **(FK_NONE | params),
    }


def _key(name: str) -> dict[str, Any]:
    return _col(name, "integer", {"type": "sequence", "start": 1, "step": 1}, primary_key=True)


def _table(name: str, rule: dict[str, Any], *columns: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "row_rule": rule,
        "columns": list(columns),
        "correlation": None,
        "constraints": [],
    }


def _scenario(name: str, entity: str, share: float, **parts: Any) -> dict[str, Any]:
    empty: dict[str, list[Any]] = {
        "labels": [],
        "overrides": [],
        "correlations": [],
        "cardinality": [],
        "sequences": [],
        "defects": [],
    }
    return {
        "name": name,
        "category": "target_scenario",
        "share": share,
        "unit": "entity",
        "entity": entity,
        **(empty | parts),
    }


def _override(column: str, generator: dict[str, Any], table: str = "orders") -> dict[str, Any]:
    return {"table": table, "column": column, "generator": generator}


def _steps(*states: str, gap: tuple[float, float] = (60, 120)) -> list[dict[str, Any]]:
    return [
        {
            "count": 1,
            "gap_min_seconds": gap[0],
            "gap_max_seconds": gap[1],
            "columns": [{"column": "status", "generator": {"type": "constant", "value": s}}],
        }
        for s in states
    ]


def _sequence(steps: list[dict[str, Any]]) -> dict[str, Any]:
    return {"table": "orders", "entity_key": "customer_id", "time_column": "placed", "steps": steps}


def _customers() -> dict[str, Any]:
    table = _table(
        "customers",
        {"kind": "fixed", "rows": 4_000},
        _key("id"),
        _col("tier", "text", {"type": "categorical", "values": ["a", "b"], "weights": [0.7, 0.3]}),
        _col(
            "income",
            "float",
            {"type": "empirical", "quantiles": [10.0, 50.0, 300.0], "decimals": 2},
        ),
        _col("score", "float", {"type": "empirical", "quantiles": [300.0, 850.0], "decimals": 0}),
    )
    table["correlation"] = {"columns": ["income", "score"], "spearman": [[1, 0.8], [0.8, 1]]}
    return table


def _orders() -> dict[str, Any]:
    rule = {
        "kind": "per_parent",
        "parent_table": "customers",
        "parent_column": "id",
        "child_column": "customer_id",
        "counts": [{"children": 1, "weight": 1.0}, {"children": 3, "weight": 1.0}],
        "target_rows": None,
    }
    return _table(
        "orders",
        rule,
        _key("id"),
        _col("customer_id", "integer", _fk("customers")),
        _col("merchant_id", "integer", _fk("merchants")),
        _col("amount", "float", {"type": "empirical", "quantiles": [1.0, 100.0], "decimals": 2}),
        _col(
            "channel",
            "text",
            {"type": "categorical", "values": ["web", "store"], "weights": [0.5, 0.5]},
        ),
        _col("placed", "datetime", WINDOW | {"hour_weights": None}),
        _col("status", "text", {"type": "constant", "value": "normal"}),
        _col("note", "text", {"type": "constant", "value": "ok"}),
    )


NIGHT = [0.0] * 24
NIGHT[3] = 1.0
SCENARIOS = [
    _scenario(
        "value_shift",
        "orders",
        0.05,
        overrides=[
            _override(
                "amount", {"type": "empirical", "quantiles": [5_000.0, 6_000.0], "decimals": 2}
            )
        ],
    ),
    _scenario(
        "mix_shift",
        "customers",
        0.05,
        overrides=[
            _override(
                "tier", {"type": "categorical", "values": ["b"], "weights": [1.0]}, "customers"
            )
        ],
    ),
    _scenario(
        "conditional_rule",
        "orders",
        0.05,
        overrides=[
            _override(
                "amount",
                {
                    "type": "conditional",
                    "column": "channel",
                    "cases": [{"values": ["web"], "generator": {"type": "constant", "value": 1.0}}],
                    "otherwise": {"type": "constant", "value": 2.0},
                },
            )
        ],
    ),
    _scenario(
        "sequence", "customers", 0.05, sequences=[_sequence(_steps("x", "x", "x", gap=(30, 60)))]
    ),
    _scenario(
        "timing_shift",
        "orders",
        0.05,
        overrides=[_override("placed", WINDOW | {"hour_weights": NIGHT})],
    ),
    _scenario(
        "gap_or_absence",
        "customers",
        0.05,
        cardinality=[{"table": "orders", "counts": [{"children": 0, "weight": 1}]}],
    ),
    _scenario(
        "relationship_shape",
        "customers",
        0.05,
        cardinality=[{"table": "orders", "counts": [{"children": 5, "weight": 1}]}],
    ),
    _scenario(
        "cross_entity_link",
        "orders",
        0.05,
        overrides=[_override("merchant_id", _fk("merchants", "shared_group", group_size=3))],
    ),
    _scenario(
        "state_path", "customers", 0.05, sequences=[_sequence(_steps("opened", "used", "blocked"))]
    ),
    _scenario(
        "correlation_change",
        "customers",
        0.2,
        correlations=[
            {
                "table": "customers",
                "columns": ["income", "score"],
                "spearman": [[1, -0.8], [-0.8, 1]],
            }
        ],
    ),
    _scenario(
        "boundary_value",
        "orders",
        0.05,
        overrides=[_override("amount", {"type": "boundary", "points": [0.0, 99_999.0]})],
    ),
    _scenario(
        "volume_skew",
        "orders",
        0.05,
        overrides=[
            _override("merchant_id", _fk("merchants", "weighted", hot_keys=2, hot_share=0.9))
        ],
    ),
    _scenario(
        "text_content",
        "orders",
        0.05,
        overrides=[
            _override("note", {"type": "text_pool", "values": ["parcel lost", "late again"]})
        ],
    ),
    _scenario(
        "rule_break",
        "orders",
        0.05,
        defects=[{"table": "orders", "column": "amount", "kind": "out_of_range", "rate": 1.0}],
    ),
]


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Rows]:
    merchants = _table("merchants", {"kind": "fixed", "rows": 200}, _key("id"))
    spec = GenerationSpec.model_validate(
        {
            "version": 2,
            "seed": 3,
            "profile_versions": [],
            "tables": [_customers(), merchants, _orders()],
            "scenarios": SCENARIOS,
            "output": {"compression": "zstd", "chunk_rows": 250_000},
        }
    )
    assert validate_spec(spec, GENERATION, allow_defects=True) == []
    out = tmp_path_factory.mktemp("primitives")
    generate_dataset(spec, out, RunOptions(), GENERATION)
    return {
        t: pq.read_table(out / "data" / t / "part-00000.parquet").to_pylist()
        for t in ("customers", "merchants", "orders")
    }


def _of(rows: Rows, scenario: str) -> Rows:
    picked = [r for r in rows if r["_scenario"] == scenario]
    assert picked, f"no rows for {scenario}"
    return picked


def _events(orders: Rows, scenario: str) -> list[Rows]:
    grouped: dict[int, Rows] = defaultdict(list)
    for row in _of(orders, scenario):
        grouped[row["customer_id"]].append(row)
    return [sorted(rows, key=lambda r: r["placed"]) for rows in grouped.values()]


def _gaps(events: Rows) -> list[float]:
    stamps = [datetime.fromisoformat(r["placed"]) for r in events]
    return [(b - a).total_seconds() for a, b in pairwise(stamps)]


def test_every_primitive_has_a_test() -> None:
    assert {s["name"] for s in SCENARIOS} == {p.value for p in Primitive}


def test_value_shift(generated: dict[str, Rows]) -> None:
    assert all(5_000 <= r["amount"] <= 6_000 for r in _of(generated["orders"], "value_shift"))


def test_mix_shift(generated: dict[str, Rows]) -> None:
    assert {r["tier"] for r in _of(generated["customers"], "mix_shift")} == {"b"}
    typical = Counter(r["tier"] for r in _of(generated["customers"], "typical"))
    assert typical["a"] > typical["b"]


def test_conditional_rule(generated: dict[str, Rows]) -> None:
    rows = _of(generated["orders"], "conditional_rule")
    assert all(r["amount"] == (1.0 if r["channel"] == "web" else 2.0) for r in rows)


def test_sequence(generated: dict[str, Rows]) -> None:
    for events in _events(generated["orders"], "sequence"):
        assert len(events) == 3
        assert all(29 <= g <= 61 for g in _gaps(events))


def test_timing_shift(generated: dict[str, Rows]) -> None:
    assert {r["placed"][11:13] for r in _of(generated["orders"], "timing_shift")} == {"03"}


def test_gap_or_absence(generated: dict[str, Rows]) -> None:
    absent = {r["id"] for r in _of(generated["customers"], "gap_or_absence")}
    assert not absent & {r["customer_id"] for r in generated["orders"]}


def test_relationship_shape(generated: dict[str, Rows]) -> None:
    assert {len(e) for e in _events(generated["orders"], "relationship_shape")} == {5}


def test_cross_entity_link(generated: dict[str, Rows]) -> None:
    rows = _of(generated["orders"], "cross_entity_link")
    assert len({r["merchant_id"] for r in rows}) == 3
    assert len({r["customer_id"] for r in rows}) > 3  # unrelated customers share merchants


def test_state_path(generated: dict[str, Rows]) -> None:
    for events in _events(generated["orders"], "state_path"):
        assert [r["status"] for r in events] == ["opened", "used", "blocked"]


def test_correlation_change(generated: dict[str, Rows]) -> None:
    def rho(name: str) -> float:
        rows = _of(generated["customers"], name)
        return float(
            stats.spearmanr([r["income"] for r in rows], [r["score"] for r in rows]).statistic
        )

    assert rho("correlation_change") == pytest.approx(-0.8, abs=0.06)
    assert rho("typical") == pytest.approx(0.8, abs=0.06)


def test_boundary_value(generated: dict[str, Rows]) -> None:
    assert {r["amount"] for r in _of(generated["orders"], "boundary_value")} <= {0.0, 99_999.0}


def test_volume_skew(generated: dict[str, Rows]) -> None:
    counts = Counter(r["merchant_id"] for r in _of(generated["orders"], "volume_skew"))
    top_two = sum(c for _, c in counts.most_common(2))
    assert top_two / sum(counts.values()) == pytest.approx(0.9, abs=0.06)


def test_text_content(generated: dict[str, Rows]) -> None:
    assert {r["note"] for r in _of(generated["orders"], "text_content")} == {
        "parcel lost",
        "late again",
    }


def test_rule_break(generated: dict[str, Rows]) -> None:
    rows = _of(generated["orders"], "rule_break")
    assert all(r["_defect"] == "out_of_range:amount" and r["amount"] > 1_000 for r in rows)
    assert all(r["_defect"] is None for r in _of(generated["orders"], "typical"))
    assert np.mean([r["amount"] for r in _of(generated["orders"], "typical")]) < 100
