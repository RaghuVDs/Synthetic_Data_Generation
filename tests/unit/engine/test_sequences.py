from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest

from synthgen.engine.keys import plan_children
from synthgen.engine.runner import RunOptions, generate_dataset
from synthgen.engine.sequences import order_events, step_of_rank
from synthgen.spec.merge import merge_overlays
from synthgen.spec.models import GenerationSpec, ScenarioSpec
from synthgen.spec.shapes import SequenceSpec
from synthgen.spec.validator import validate_spec

from .test_runner import BIG, GENERATION, _col, make_spec

STEPS = [
    {
        "count": 3,
        "gap_min_seconds": 60,
        "gap_max_seconds": 120,
        "columns": [{"column": "state", "generator": {"type": "constant", "value": "probe"}}],
    },
    {
        "count": 1,
        "gap_min_seconds": 3_600,
        "gap_max_seconds": 7_200,
        "columns": [{"column": "state", "generator": {"type": "constant", "value": "cashout"}}],
    },
]
SEQUENCE = {"table": "orders", "entity_key": "customer_id", "time_column": "placed", "steps": STEPS}


def _spec(customers: int = 2_000, **scenario: Any) -> GenerationSpec:
    data = make_spec(customers=customers).model_dump()
    data["tables"][1]["columns"] += [
        _col(
            "placed",
            "datetime",
            {
                "type": "datetime",
                "start": "2024-01-01 00:00:00",
                "end": "2024-12-31 23:59:59",
                "granularity": "datetime",
                "weekday_weights": None,
                "hour_weights": None,
                "after": None,
            },
        ),
        _col("state", "text", {"type": "constant", "value": "normal"}),
    ]
    baseline = GenerationSpec.model_validate(data)
    overlay = ScenarioSpec.model_validate({**BIG, "labels": [], "overrides": [], **scenario})
    return merge_overlays(baseline, [overlay])


def _orders(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path / "data" / "orders" / "part-00000.parquet").to_pylist()


def test_step_of_rank_and_event_order() -> None:
    sequence = SequenceSpec.model_validate(SEQUENCE)
    assert step_of_rank(sequence, np.array([0, 2, 3, 9])).tolist() == [0, 0, 1, 1]
    order, ranks = order_events(
        np.array(["b", "a", "b", "a"], dtype=object), np.array([5.0, 9.0, 1.0, 2.0])
    )
    assert order.tolist() == [3, 1, 2, 0]
    assert ranks.tolist() == [0, 1, 0, 1]


def test_sequence_lays_out_ordered_events_with_a_state_path(tmp_path: Path) -> None:
    spec = _spec(sequences=[SEQUENCE])
    assert validate_spec(spec, GENERATION) == []
    generate_dataset(spec, tmp_path, RunOptions(), GENERATION)
    orders = _orders(tmp_path)
    per_customer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in orders:
        if row["_scenario"] == "big":
            per_customer[row["customer_id"]].append(row)
    assert len(per_customer) == 200
    for events in per_customer.values():
        events.sort(key=lambda r: r["placed"])
        assert [r["state"] for r in events] == ["probe", "probe", "probe", "cashout"]
        stamps = [datetime.fromisoformat(r["placed"]) for r in events]
        gaps = [(b - a).total_seconds() for a, b in pairwise(stamps)]
        assert all(59 <= g <= 121 for g in gaps[:2])
        assert 3_599 <= gaps[2] <= 7_201
    assert all(r["state"] == "normal" for r in orders if r["_scenario"] == "typical")
    assert len(orders) == 4_000  # 2,000 customers x 2 children on average


def test_cardinality_can_remove_children_and_keeps_the_total(tmp_path: Path) -> None:
    spec = _spec(cardinality=[{"table": "orders", "counts": [{"children": 0, "weight": 1}]}])
    assert validate_spec(spec, GENERATION) == []
    generate_dataset(spec, tmp_path, RunOptions(), GENERATION)
    orders = _orders(tmp_path)
    assert not [r for r in orders if r["_scenario"] == "big"]
    assert len(orders) == 4_000


def test_plan_children_shapes_only_scenario_parents() -> None:
    spec = _spec(sequences=[SEQUENCE])
    rule = spec.tables[1].row_rule
    assert rule.kind == "per_parent"
    parents = np.array(["typical"] * 90 + ["big"] * 10, dtype=object)
    counts = plan_children(
        rule, "orders", parents, spec.scenarios, np.random.default_rng(1), use_target=True
    )
    assert counts[90:].tolist() == [4] * 10
    assert counts.sum() == 200
    plain = plan_children(
        rule, "accounts", parents, spec.scenarios, np.random.default_rng(1), use_target=True
    )
    assert plain.sum() == 200 and set(plain[90:].tolist()) <= {1, 2, 3, 4}


@pytest.mark.parametrize(
    ("changes", "fragment"),
    [
        (
            {"cardinality": [{"table": "customers", "counts": [{"children": 1, "weight": 1}]}]},
            "not sized by a parent",
        ),
        ({"entity": "orders", "sequences": [SEQUENCE]}, "does not hang below"),
        ({"sequences": [{**SEQUENCE, "entity_key": "id"}]}, "entity_key must be customer_id"),
        ({"sequences": [{**SEQUENCE, "time_column": "amount"}]}, "not a date or datetime"),
        ({"sequences": [SEQUENCE, SEQUENCE]}, "one sequence per table"),
        (
            {"sequences": [{**SEQUENCE, "steps": [{**STEPS[0], "count": 1_001}]}]},
            "at most 1000 events",
        ),
        (
            {
                "sequences": [
                    {
                        **SEQUENCE,
                        "steps": [
                            {
                                **STEPS[0],
                                "columns": [
                                    {
                                        "column": "placed",
                                        "generator": {"type": "constant", "value": None},
                                    }
                                ],
                            }
                        ],
                    }
                ]
            },
            "cannot set placed",
        ),
        (
            {
                "sequences": [
                    {
                        **SEQUENCE,
                        "steps": [
                            {
                                **STEPS[0],
                                "columns": [{"column": "state", "generator": {"type": "uuid"}}],
                            }
                        ],
                    }
                ]
            },
            "",
        ),
        (
            {
                "sequences": [
                    {
                        **SEQUENCE,
                        "steps": [
                            {
                                **STEPS[0],
                                "columns": [{"column": "amount", "generator": {"type": "uuid"}}],
                            }
                        ],
                    }
                ]
            },
            "uuid generator does not suit",
        ),
    ],
)
def test_shape_problems_are_overlay_errors(changes: dict[str, Any], fragment: str) -> None:
    errors = validate_spec(_spec(**changes), GENERATION)
    if not fragment:
        assert errors == []
        return
    assert errors and all(e.origin == "overlay" for e in errors)
    assert any(fragment in e.message for e in errors), errors


def test_gaps_must_be_positive_and_ordered() -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        SequenceSpec.model_validate({**SEQUENCE, "steps": [{**STEPS[0], "gap_min_seconds": 0}]})
    with pytest.raises(ValueError, match="below its minimum"):
        SequenceSpec.model_validate({**SEQUENCE, "steps": [{**STEPS[0], "gap_max_seconds": 1}]})
