from __future__ import annotations

import numpy as np
import pytest

from synthgen.domain.errors import EngineError
from synthgen.engine.hashing import entity_hash
from synthgen.engine.keys import children_per_parent, expected_children
from synthgen.engine.scenarios import assign_scenarios, categories, scenario_counts
from synthgen.spec.models import TYPICAL, PerParentRows, ScenarioSpec
from synthgen.spec.shapes import ChildCount


def _rule(target: int | None = None) -> PerParentRows:
    return PerParentRows(
        kind="per_parent",
        parent_table="p",
        parent_column="id",
        child_column="p_id",
        counts=[
            ChildCount(children=0, weight=1),
            ChildCount(children=1, weight=1),
            ChildCount(children=4, weight=2),
        ],
        target_rows=target,
    )


@pytest.mark.parametrize("parents", [1, 7, 100, 10_000])
def test_children_total_is_exact(parents: int) -> None:
    counts = children_per_parent(_rule(), parents, np.random.default_rng(0), use_target=True)
    assert counts.size == parents
    assert (
        counts.sum()
        == expected_children(_rule(), parents, use_target=True)
        == round(parents * 2.25)
    )
    assert np.min(counts) >= 0


def test_children_follow_distribution() -> None:
    counts = children_per_parent(_rule(), 8_000, np.random.default_rng(1), use_target=True)
    shares = {c: np.mean(counts == c) for c in (0, 1, 4)}
    assert shares[0] == pytest.approx(0.25, abs=0.01)
    assert shares[4] == pytest.approx(0.5, abs=0.01)


@pytest.mark.parametrize("target", [0, 10, 5_000])
def test_children_hit_target_unless_sampling(target: int) -> None:
    rule = _rule(target)
    assert children_per_parent(rule, 100, np.random.default_rng(2), use_target=True).sum() == target
    assert children_per_parent(rule, 100, np.random.default_rng(2), use_target=False).sum() == 225


def test_children_without_parents() -> None:
    assert children_per_parent(_rule(), 0, np.random.default_rng(0), use_target=True).size == 0
    with pytest.raises(EngineError):
        children_per_parent(_rule(5), 0, np.random.default_rng(0), use_target=True)


def test_entity_hash_is_seeded_and_stable() -> None:
    ints = np.arange(1, 1_000)
    assert np.array_equal(entity_hash(ints, 3), entity_hash(ints, 3))
    assert not np.array_equal(entity_hash(ints, 3), entity_hash(ints, 4))
    assert len(set(entity_hash(ints, 3).tolist())) == ints.size
    texts = np.array(["a", "b", "c"], dtype=object)
    assert entity_hash(texts, 1).tolist() == entity_hash(texts, 1).tolist()


def _scenario(name: str, share: float) -> ScenarioSpec:
    return ScenarioSpec.model_validate(
        {
            "name": name,
            "category": "rare",
            "share": share,
            "unit": "entity",
            "entity": "t",
            "labels": [],
            "correlations": [],
            "cardinality": [],
            "sequences": [],
            "overrides": [],
        }
    )


def test_scenario_counts_are_exact_with_minimum_one() -> None:
    assert scenario_counts([0.05, 0.001, 0.0], 100) == [5, 1, 0]
    with pytest.raises(EngineError):
        scenario_counts([0.6, 0.6], 10)


def test_assignment_is_exact_and_skips_inherited_rows() -> None:
    keys = np.arange(1_000)
    current = np.full(1_000, TYPICAL, dtype=object)
    current[:100] = "inherited"
    assigned = assign_scenarios(keys, current, [_scenario("a", 0.1), _scenario("b", 0.02)], seed=5)
    assert np.sum(assigned == "a") == 90
    assert np.sum(assigned == "b") == 18
    assert np.all(assigned[:100] == "inherited")
    again = assign_scenarios(keys, current, [_scenario("a", 0.1), _scenario("b", 0.02)], seed=5)
    assert np.array_equal(assigned, again)


def test_categories_label_rows() -> None:
    assigned = np.array(["a", TYPICAL], dtype=object)
    assert categories(assigned, [_scenario("a", 0.5)]).tolist() == ["rare", TYPICAL]
