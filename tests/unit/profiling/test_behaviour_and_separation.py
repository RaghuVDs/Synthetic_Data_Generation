from __future__ import annotations

import numpy as np
import pytest

from synthgen.profiling.behaviour import behaviour_stats, spread
from synthgen.profiling.stratified import class_separation, ks_statistic, total_variation

HOUR = 3_600


def test_behaviour_counts_gaps_and_bursts() -> None:
    entities = np.array([1, 1, 1, 2, 2])
    seconds = np.array([0, 600, 2 * HOUR, 0, 30 * HOUR])
    stats = behaviour_stats(entities, seconds, np.array([10.0, 20.0, 30.0, 5.0, 5.0]))
    assert (stats.entities, stats.events) == (2, 5)
    assert stats.events_per_entity.max == 3.0
    assert stats.inter_arrival_hours is not None
    assert stats.inter_arrival_hours.max == pytest.approx(30.0)
    assert stats.burst_1h.max == 2.0
    assert stats.burst_24h.max == 3.0
    assert stats.value_to_entity_median is not None
    assert stats.value_to_entity_median.max == pytest.approx(1.5)


def test_behaviour_without_repeat_events_or_measure() -> None:
    stats = behaviour_stats(np.array([1, 2]), np.array([0, 5]), None)
    assert stats.inter_arrival_hours is None
    assert stats.value_to_entity_median is None


def test_spread_of_nothing() -> None:
    assert spread(np.zeros(0)).max == 0.0


def test_ks_statistic() -> None:
    assert ks_statistic(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0
    assert ks_statistic(np.array([1.0, 2.0]), np.array([5.0, 6.0])) == 1.0
    assert ks_statistic(np.zeros(0), np.array([1.0])) == 0.0


def test_total_variation() -> None:
    assert total_variation(["a", "b"], ["a", "b"]) == 0.0
    assert total_variation(["a"], ["b"]) == 1.0
    assert total_variation([], ["b"]) == 0.0


def test_class_separation_ranks_columns() -> None:
    positive = np.array([True, True, False, False])
    ranked = class_separation(
        positive,
        {"loud": np.array([9.0, 9.0, 1.0, 1.0]), "quiet": np.array([1.0, 2.0, 1.0, 2.0])},
        {"kind": ["x", "x", "x", "y"]},
    )
    assert [s.column for s in ranked] == ["loud", "kind", "quiet"]
    assert ranked[0].score == 1.0
