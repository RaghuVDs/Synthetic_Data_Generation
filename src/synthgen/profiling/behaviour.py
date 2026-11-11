"""Behaviour per entity over time (SPEC 8.3 H): velocity, inter-arrival and bursts.

Pure functions over sampled arrays. M2/M3 compute these on demand for the planner's tools.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from synthgen.domain.base import StrictModel
from synthgen.profiling.shape_models import Spread

SECONDS_PER_HOUR = 3_600
BURST_WINDOWS_HOURS = (1, 24)
PERCENTILES = (50.0, 90.0, 99.0)


class BehaviourStats(StrictModel):
    """How entities act over time."""

    entities: int
    events: int
    events_per_entity: Spread
    inter_arrival_hours: Spread | None
    burst_1h: Spread
    burst_24h: Spread
    value_to_entity_median: Spread | None


def spread(values: NDArray[np.float64]) -> Spread:
    """Percentile summary; all zeros for an empty input."""
    if values.size == 0:
        return Spread(p50=0.0, p90=0.0, p99=0.0, max=0.0)
    p50, p90, p99 = (float(v) for v in np.percentile(values, PERCENTILES))
    return Spread(p50=p50, p90=p90, p99=p99, max=float(np.max(values)))


def behaviour_stats(
    entities: NDArray[np.int64], seconds: NDArray[np.int64], measure: NDArray[np.float64] | None
) -> BehaviourStats:
    """Statistics of events grouped by entity code and ordered by time."""
    order = np.lexsort((seconds, entities))
    entity, time = entities[order], seconds[order]
    _, starts, counts = np.unique(entity, return_index=True, return_counts=True)
    same = entity[1:] == entity[:-1]
    gaps = (time[1:] - time[:-1])[same].astype(np.float64) / SECONDS_PER_HOUR
    bursts = [_bursts(entity, time, hours * SECONDS_PER_HOUR) for hours in BURST_WINDOWS_HOURS]
    return BehaviourStats(
        entities=int(counts.size),
        events=int(entity.size),
        events_per_entity=spread(counts.astype(np.float64)),
        inter_arrival_hours=spread(gaps) if gaps.size else None,
        burst_1h=spread(bursts[0]),
        burst_24h=spread(bursts[1]),
        value_to_entity_median=None
        if measure is None
        else spread(_relative_to_median(measure[order], starts, counts)),
    )


def _bursts(entity: NDArray[np.int64], time: NDArray[np.int64], window: int) -> NDArray[np.float64]:
    """Per entity, the most events starting inside any window of this length."""
    if entity.size == 0:
        return np.zeros(0)
    start = int(np.min(time))
    span = int(np.max(time)) - start + window + 1
    key = entity.astype(np.int64) * span + (time - start)
    within = np.searchsorted(key, key + window, side="left") - np.arange(key.size)
    _, inverse = np.unique(entity, return_inverse=True)
    best = np.zeros(int(np.max(inverse)) + 1)
    np.maximum.at(best, inverse, within.astype(np.float64))
    return best


def _relative_to_median(
    values: NDArray[np.float64], starts: NDArray[np.int64], counts: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Each value divided by its entity's median (entities with a zero median are skipped)."""
    ratios = []
    for start, count in zip(starts.tolist(), counts.tolist(), strict=True):
        own = values[start : start + count]
        median = float(np.median(own))
        if median:
            ratios.append(own / median)
    return np.concatenate(ratios) if ratios else np.zeros(0)
