"""Behaviour per entity over time (SPEC 8.3 H): velocity, gaps, bursts, lifetimes, states.

Pure functions over event arrays. Entities are integer codes; times are epoch seconds. Most
functions expect events sorted by entity, then time (see `sort_events`).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from synthgen.profiling.relation_models import BehaviourStats, TransitionMatrix
from synthgen.profiling.shape_models import Spread

SECONDS_PER_HOUR = 3_600
SECONDS_PER_DAY = 86_400
MIN_AUTOCORRELATION_PAIRS = 3
BURST_WINDOWS_HOURS = (1, 24)
PERCENTILES = (50.0, 90.0, 99.0)


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
    """Each value divided by its entity's median; missing values and zero medians are skipped."""
    ratios = []
    for start, count in zip(starts.tolist(), counts.tolist(), strict=True):
        own = values[start : start + count]
        own = own[~np.isnan(own)]
        median = float(np.median(own)) if own.size else 0.0
        if median:
            ratios.append(own / median)
    return np.concatenate(ratios) if ratios else np.zeros(0)


def sort_events(entities: NDArray[np.int64], seconds: NDArray[np.int64]) -> NDArray[np.int64]:
    """Order that sorts events by entity, then time."""
    return np.lexsort((seconds, entities))


def per_period(entity: NDArray[np.int64], time: NDArray[np.int64], period: int) -> Spread:
    """Events per entity within each period (day or week) in which the entity was active."""
    if entity.size == 0:
        return spread(np.zeros(0))
    _, counts = np.unique(np.stack([entity, time // period]), axis=1, return_counts=True)
    return spread(counts.astype(np.float64))


def lifetimes_days(entity: NDArray[np.int64], time: NDArray[np.int64]) -> Spread:
    """Days from each entity's first to its last event."""
    if entity.size == 0:
        return spread(np.zeros(0))
    _, inverse = np.unique(entity, return_inverse=True)
    first = np.full(int(np.max(inverse)) + 1, np.iinfo(np.int64).max)
    last = np.full(first.size, np.iinfo(np.int64).min)
    np.minimum.at(first, inverse, time)
    np.maximum.at(last, inverse, time)
    return spread((last - first).astype(np.float64) / SECONDS_PER_DAY)


def lag1_autocorrelation(entity: NDArray[np.int64], values: NDArray[np.float64]) -> float | None:
    """Correlation of consecutive present values of one entity; None when it is undefined."""
    same = (entity[1:] == entity[:-1]) & ~np.isnan(values[1:]) & ~np.isnan(values[:-1])
    x, y = values[:-1][same], values[1:][same]
    if x.size < MIN_AUTOCORRELATION_PAIRS or not np.std(x) or not np.std(y):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def distinct_per_entity(entity: NDArray[np.int64], other: NDArray[np.int64]) -> Spread:
    """How many distinct values of another key each entity has (nulls, coded -1, excluded)."""
    keep = other >= 0
    if not keep.any():
        return spread(np.zeros(0))
    pairs = np.unique(np.stack([entity[keep], other[keep]]), axis=1)
    _, counts = np.unique(pairs[0], return_counts=True)
    return spread(counts.astype(np.float64))


def transitions(
    column: str, entity: NDArray[np.int64], codes: NDArray[np.int64], states: list[str]
) -> TransitionMatrix:
    """Row-normalised counts of state changes between consecutive events of an entity."""
    same = (entity[1:] == entity[:-1]) & (codes[1:] >= 0) & (codes[:-1] >= 0)
    counts = np.zeros((len(states), len(states)))
    np.add.at(counts, (codes[:-1][same], codes[1:][same]), 1.0)
    totals = counts.sum(axis=1, keepdims=True)
    probabilities = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    return TransitionMatrix(
        column=column,
        states=states,
        probabilities=[[float(p) for p in row] for row in probabilities],
    )
