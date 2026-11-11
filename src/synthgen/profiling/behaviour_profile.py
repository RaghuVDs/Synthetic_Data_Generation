"""Behaviour profiles of an event table (SPEC 8.3 H): one per foreign key x main timestamp.

A row sample would stretch the gaps between an entity's events, so this module samples
entities instead and reads their full histories through whitelisted grouped SQL.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from synthgen.config import ProfilingSettings
from synthgen.domain.metadata import Relationship, Sensitivity
from synthgen.domain.ports import Connector
from synthgen.domain.query import AggExpr, AggFunc, Condition, Filter, Operator
from synthgen.profiling.behaviour import (
    SECONDS_PER_DAY,
    SECONDS_PER_HOUR,
    behaviour_stats,
    distinct_per_entity,
    lag1_autocorrelation,
    lifetimes_days,
    per_period,
    sort_events,
    transitions,
)
from synthgen.profiling.fitting import fit_distributions
from synthgen.profiling.models import ColumnProfile, InferredType
from synthgen.profiling.relation_models import BehaviourProfile, Counterparties
from synthgen.profiling.temporal import epoch_seconds

GAP_FAMILIES = frozenset({"expon", "gamma", "lognorm"})
IDENTIFIER_TYPES = (InferredType.INTEGER, InferredType.TEXT)
DAYS_PER_WEEK = 7
OTHER = "other"


@dataclass(frozen=True)
class EventColumns:
    """The columns of an event table that behaviour statistics read."""

    time: str
    measure: str | None
    states: list[ColumnProfile]
    keys: list[str]


def _usable(column: ColumnProfile, keys: set[str]) -> bool:
    return column.name not in keys and column.sensitivity is not Sensitivity.SENSITIVE


def event_columns(
    columns: Sequence[ColumnProfile], keys: list[str], settings: ProfilingSettings
) -> EventColumns | None:
    """Main timestamp (first date-time, else first date), main measure and state columns.

    None when the table has no foreign key or no usable date column: it is not an event table.
    """
    identifiers = {
        c.name for c in columns if c.is_unique.value and c.inferred_type in IDENTIFIER_TYPES
    }
    others = set(keys) | identifiers
    usable = [c for c in columns if _usable(c, others)]
    times = [c for c in usable if c.inferred_type is InferredType.DATETIME] or [
        c for c in usable if c.inferred_type is InferredType.DATE
    ]
    if not keys or not times:
        return None
    measures = [c for c in usable if c.inferred_type is InferredType.FLOAT] or [
        c for c in usable if c.inferred_type is InferredType.INTEGER
    ]
    states = [
        c
        for c in usable
        if c.categorical is not None
        and 1 < c.distinct_count.value <= settings.max_conditional_levels
    ]
    return EventColumns(
        time=times[0].name,
        measure=measures[0].name if measures else None,
        states=states[: settings.max_state_columns],
        keys=keys,
    )


@dataclass(frozen=True)
class EventSource:
    """Where an event table's histories are read from, and with which settings and seed."""

    connector: Connector
    table: str
    settings: ProfilingSettings
    seed: int


def _histories(source: EventSource, entity: str, columns: list[str]) -> pd.DataFrame:
    """Every event of a seeded sample of entities, ordered by the grouped columns."""
    settings = source.settings
    n = AggExpr(func=AggFunc.COUNT, column=None, alias="n")
    grouped = source.connector.aggregate(source.table, [n], group_by=[entity])
    keys = [r[entity] for r in grouped if r[entity] is not None]
    rng = np.random.default_rng(source.seed)
    size = min(len(keys), settings.behaviour_entities)
    picked = [keys[i] for i in sorted(rng.choice(len(keys), size=size, replace=False))]
    group = [entity, *dict.fromkeys(c for c in columns if c != entity)]
    rows: list[dict[str, Any]] = []  # Any: SQLite cells
    for start in range(0, len(picked), settings.behaviour_key_batch):
        batch = tuple(picked[start : start + settings.behaviour_key_batch])
        where = Filter(conditions=(Condition(column=entity, op=Operator.IN, value=batch),))
        rows += source.connector.aggregate(source.table, [n], group_by=group, where=where)
    frame = pd.DataFrame(rows, columns=[*group, "n"])
    return frame.loc[frame.index.repeat(frame["n"].astype(int))].reset_index(drop=True)


def _codes(
    values: pd.Series, column: ColumnProfile, settings: ProfilingSettings
) -> tuple[NDArray[np.int64], list[str]]:
    """State codes (nulls -1); levels of internal columns seen on fewer rows fold into `other`."""
    texts = values.where(values.isna(), values.astype(str))
    if column.sensitivity is Sensitivity.INTERNAL and column.categorical is not None:
        counts = {v.value: v.count for v in column.categorical.frequencies.value}
        threshold = settings.internal_min_count
        rare = {v for v in texts.dropna().unique() if counts.get(v, 0) < threshold}
        texts = texts.where(~texts.isin(rare), OTHER)
    codes, states = pd.factorize(texts, sort=True)
    return codes.astype(np.int64), [str(s) for s in states]


@dataclass(frozen=True)
class Events:
    """One entity's sampled histories, sorted by entity code, then time."""

    frame: pd.DataFrame
    codes: NDArray[np.int64]
    time: NDArray[np.int64]
    measure: NDArray[np.float64] | None


def _events(source: EventSource, entity: str, columns: EventColumns) -> Events:
    wanted = [columns.time, *([columns.measure] if columns.measure else [])]
    wanted += [s.name for s in columns.states]
    frame = _histories(source, entity, [*wanted, *columns.keys])
    seconds = epoch_seconds(frame[columns.time].tolist())
    present = ~np.isnan(seconds)
    frame, time = frame[present], seconds[present].astype(np.int64)
    _, codes = np.unique(frame[entity].astype(str).to_numpy(), return_inverse=True)
    order = sort_events(codes.astype(np.int64), time)
    frame = frame.iloc[order]
    measure = None
    if columns.measure is not None:
        measure = pd.to_numeric(frame[columns.measure], errors="coerce").to_numpy(np.float64)
    return Events(frame, codes[order].astype(np.int64), time[order], measure)


def behaviour_profile(
    source: EventSource, entity: str, columns: EventColumns, profiles: Sequence[ColumnProfile]
) -> BehaviourProfile:
    """Group H for one entity key, from the full histories of sampled entities."""
    settings = source.settings
    events = _events(source, entity, columns)
    codes, time = events.codes, events.time
    same = codes[1:] == codes[:-1]
    gaps = (time[1:] - time[:-1])[same].astype(np.float64) / SECONDS_PER_HOUR
    positive = gaps[gaps > 0][: settings.fit_sample_size]
    by_name = {c.name: c for c in profiles}
    return BehaviourProfile(
        entity_column=entity,
        time_column=columns.time,
        measure_column=columns.measure,
        stats=behaviour_stats(codes, time, events.measure),
        per_active_day=per_period(codes, time, SECONDS_PER_DAY),
        per_active_week=per_period(codes, time, DAYS_PER_WEEK * SECONDS_PER_DAY),
        inter_arrival_fits=fit_distributions(positive, full=True, families=GAP_FAMILIES),
        gap_autocorrelation=lag1_autocorrelation(codes[1:][same], gaps) if gaps.size else None,
        value_autocorrelation=None
        if events.measure is None
        else lag1_autocorrelation(codes, events.measure),
        lifetime_days=lifetimes_days(codes, time),
        counterparties=[
            Counterparties(
                column=key,
                per_entity=distinct_per_entity(
                    codes, pd.factorize(events.frame[key])[0].astype(np.int64)
                ),
            )
            for key in columns.keys
            if key != entity
        ],
        transitions=[
            transitions(s.name, codes, *_codes(events.frame[s.name], by_name[s.name], settings))
            for s in columns.states
        ],
    )


def behaviour_profiles(
    source: EventSource,
    profiles: Sequence[ColumnProfile],
    relationships: Sequence[Relationship],
) -> list[BehaviourProfile]:
    """One behaviour profile per foreign key of an event table; none for other tables."""
    keys = list(
        dict.fromkeys(
            r.child_column
            for r in relationships
            if r.child_table == source.table and not r.is_self_reference
        )
    )
    columns = event_columns(profiles, keys, source.settings)
    if columns is None:
        return []
    return [behaviour_profile(source, key, columns, profiles) for key in keys]
