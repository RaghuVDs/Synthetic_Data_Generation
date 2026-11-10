"""Vectorised column generators. Pure: spec in, values out; all randomness from the given stream."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from synthgen.domain.errors import EngineError
from synthgen.spec.generators import (
    CategoricalGen,
    ConstantGen,
    DatetimeGen,
    EmpiricalGen,
    ForeignKeyGen,
    Generator,
    PatternGen,
    SequenceGen,
    UuidGen,
)

SECONDS_PER_DAY = 86_400
SECONDS_PER_HOUR = 3_600
WEEKDAYS = 7
HOURS = 24
UUID_BYTES = 16
Values = NDArray[Any]  # Any: int64, float64 or object (text, mixed categorical values, None)
KeyLookup = Callable[[str, str], Values]


@dataclass(frozen=True)
class GenContext:
    """What generators may read besides their own spec: row offset and parent keys."""

    row_offset: int
    parent_keys: KeyLookup


def generate(gen: Generator, n: int, rng: np.random.Generator, ctx: GenContext) -> Values:
    """Draw n values for one column."""
    draw: Callable[[Any, int, np.random.Generator, GenContext], Values] = _DRAWERS[gen.type]
    return draw(gen, n, rng, ctx)


def _sequence(gen: SequenceGen, n: int, _rng: np.random.Generator, ctx: GenContext) -> Values:
    return gen.start + gen.step * np.arange(ctx.row_offset, ctx.row_offset + n, dtype=np.int64)


def _foreign_key(gen: ForeignKeyGen, n: int, rng: np.random.Generator, ctx: GenContext) -> Values:
    keys = ctx.parent_keys(gen.parent_table, gen.parent_column)
    if keys.size == 0:
        raise EngineError(f"no keys in {gen.parent_table}.{gen.parent_column} to reference")
    return keys[rng.integers(0, keys.size, n)]


def _empirical(gen: EmpiricalGen, n: int, rng: np.random.Generator, _ctx: GenContext) -> Values:
    grid = np.linspace(0.0, 1.0, len(gen.quantiles))
    values = np.interp(rng.random(n), grid, np.asarray(gen.quantiles, dtype=np.float64))
    return values if gen.decimals is None else np.round(values, gen.decimals)


def _categorical(gen: CategoricalGen, n: int, rng: np.random.Generator, _ctx: GenContext) -> Values:
    picks = rng.choice(len(gen.values), size=n, p=np.asarray(gen.weights) / sum(gen.weights))
    return np.asarray(gen.values, dtype=object)[picks]


def _constant(gen: ConstantGen, n: int, _rng: np.random.Generator, _ctx: GenContext) -> Values:
    return np.full(n, gen.value, dtype=object)


def _uuids(_gen: UuidGen, n: int, rng: np.random.Generator, _ctx: GenContext) -> Values:
    raw = rng.integers(0, 256, (n, UUID_BYTES), dtype=np.uint8)
    raw[:, 6] = (raw[:, 6] & 0x0F) | 0x40  # version 4
    raw[:, 8] = (raw[:, 8] & 0x3F) | 0x80  # RFC 4122 variant
    hexed = [row.tobytes().hex() for row in raw]
    return np.array(
        [f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}" for h in hexed], dtype=object
    )


def _epoch(text: str) -> int:
    return int(np.datetime64(text.replace(" ", "T"), "s").astype(np.int64))


def _datetimes(gen: DatetimeGen, n: int, rng: np.random.Generator, _ctx: GenContext) -> Values:
    """Days weighted by weekday, then hours weighted by hour of day, inside [start, end]."""
    start, end = _epoch(gen.start), _epoch(gen.end)
    days = np.arange(start // SECONDS_PER_DAY, end // SECONDS_PER_DAY + 1)
    weekday = (days + 3) % WEEKDAYS  # 1970-01-01 was a Thursday; Monday = 0
    weights = np.asarray(gen.weekday_weights or [1.0] * WEEKDAYS)[weekday]
    if weights.sum() == 0:
        weights = np.ones(days.size)
    chosen = rng.choice(days, size=n, p=weights / weights.sum())
    if gen.granularity == "date":
        return np.datetime_as_string(chosen.astype("datetime64[D]"), unit="D").astype(object)
    hours = rng.choice(HOURS, size=n, p=np.asarray(gen.hour_weights or [1.0 / HOURS] * HOURS))
    seconds = (
        chosen * SECONDS_PER_DAY + hours * SECONDS_PER_HOUR + rng.integers(0, SECONDS_PER_HOUR, n)
    )
    text = np.datetime_as_string(np.clip(seconds, start, end).astype("datetime64[s]"), unit="s")
    return np.char.replace(text, "T", " ").astype(object)


def _patterns(gen: PatternGen, n: int, rng: np.random.Generator, _ctx: GenContext) -> Values:
    """`A` becomes a lowercase letter, `9` a digit; other characters are copied."""
    picks = rng.choice(len(gen.masks), size=n, p=np.asarray(gen.weights) / sum(gen.weights))
    out = np.empty(n, dtype=object)
    for index, mask in enumerate(gen.masks):
        rows = np.flatnonzero(picks == index)
        if rows.size == 0:
            continue
        if not mask:
            out[rows] = ""
            continue
        codes = np.tile(np.array([ord(c) for c in mask], dtype=np.uint32), (rows.size, 1))
        for symbol, low, high in (("A", ord("a"), ord("z")), ("9", ord("0"), ord("9"))):
            cols = [i for i, c in enumerate(mask) if c == symbol]
            if cols:
                codes[:, cols] = rng.integers(
                    low, high + 1, (rows.size, len(cols)), dtype=np.uint32
                )
        out[rows] = np.ascontiguousarray(codes).view(f"<U{len(mask)}").ravel().astype(object)
    return out


_DRAWERS: dict[str, Callable[[Any, int, np.random.Generator, GenContext], Values]] = {
    "sequence": _sequence,
    "uuid": _uuids,
    "foreign_key": _foreign_key,
    "empirical": _empirical,
    "categorical": _categorical,
    "datetime": _datetimes,
    "pattern": _patterns,
    "constant": _constant,
}
"""One drawing function per generator type; each takes (spec, n, stream, context)."""
