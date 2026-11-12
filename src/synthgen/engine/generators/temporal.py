"""The datetime generator: days weighted by weekday, then hours weighted by hour of day.

With `after`, a value is an earlier date of the row (or its parent) plus a lag drawn from the
profiled lag quantiles, so child dates keep their parent's timeline instead of being repaired.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from synthgen.engine.generators.context import GenContext, Values
from synthgen.spec.evaluator import to_internal, to_output
from synthgen.spec.expressions import ExprType
from synthgen.spec.generators import DateLag, DatetimeGen

SECONDS_PER_DAY = 86_400
SECONDS_PER_HOUR = 3_600
WEEKDAYS = 7
HOURS = 24
EPOCH_WEEKDAY = 3
"""1970-01-01 was a Thursday; Monday is 0."""
Floats = NDArray[np.float64]


def epoch_seconds(text: str) -> int:
    """Seconds since 1970 of an ISO date or date-time."""
    return int(np.datetime64(text.replace(" ", "T"), "s").astype(np.int64))


def datetimes(gen: DatetimeGen, n: int, rng: np.random.Generator, ctx: GenContext) -> Values:
    """ISO dates or date-times inside [start, end] with the spec's weekday and hour mixes."""
    if gen.after is not None:
        return _after(gen, gen.after, n, rng, ctx)
    start, end = epoch_seconds(gen.start), epoch_seconds(gen.end)
    days = np.arange(start // SECONDS_PER_DAY, end // SECONDS_PER_DAY + 1)
    weekday = (days + EPOCH_WEEKDAY) % WEEKDAYS
    weights = np.asarray(gen.weekday_weights or [1.0] * WEEKDAYS)[weekday]
    if weights.sum() == 0:
        weights = np.ones(days.size)
    chosen = rng.choice(days, size=n, p=weights / weights.sum())
    if gen.granularity == "date":
        return np.datetime_as_string(chosen.astype("datetime64[D]"), unit="D").astype(object)
    hours = rng.choice(HOURS, size=n, p=_hour_weights(gen))
    seconds = (
        chosen * SECONDS_PER_DAY + hours * SECONDS_PER_HOUR + rng.integers(0, SECONDS_PER_HOUR, n)
    )
    text = np.datetime_as_string(np.clip(seconds, start, end).astype("datetime64[s]"), unit="s")
    return np.char.replace(text, "T", " ").astype(object)


def _hour_weights(gen: DatetimeGen) -> NDArray[np.float64]:
    return np.asarray(gen.hour_weights or [1.0 / HOURS] * HOURS, dtype=np.float64)


def _after(
    gen: DatetimeGen, lag: DateLag, n: int, rng: np.random.Generator, ctx: GenContext
) -> Values:
    """The earlier date plus a lag from the quantiles, truncated so it stays at or before `end`.

    Truncating the lag distribution (not clipping the date) keeps values off a pile at `end`.
    A date-time takes its day from the lag and its hour from the hour mix when that keeps it
    after the earlier value. Rows whose earlier date is null are drawn like any other row.
    """
    kind, raw = ctx.frame[lag.column]
    base = to_internal(raw, kind)
    end = float(epoch_seconds(gen.end))
    if gen.granularity == "date":
        end += SECONDS_PER_DAY - 1
    quantiles = np.asarray(lag.lag_quantiles, dtype=np.float64)
    levels = np.linspace(0.0, 1.0, quantiles.size)
    room = np.interp(np.nan_to_num(end - base, nan=np.inf), quantiles, levels, left=0.0)
    drawn = base + np.interp(rng.random(n) * room, levels, quantiles)
    if gen.granularity == "date":
        day = np.floor(drawn / SECONDS_PER_DAY) * SECONDS_PER_DAY
        value = np.where(day < base, day + SECONDS_PER_DAY, day)
    else:
        value = _retimed(gen, drawn, (base, end), rng)
    missing = np.isnan(base)
    out = to_output(value, ExprType.DATE if gen.granularity == "date" else ExprType.DATETIME)
    if missing.any():
        plain = gen.model_copy(update={"after": None})
        out[missing] = datetimes(
            plain, int(missing.sum()), rng, ctx.subset(np.flatnonzero(missing))
        )
    return out


def _retimed(
    gen: DatetimeGen, drawn: Floats, bounds: tuple[Floats, float], rng: np.random.Generator
) -> Floats:
    """Keep the drawn day; take the time of day from the hour mix where it stays in bounds."""
    base, end = bounds
    n = drawn.size
    hours = rng.choice(HOURS, size=n, p=_hour_weights(gen))
    seconds = hours * SECONDS_PER_HOUR + rng.integers(0, SECONDS_PER_HOUR, n)
    retimed = np.floor(drawn / SECONDS_PER_DAY) * SECONDS_PER_DAY + seconds
    keep = (retimed >= base) & (retimed <= end)
    return np.asarray(np.where(keep, retimed, np.ceil(drawn)), dtype=np.float64)
