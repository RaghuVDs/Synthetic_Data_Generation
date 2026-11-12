"""The datetime generator: days weighted by weekday, then hours weighted by hour of day."""

from __future__ import annotations

import numpy as np

from synthgen.engine.generators.context import GenContext, Values
from synthgen.spec.generators import DatetimeGen

SECONDS_PER_DAY = 86_400
SECONDS_PER_HOUR = 3_600
WEEKDAYS = 7
HOURS = 24
EPOCH_WEEKDAY = 3
"""1970-01-01 was a Thursday; Monday is 0."""


def epoch_seconds(text: str) -> int:
    """Seconds since 1970 of an ISO date or date-time."""
    return int(np.datetime64(text.replace(" ", "T"), "s").astype(np.int64))


def datetimes(gen: DatetimeGen, n: int, rng: np.random.Generator, _ctx: GenContext) -> Values:
    """ISO dates or date-times inside [start, end] with the spec's weekday and hour mixes."""
    start, end = epoch_seconds(gen.start), epoch_seconds(gen.end)
    days = np.arange(start // SECONDS_PER_DAY, end // SECONDS_PER_DAY + 1)
    weekday = (days + EPOCH_WEEKDAY) % WEEKDAYS
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
