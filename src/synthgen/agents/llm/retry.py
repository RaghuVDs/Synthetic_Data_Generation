"""Backoff and a client-side rate limiter for LLM calls (SPEC 4.5)."""

from __future__ import annotations

import random
from collections.abc import Callable

from synthgen.config import LlmSettings

SECONDS_PER_MINUTE = 60.0
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


def backoff_seconds(
    attempt: int, retry_after: str | None, settings: LlmSettings, rng: random.Random
) -> float:
    """Wait before retry `attempt` (0-based): numeric Retry-After, else jittered exponential."""
    if retry_after is not None and retry_after.strip().replace(".", "", 1).isdigit():
        return min(float(retry_after), settings.backoff_max_seconds)
    ceiling = min(settings.backoff_max_seconds, settings.backoff_base_seconds * 2.0**attempt)
    return ceiling * (1.0 - settings.backoff_jitter * rng.random())


class RateLimiter:
    """Spaces requests evenly so no more than `requests_per_minute` start in any minute."""

    def __init__(
        self,
        requests_per_minute: int,
        clock: Callable[[], float],
        sleep: Callable[[float], None],
    ) -> None:
        """Clock and sleep are injected so tests run instantly."""
        self._interval = SECONDS_PER_MINUTE / requests_per_minute
        self._clock = clock
        self._sleep = sleep
        self._next_start: float | None = None

    def wait(self) -> None:
        """Block until the next request may start."""
        now = self._clock()
        if self._next_start is not None and now < self._next_start:
            self._sleep(self._next_start - now)
            now = self._next_start
        self._next_start = now + self._interval
