"""Core categorical and text-shape statistics (SPEC 8.3 D)."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np

from synthgen.config import ProfilingSettings
from synthgen.profiling.models import CategoricalProfile, Measured, StatSource, ValueCount

MIN_ZIPF_VALUES = 3


def categorical_profile(
    counts: Sequence[tuple[str | int | float, int]],
    total: int,
    source: StatSource,
    settings: ProfilingSettings,
    threshold: int = 0,
) -> CategoricalProfile:
    """Frequencies (all values, or the top ones plus `other`), entropy and rare values.

    `counts` are exact (SQL) or sampled value counts; `total` is the non-null row count they cover.
    Values counted fewer than `threshold` times fold into `other`; entropy still uses them all.
    """
    ordered = sorted(counts, key=lambda vc: (-vc[1], str(vc[0])))
    top = (
        ordered
        if len(ordered) <= settings.max_full_frequencies
        else ordered[: settings.top_frequencies]
    )
    keep = [(v, n) for v, n in top if n >= threshold]
    kept_total = sum(n for _, n in keep)
    shares = [n / total for _, n in ordered] if total else []
    entropy = -sum(p * math.log(p) for p in shares if p > 0)
    normalized = entropy / math.log(len(shares)) if len(shares) > 1 else 0.0
    rare = sum(1 for p in shares if p < settings.rare_value_share)
    exponent = zipf_exponent([n for _, n in ordered])
    return CategoricalProfile(
        frequencies=Measured[list[ValueCount]](
            value=[ValueCount(value=v, count=n) for v, n in keep], source=source
        ),
        other_count=Measured[int](value=total - kept_total, source=source),
        entropy=Measured[float](value=entropy, source=StatSource.DERIVED),
        normalized_entropy=Measured[float](value=normalized, source=StatSource.DERIVED),
        rare_count=Measured[int](value=rare, source=StatSource.DERIVED),
        zipf_exponent=None
        if exponent is None
        else Measured[float](value=exponent, source=StatSource.DERIVED),
    )


def zipf_exponent(counts_desc: Sequence[int]) -> float | None:
    """Slope s of log(frequency) = c - s log(rank), from counts sorted descending."""
    positive = [c for c in counts_desc if c > 0]
    if len(positive) < MIN_ZIPF_VALUES:
        return None
    ranks = np.log(np.arange(1, len(positive) + 1))
    slope = float(np.polyfit(ranks, np.log(np.asarray(positive, dtype=np.float64)), 1)[0])
    return -slope


def sample_counts(values: Sequence[str | int | float]) -> list[tuple[str | int | float, int]]:
    """Value counts of a sample."""
    return list(Counter(values).items())
