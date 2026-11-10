"""Core categorical and text-shape statistics (SPEC 8.3 D)."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from synthgen.config import ProfilingSettings
from synthgen.profiling.inference import mask_shares
from synthgen.profiling.models import (
    CategoricalProfile,
    Measured,
    StatSource,
    TextShape,
    ValueCount,
)

MASK_LIMIT = 20


def categorical_profile(
    counts: Sequence[tuple[str | int | float, int]],
    total: int,
    source: StatSource,
    settings: ProfilingSettings,
) -> CategoricalProfile:
    """Frequencies (all values, or the top ones plus `other`), entropy and rare values.

    `counts` are exact (SQL) or sampled value counts; `total` is the non-null row count they cover.
    """
    ordered = sorted(counts, key=lambda vc: (-vc[1], str(vc[0])))
    keep = (
        ordered
        if len(ordered) <= settings.max_full_frequencies
        else ordered[: settings.top_frequencies]
    )
    kept_total = sum(n for _, n in keep)
    shares = [n / total for _, n in ordered] if total else []
    entropy = -sum(p * math.log(p) for p in shares if p > 0)
    normalized = entropy / math.log(len(shares)) if len(shares) > 1 else 0.0
    rare = sum(1 for p in shares if p < settings.rare_value_share)
    return CategoricalProfile(
        frequencies=Measured[list[ValueCount]](
            value=[ValueCount(value=v, count=n) for v, n in keep], source=source
        ),
        other_count=Measured[int](value=total - kept_total, source=source),
        entropy=Measured[float](value=entropy, source=StatSource.DERIVED),
        normalized_entropy=Measured[float](value=normalized, source=StatSource.DERIVED),
        rare_count=Measured[int](value=rare, source=StatSource.DERIVED),
    )


def sample_counts(values: Sequence[str | int | float]) -> list[tuple[str | int | float, int]]:
    """Value counts of a sample."""
    return list(Counter(values).items())


def text_shape(texts: Sequence[str]) -> TextShape:
    """Lengths and masks of non-null values; holds no value, so it is safe for sensitive columns."""
    lengths = [len(t) for t in texts] or [0]
    sample = StatSource.SAMPLE
    return TextShape(
        length_min=Measured[int](value=min(lengths), source=sample),
        length_max=Measured[int](value=max(lengths), source=sample),
        length_mean=Measured[float](value=sum(lengths) / len(lengths), source=sample),
        masks=Measured(value=mask_shares(texts, MASK_LIMIT), source=sample),
    )
