from __future__ import annotations

import math

import numpy as np
import pytest

from synthgen.config import ProfilingSettings
from synthgen.profiling.categorical import categorical_profile, sample_counts
from synthgen.profiling.models import StatSource
from synthgen.profiling.numeric import SqlMoments, decimal_places, numeric_profile
from synthgen.profiling.temporal import temporal_profile
from synthgen.profiling.text import text_shape


def _moments(values: np.ndarray) -> SqlMoments:
    return SqlMoments(
        count=values.size,
        minimum=float(values.min()),
        maximum=float(values.max()),
        total=float(values.sum()),
        sum_squares=float((values**2).sum()),
    )


def test_numeric_profile_matches_numpy(profiling_settings: ProfilingSettings) -> None:
    values = np.random.default_rng(1).lognormal(3.0, 0.7, 5_000).round(2)
    profile = numeric_profile(
        _moments(values), values, profiling_settings, distinct=values.size, full_fit=True
    )
    assert profile.mean.value == pytest.approx(values.mean())
    assert profile.std.value == pytest.approx(values.std(ddof=1))
    assert profile.minimum.source is StatSource.SQL
    assert profile.quantiles.source is StatSource.SAMPLE
    quantiles = profile.quantiles.value
    assert len(quantiles) == profiling_settings.quantile_points
    assert quantiles == sorted(quantiles)
    assert sum(profile.histogram.value.counts) == values.size
    assert len(profile.histogram.value.edges) == profiling_settings.histogram_bins + 1
    assert profile.decimal_places.value == 2
    assert not profile.is_integer.value
    assert dict(profile.percentiles.value)[50.0] == pytest.approx(np.median(values))


def test_numeric_rates_and_integers(profiling_settings: ProfilingSettings) -> None:
    values = np.array([-2.0, 0.0, 0.0, 3.0, 5.0])
    profile = numeric_profile(
        _moments(values), values, profiling_settings, distinct=values.size, full_fit=True
    )
    assert profile.zero_rate.value == pytest.approx(0.4)
    assert profile.negative_rate.value == pytest.approx(0.2)
    assert profile.is_integer.value
    assert profile.decimal_places.value == 0


def test_decimal_places_caps_at_six() -> None:
    assert decimal_places(np.array([math.pi])) == 6
    assert decimal_places(np.array([1.5, 2.25])) == 2


def test_categorical_full_frequencies(profiling_settings: ProfilingSettings) -> None:
    profile = categorical_profile([("a", 50), ("b", 50)], 100, StatSource.SQL, profiling_settings)
    assert [v.value for v in profile.frequencies.value] == ["a", "b"]
    assert profile.other_count.value == 0
    assert profile.entropy.value == pytest.approx(math.log(2))
    assert profile.normalized_entropy.value == pytest.approx(1.0)
    assert profile.rare_count.value == 0


def test_categorical_top_values_and_other(profiling_settings: ProfilingSettings) -> None:
    counts = [(f"v{i}", 1) for i in range(profiling_settings.max_full_frequencies + 1)]
    profile = categorical_profile(counts, len(counts), StatSource.SAMPLE, profiling_settings)
    assert len(profile.frequencies.value) == profiling_settings.top_frequencies
    assert profile.other_count.value == len(counts) - profiling_settings.top_frequencies
    assert profile.rare_count.value == len(counts)


def test_sample_counts() -> None:
    assert sorted(sample_counts(["a", "b", "a"])) == [("a", 2), ("b", 1)]


def test_text_shape() -> None:
    shape = text_shape(["ab", "abcd"])
    assert (shape.length_min.value, shape.length_max.value) == (2, 4)
    assert shape.length_mean.value == 3
    assert {m.mask for m in shape.masks.value} == {"AA", "AAAA"}
    assert text_shape([]).length_max.value == 0


def test_temporal_profile() -> None:
    texts = ["2024-03-04 09:15:00", "2024-03-05 09:45:00", "2024-12-08 23:00:00"]
    profile = temporal_profile("2024-03-04 09:15:00", "2024-12-08 23:00:00", texts)
    assert profile.weekday_counts.value == [1, 1, 0, 0, 0, 0, 1]  # Mon, Tue, Sun
    assert profile.month_counts.value[2] == 2
    assert profile.month_counts.value[11] == 1
    assert profile.hour_counts.value[9] == 2
    assert profile.hour_counts.value[23] == 1
    assert profile.span_days.value == pytest.approx(279.57, abs=0.01)
    assert profile.minimum.source is StatSource.SQL
