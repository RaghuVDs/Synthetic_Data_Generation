"""Numeric shape (SPEC 8.3 C): spread, outliers, modes, steps, first digits and normality.

Pure functions over sampled values. KDE, fits and normality tests use the seeded fit subsample.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import signal, stats

from synthgen.config import ProfilingSettings
from synthgen.profiling.shape_models import NormalityTests, NumericShape

FloatArray = NDArray[np.float64]
MIN_FOR_TESTS = 20
MIN_FOR_BENFORD = 100
BENFORD_MIN_DECADES = 2.0
PEAK_PROMINENCE = 0.05
MODIFIED_Z_SCALE = 0.6745
BENFORD_EXPECTED = np.log10(1 + 1 / np.arange(1, 10))


def numeric_shape(
    sample: FloatArray,
    subsample: FloatArray,
    step: float | None,
    settings: ProfilingSettings,
    *,
    full: bool,
) -> NumericShape:
    """Shape statistics of a non-empty sample; `step` comes from exact SQL facts.

    KDE modes only run with `full`; wide tables skip them to keep profiling time linear.
    """
    std = float(np.std(sample))
    mean = float(np.mean(sample))
    q1, median, q3 = (float(v) for v in np.percentile(sample, (25, 50, 75)))
    iqr = q3 - q1
    mad = float(stats.median_abs_deviation(sample))
    fence = settings.iqr_fence * iqr
    return NumericShape(
        range=float(np.max(sample) - np.min(sample)),
        iqr=iqr,
        mad=mad,
        variation=std / abs(mean) if mean else None,
        skew=float(stats.skew(sample)) if std else 0.0,
        kurtosis=float(stats.kurtosis(sample)) if std else 0.0,
        round_rates=[(b, _multiple_rate(sample, b)) for b in settings.round_bases],
        iqr_outlier_rate=float(np.mean((sample < q1 - fence) | (sample > q3 + fence))),
        zscore_outlier_rate=float(np.mean(np.abs(sample - mean) > settings.zscore_limit * std))
        if std
        else 0.0,
        modified_z_outlier_rate=float(
            np.mean(MODIFIED_Z_SCALE * np.abs(sample - median) / mad > settings.modified_z_limit)
        )
        if mad
        else 0.0,
        modes=modes(subsample, settings.kde_points) if full else [],
        constant_step=step,
        benford_pvalue=benford_pvalue(sample),
    )


def _multiple_rate(sample: FloatArray, base: float) -> float:
    remainder = np.mod(sample, base)
    return float(np.mean(np.isclose(remainder, 0.0) | np.isclose(remainder, base)))


def constant_step(count: int, distinct: int, minimum: float, maximum: float) -> float | None:
    """The step when every value is distinct and evenly spaced from min to max (keys, sequences)."""
    if count < 2 or distinct != count:  # noqa: PLR2004 - a step needs two values
        return None
    step = (maximum - minimum) / (count - 1)
    return step if step > 0 and float(step).is_integer() else None


def modes(subsample: FloatArray, points: int) -> list[float]:
    """Peaks of a Gaussian KDE; empty for constant or tiny samples."""
    if subsample.size < MIN_FOR_TESTS or np.unique(subsample).size < 2 or not np.std(subsample):  # noqa: PLR2004
        return []
    kde: stats.gaussian_kde = stats.gaussian_kde(subsample)
    grid = np.linspace(float(np.min(subsample)), float(np.max(subsample)), points)
    density = kde(grid)
    peaks, _ = signal.find_peaks(density, prominence=PEAK_PROMINENCE * float(np.max(density)))
    return [float(grid[i]) for i in peaks]


def benford_pvalue(sample: FloatArray) -> float | None:
    """Chi-square p-value of first digits against Benford's law, when the data spans decades."""
    positive = sample[sample > 0]
    if positive.size < MIN_FOR_BENFORD:
        return None
    if math.log10(float(np.max(positive)) / float(np.min(positive))) < BENFORD_MIN_DECADES:
        return None
    digits = (positive / 10 ** np.floor(np.log10(positive))).astype(int)
    observed = np.bincount(np.clip(digits, 1, 9), minlength=10)[1:]
    return float(stats.chisquare(observed, BENFORD_EXPECTED * observed.sum()).pvalue)


def normality(subsample: FloatArray, settings: ProfilingSettings) -> NormalityTests | None:
    """Shapiro, D'Agostino, Anderson-Darling and Jarque-Bera; None for constant or tiny samples."""
    if subsample.size < MIN_FOR_TESTS or not np.std(subsample):
        return None
    # scipy-stubs 1.18.1.1 lacks `method` (added in SciPy 1.17; required from 1.19)
    anderson = stats.anderson(subsample, dist="norm", method="interpolate")  # type: ignore[call-overload]
    return NormalityTests(
        shapiro_pvalue=float(stats.shapiro(subsample[: settings.shapiro_max_rows]).pvalue),
        normaltest_pvalue=float(stats.normaltest(subsample).pvalue),
        anderson_statistic=float(anderson.statistic),
        anderson_pvalue=float(anderson.pvalue),
        jarque_bera_pvalue=float(stats.jarque_bera(subsample).pvalue),
    )
