"""Pairwise association measures (SPEC 8.3 F). Pure functions over aligned, null-free arrays.

Categorical columns arrive as integer codes 0..k-1 with their labels; numeric ones as floats.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.stats import contingency

from synthgen.profiling.relation_models import (
    CategoricalPair,
    ConditionalShares,
    LevelShares,
    LevelSummary,
    MixedPair,
    NumericPair,
)

FloatArray = NDArray[np.float64]
Codes = NDArray[np.int64]
BINARY = 2


def _finite(value: float, default: float) -> float:
    return value if np.isfinite(value) else default


def numeric_pair(a: str, b: str, x: FloatArray, y: FloatArray) -> NumericPair:
    """Pearson, Spearman and Kendall correlations; zero when either side is constant."""
    if not np.std(x) or not np.std(y):
        return NumericPair(a=a, b=b, pearson=0.0, spearman=0.0, kendall=0.0)
    return NumericPair(
        a=a,
        b=b,
        pearson=_finite(float(stats.pearsonr(x, y).statistic), 0.0),
        spearman=_finite(float(stats.spearmanr(x, y).statistic), 0.0),
        kendall=_finite(float(stats.kendalltau(x, y).statistic), 0.0),
    )


def _entropy(counts: NDArray[np.int64]) -> float:
    return float(stats.entropy(counts.astype(np.float64))) if counts.sum() else 0.0


def theils_u(given: Codes, target: Codes) -> float:
    """Share of the target's entropy explained by knowing `given` (0 none, 1 fully)."""
    table = contingency.crosstab(given, target).count
    h_target = _entropy(table.sum(axis=0))
    if not h_target:
        return 0.0
    total = float(table.sum())
    conditional = sum(float(row.sum()) / total * _entropy(row) for row in table)
    return (h_target - conditional) / h_target


def cramers_v(x: Codes, y: Codes) -> float:
    """Cramér's V of two code arrays; zero when either has one level."""
    table = contingency.crosstab(x, y).count
    if min(table.shape) < BINARY:
        return 0.0
    return float(contingency.association(table, method="cramer"))


def categorical_pair(a: str, b: str, x: Codes, y: Codes) -> CategoricalPair:
    """Cramér's V, chi-square p-value and Theil's U in both directions."""
    table = contingency.crosstab(x, y).count
    pvalue = 1.0 if min(table.shape) < BINARY else float(stats.chi2_contingency(table).pvalue)
    return CategoricalPair(
        a=a,
        b=b,
        cramers_v=cramers_v(x, y),
        chi2_pvalue=pvalue,
        theils_u_b_given_a=theils_u(x, y),
        theils_u_a_given_b=theils_u(y, x),
    )


def correlation_ratio(values: FloatArray, codes: Codes) -> float:
    """Eta: square root of the between-level share of the total sum of squares."""
    total = float(np.sum((values - values.mean()) ** 2))
    if not total:
        return 0.0
    sizes = np.bincount(codes)
    means = np.bincount(codes, weights=values) / np.maximum(sizes, 1)
    between = float(np.sum(sizes * (means - values.mean()) ** 2))
    return float(np.sqrt(between / total))


def _pvalues(groups: list[FloatArray]) -> tuple[float, float]:
    usable = [g for g in groups if g.size > 1]
    if len(usable) < BINARY:
        return 1.0, 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # constant groups leave both statistics undefined
        anova = _finite(float(stats.f_oneway(*usable).pvalue), 1.0)
        try:
            kruskal = _finite(float(stats.kruskal(*usable).pvalue), 1.0)
        except ValueError:  # every value identical: no evidence of a difference
            kruskal = 1.0
    return anova, kruskal


def mixed_pair(  # noqa: PLR0913 - two named columns and their aligned data
    numeric: str,
    categorical: str,
    values: FloatArray,
    codes: Codes,
    *,
    labels: Sequence[str],
    level_limit: int,
) -> MixedPair:
    """Correlation ratio, ANOVA, Kruskal-Wallis, two-level point-biserial and level summaries."""
    groups = [values[codes == k] for k in range(len(labels))]
    anova, kruskal = _pvalues(groups)
    biserial = None
    if len(labels) == BINARY and np.std(values) and 0 < np.mean(codes) < 1:
        biserial = _finite(float(stats.pointbiserialr(codes, values).statistic), 0.0)
    ranked = sorted(range(len(labels)), key=lambda k: -groups[k].size)[:level_limit]
    return MixedPair(
        numeric=numeric,
        categorical=categorical,
        correlation_ratio=correlation_ratio(values, codes),
        anova_pvalue=anova,
        kruskal_pvalue=kruskal,
        point_biserial=biserial,
        levels=[_summary(labels[k], groups[k], values.size) for k in ranked if groups[k].size],
    )


def _summary(level: str, group: FloatArray, total: int) -> LevelSummary:
    return LevelSummary(
        level=level,
        share=group.size / total,
        mean=float(np.mean(group)),
        median=float(np.median(group)),
        p90=float(np.percentile(group, 90)),
    )


def conditional_shares(  # noqa: PLR0913 - two named columns, their codes and labels
    given: str,
    dependent: str,
    *,
    given_codes: Codes,
    given_labels: Sequence[str],
    dependent_codes: Codes,
    dependent_labels: Sequence[str],
    level_limit: int,
) -> ConditionalShares:
    """Top values of `dependent` within each of the most frequent levels of `given`."""
    sizes = np.bincount(given_codes, minlength=len(given_labels))
    levels: list[LevelShares] = []
    for k in np.argsort(-sizes, kind="stable")[:level_limit]:
        inside = dependent_codes[given_codes == k]
        if not inside.size:
            continue
        counts = np.bincount(inside, minlength=len(dependent_labels))
        top = np.argsort(-counts, kind="stable")[:level_limit]
        levels.append(
            LevelShares(
                level=given_labels[k],
                share=inside.size / given_codes.size,
                values=[(dependent_labels[v], counts[v] / inside.size) for v in top if counts[v]],
            )
        )
    return ConditionalShares(given=given, dependent=dependent, levels=levels)
