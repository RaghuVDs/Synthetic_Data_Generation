"""Deterministic column rules (SPEC 8.3 F): dependencies, ordering, arithmetic, null co-occurrence.

Pure functions over aligned sample arrays. Each search first checks a short prefix of the rows
so that the many pairs that clearly fail cost almost nothing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations, permutations

import numpy as np
from numpy.typing import NDArray

from synthgen.profiling.relation_models import ColumnRule, NullPair, RuleKind

FloatArray = NDArray[np.float64]
Codes = NDArray[np.int64]
PREFIX_ROWS = 500
MIN_GROUP_SIZE = 2
"""A determinant needs on average two rows per value, or every near-unique column determines all."""


def mode_share(determinant: Codes, dependent: Codes) -> float:
    """Share of rows whose dependent value is the most common one for their determinant value."""
    width = int(np.max(dependent)) + 1
    pairs, counts = np.unique(determinant * width + dependent, return_counts=True)
    groups = pairs // width
    starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
    return float(np.maximum.reduceat(counts, starts).sum()) / determinant.size


def functional_dependencies(codes: Mapping[str, Codes], min_share: float) -> list[ColumnRule]:
    """Column a determines b when each value of a has one value of b. Nulls count as a value."""
    distinct = {name: int(np.unique(c).size) for name, c in codes.items()}
    order = {name: i for i, name in enumerate(codes)}
    rows = next(iter(codes.values())).size if codes else 0
    found: list[ColumnRule] = []
    for a, b in permutations(codes, 2):
        n_a, n_b = distinct[a], distinct[b]
        if n_a * MIN_GROUP_SIZE > rows or n_b < MIN_GROUP_SIZE or n_b > n_a:
            continue
        if n_a == n_b and order[a] > order[b]:
            continue  # a one-to-one mapping is recorded in one direction only
        if mode_share(codes[a][:PREFIX_ROWS], codes[b][:PREFIX_ROWS]) < min_share:
            continue
        share = mode_share(codes[a], codes[b])
        if share >= min_share:
            found.append(
                ColumnRule(kind=RuleKind.DETERMINES, a=a, b=b, result=None, holds=share, rows=rows)
            )
    return found


def ordering_rules(values: Mapping[str, FloatArray], min_share: float) -> list[ColumnRule]:
    """Rule a <= b on rows where both are present, unless their ranges alone already imply it."""
    found: list[ColumnRule] = []
    for a, b in permutations(values, 2):
        x, y = values[a], values[b]
        both = ~np.isnan(x) & ~np.isnan(y)
        if not both.any():
            continue
        x, y = x[both], y[both]
        if np.max(x) <= np.min(y) or np.all(x == y):
            continue  # implied by the ranges, or the same column twice
        share = float(np.mean(x <= y))
        if share >= min_share:
            found.append(
                ColumnRule(
                    kind=RuleKind.LESS_EQUAL,
                    a=a,
                    b=b,
                    result=None,
                    holds=share,
                    rows=int(both.sum()),
                )
            )
    return found


def arithmetic_rules(
    values: Mapping[str, FloatArray], tolerance: Mapping[str, float], min_share: float
) -> list[ColumnRule]:
    """Rules result = a + b and result = a * b, within the result column's rounding tolerance."""
    found: list[ColumnRule] = []
    for a, b in combinations(values, 2):
        for kind, combined in (
            (RuleKind.SUM, values[a] + values[b]),
            (RuleKind.PRODUCT, values[a] * values[b]),
        ):
            for result in values:
                if result in (a, b):
                    continue
                rule = _arithmetic(
                    Candidate(kind, a, b, result),
                    combined,
                    values[result],
                    tolerance[result],
                    min_share,
                )
                if rule is not None:
                    found.append(rule)
    return found


@dataclass(frozen=True)
class Candidate:
    """One arithmetic rule to test: result = a (kind) b."""

    kind: RuleKind
    a: str
    b: str
    result: str


def _arithmetic(
    candidate: Candidate, combined: FloatArray, target: FloatArray, atol: float, min_share: float
) -> ColumnRule | None:
    present = ~np.isnan(combined) & ~np.isnan(target)
    head = present[:PREFIX_ROWS]
    if not head.any():
        return None
    close_head = np.isclose(
        combined[:PREFIX_ROWS][head], target[:PREFIX_ROWS][head], rtol=0, atol=atol
    )
    if float(np.mean(close_head)) < min_share:
        return None
    share = float(np.mean(np.isclose(combined[present], target[present], rtol=0, atol=atol)))
    if share < min_share:
        return None
    return ColumnRule(
        kind=candidate.kind,
        a=candidate.a,
        b=candidate.b,
        result=candidate.result,
        holds=share,
        rows=int(present.sum()),
    )


def null_pairs(nulls: Mapping[str, NDArray[np.bool_]], min_jaccard: float) -> list[NullPair]:
    """Pairs of sometimes-null columns whose null masks overlap by at least `min_jaccard`."""
    names = [n for n, mask in nulls.items() if 0 < mask.sum() < mask.size]
    if len(names) < MIN_GROUP_SIZE:
        return []
    matrix = np.column_stack([nulls[n] for n in names]).astype(np.int64)
    inter = matrix.T @ matrix
    counts = np.diag(inter)
    rows = matrix.shape[0]
    found: list[NullPair] = []
    for i, j in combinations(range(len(names)), 2):
        union = counts[i] + counts[j] - inter[i, j]
        jaccard = inter[i, j] / union if union else 0.0
        if jaccard >= min_jaccard:
            found.append(
                NullPair(
                    a=names[i],
                    b=names[j],
                    jaccard=float(jaccard),
                    both_null_rate=float(inter[i, j] / rows),
                )
            )
    return found
