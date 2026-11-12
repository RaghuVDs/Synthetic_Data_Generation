"""Privacy checks (SPEC 8.6, PLAN Q7, A14). A hard gate: output never leaves with a failure.

- Exact match: synthetic values of identifier-like sensitive columns (cardinality ratio at least
  `privacy_identifier_min_ratio`, not dates, not names or places drawn from small vocabularies)
  never occur in the source; other sensitive columns report their matches as info.
- Row hashes: synthetic rows (non-key, non-label columns) never equal a row of the profiling
  sample; tables with fewer value combinations than the sample are skipped as info.
- Suppressed values: an internal column never outputs a source value held by fewer than
  `internal_min_count` rows (a value the profile folded away).
- Text pools: model-written text never equals a source value of a sensitive text column.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from synthgen.domain.metadata import Sensitivity
from synthgen.domain.ports import Connector
from synthgen.domain.report import CheckFamily, CheckResult, CheckStatus
from synthgen.engine.schema import SCENARIO_COLUMN
from synthgen.profiling.models import TableProfile
from synthgen.spec.generators import SemanticGen, SemanticKind, TextPoolGen
from synthgen.spec.models import ColumnType, GenerationSpec, TableSpec
from synthgen.validation.integrity_rules import traced
from synthgen.validation.lookup import source_counts
from synthgen.validation.output import OutputData, ValidationInputs

_TIMES = (ColumnType.DATE, ColumnType.DATETIME)
LOW_ENTROPY = frozenset(
    {
        SemanticKind.FIRST_NAME,
        SemanticKind.LAST_NAME,
        SemanticKind.FULL_NAME,
        SemanticKind.CITY,
        SemanticKind.POSTCODE,
        SemanticKind.COMPANY,
        SemanticKind.DATE_OF_BIRTH,
    }
)
"""Fake values of these kinds come from small vocabularies and match real ones by chance (Q7)."""


def _result(  # noqa: PLR0913 - the fields of one report row
    where: tuple[str, str | None],
    check: str,
    matches: float | None,
    *,
    hard: bool,
    detail: str | None = None,
    scenario: str | None = None,
) -> CheckResult:
    if matches is None or not hard:
        status = CheckStatus.INFO
    else:
        status = CheckStatus.PASS if matches == 0 else CheckStatus.FAIL
    return CheckResult(
        family=CheckFamily.PRIVACY,
        check=check,
        table=where[0],
        column=where[1],
        metric="matches",
        value=matches,
        threshold=0.0,
        status=status,
        scenario=scenario,
        detail=detail,
    )


def check_privacy(inputs: ValidationInputs, data: OutputData) -> list[CheckResult]:
    """Every privacy check; without an open source they cannot run and report info."""
    source = inputs.source
    if source is None:
        return [_result(("*", None), "source_lookup", None, hard=False, detail="source not open")]
    results: list[CheckResult] = []
    for table in inputs.spec.tables:
        profile = inputs.profiles.get(table.name)
        if profile is None or data.entry(table.name) is None:
            continue
        results += _exact_matches(inputs, data, source, table, profile)
        results += _suppressed(inputs, data, source, table, profile)
        results.append(_row_hashes(inputs, data, source, table, profile))
    results += _text_pools(inputs, source)
    return results


def _sample(rows: int, limit: int) -> NDArray[np.int64]:
    """At most `limit` row positions, chosen with a fixed seed so reports are reproducible."""
    if rows <= limit:
        return np.arange(rows, dtype=np.int64)
    return np.sort(np.random.default_rng(0).choice(rows, size=limit, replace=False))


def _exact_matches(
    inputs: ValidationInputs,
    data: OutputData,
    source: Connector,
    table: TableSpec,
    profile: TableProfile,
) -> list[CheckResult]:
    settings = inputs.settings
    results: list[CheckResult] = []
    for column in (c for c in table.columns if c.sensitive and not c.primary_key):
        stats = profile.column(column.name)
        if stats is None:
            continue
        column_values = data.column(table.name, column.name)
        rows = _sample(column_values.size, settings.privacy_sample_size)
        values = column_values[rows]
        found = source_counts(
            source, table.name, column.name, values.tolist(), settings.privacy_lookup_batch
        )
        hit = np.array([v is not None and str(v) in found for v in values.tolist()], dtype=bool)
        scenario = traced(hit, data.label(table.name, SCENARIO_COLUMN)[rows])
        identifier = (
            stats.cardinality_ratio.value >= settings.privacy_identifier_min_ratio
            and column.type not in _TIMES
            and not (
                isinstance(column.generator, SemanticGen) and column.generator.kind in LOW_ENTROPY
            )
        )
        detail = None if identifier else "not identifier-like; reported only"
        results.append(
            _result(
                (table.name, column.name),
                "exact_match",
                float(hit.sum()),
                hard=identifier,
                detail=detail,
                scenario=scenario,
            )
        )
    return results


def _suppressed(
    inputs: ValidationInputs,
    data: OutputData,
    source: Connector,
    table: TableSpec,
    profile: TableProfile,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    for column in table.columns:
        stats = profile.column(column.name)
        if (
            stats is None
            or stats.sensitivity is not Sensitivity.INTERNAL
            or stats.categorical is None
        ):
            continue
        kept = {str(v.value) for v in stats.categorical.frequencies.value}
        values = data.column(table.name, column.name)
        unseen = {str(v) for v in values.tolist() if v is not None and str(v) not in kept}
        found = source_counts(
            source, table.name, column.name, unseen, inputs.settings.privacy_lookup_batch
        )
        rare = {v for v, n in found.items() if n < inputs.min_value_count}
        hit = np.array([v is not None and str(v) in rare for v in values.tolist()], dtype=bool)
        scenario = traced(hit, data.label(table.name, SCENARIO_COLUMN))
        results.append(
            _result(
                (table.name, column.name),
                "suppressed_values",
                float(hit.sum()),
                hard=True,
                scenario=scenario,
            )
        )
    return results


def _hashable(table: TableSpec) -> list[str]:
    rule = table.row_rule
    driving = rule.child_column if rule.kind == "per_parent" else None
    return [
        c.name
        for c in table.columns
        if not c.primary_key and c.name != driving and c.generator.type != "foreign_key"
    ]


def _digests(rows: Iterable[tuple[object, ...]]) -> set[str]:
    return {
        hashlib.blake2b("\x1f".join(map(_text, row)).encode(), digest_size=16).hexdigest()
        for row in rows
    }


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _row_hashes(
    inputs: ValidationInputs,
    data: OutputData,
    source: Connector,
    table: TableSpec,
    profile: TableProfile,
) -> CheckResult:
    names = _hashable(table)
    where = (table.name, None)
    if not names:
        return _result(where, "row_hash", None, hard=False, detail="no non-key columns")
    space = math.prod(_distinct(profile, n) for n in names)
    if space < profile.sample_rows:
        return _result(
            where, "row_hash", None, hard=False, detail=f"only {space} value combinations"
        )
    sampled = source.sample(table.name, names, profile.sample_rows, inputs.profile_seed)
    known = _digests(zip(*(sampled.column(n).to_pylist() for n in names), strict=True))
    rows = _sample(data.rows(table.name), inputs.settings.privacy_sample_size)
    columns = [data.column(table.name, n)[rows] for n in names]
    synthetic = [_digests([row]) for row in zip(*(c.tolist() for c in columns), strict=True)]
    hit = np.array([bool(digest & known) for digest in synthetic], dtype=bool)
    scenario = traced(hit, data.label(table.name, SCENARIO_COLUMN)[rows])
    return _result(where, "row_hash", float(hit.sum()), hard=True, scenario=scenario)


def _distinct(profile: TableProfile, name: str) -> int:
    stats = profile.column(name)
    return max(1, stats.distinct_count.value) if stats else 1


def _text_pools(inputs: ValidationInputs, source: Connector) -> list[CheckResult]:
    """Model-written pool values against every sensitive text column of the job's tables (A14)."""
    targets = _sensitive_text_columns(inputs.spec, inputs)
    results: list[CheckResult] = []
    for scenario in inputs.spec.scenarios:
        gens = [o.generator for o in scenario.overrides]
        gens += [c.generator for q in scenario.sequences for st in q.steps for c in st.columns]
        values = {v for g in gens if isinstance(g, TextPoolGen) for v in g.values}
        if not values:
            continue
        hits = 0
        for table, column in targets:
            found = source_counts(
                source, table, column, values, inputs.settings.privacy_lookup_batch
            )
            hits += len(found)
        results.append(
            _result(("*", None), "text_pool_values", float(hits), hard=True, scenario=scenario.name)
        )
    return results


def _sensitive_text_columns(
    spec: GenerationSpec, inputs: ValidationInputs
) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for table in spec.tables:
        profile = inputs.profiles.get(table.name)
        for column in table.columns:
            stats = profile.column(column.name) if profile else None
            if (
                stats is not None
                and stats.sensitivity is Sensitivity.SENSITIVE
                and column.type is ColumnType.TEXT
            ):
                found.append((table.name, column.name))
    return found
