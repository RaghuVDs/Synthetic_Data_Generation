"""Baseline compiler: profiles in, baseline spec out. Pure code, no LLM (SPEC 3.1 node 7)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from synthgen.config import GenerationSettings
from synthgen.domain.errors import SpecCompilationError
from synthgen.domain.metadata import ColumnMeta, Relationship, Sensitivity, TableMeta, TableSet
from synthgen.domain.semantic import SemanticMap
from synthgen.profiling.models import ColumnProfile, InferredType, TableProfile
from synthgen.spec.checks import COPULA_GENERATORS
from synthgen.spec.correlation import as_matrix, nearest_correlation
from synthgen.spec.generators import (
    CategoricalGen,
    ConstantGen,
    DatetimeGen,
    EmpiricalGen,
    ForeignKeyGen,
    Generator,
    PatternGen,
    SemanticGen,
    SemanticKind,
    SequenceGen,
    UuidGen,
)
from synthgen.spec.models import (
    SPEC_FORMAT_VERSION,
    ChildCount,
    ColumnSpec,
    ColumnType,
    CorrelationSpec,
    FixedRows,
    GenerationSpec,
    OutputSpec,
    PerParentRows,
    ProfileRef,
    RowRule,
    TableSpec,
)
from synthgen.spec.semantic_kinds import K, semantic_kind

PATTERN_OTHER_SHARE = 0.5
"""Text columns whose stored frequencies miss more than this share are generated from masks."""

_TYPES = {
    InferredType.INTEGER: ColumnType.INTEGER,
    InferredType.FLOAT: ColumnType.FLOAT,
    InferredType.BOOLEAN: ColumnType.BOOLEAN,
    InferredType.TEXT: ColumnType.TEXT,
    InferredType.DATE: ColumnType.DATE,
    InferredType.DATETIME: ColumnType.DATETIME,
    InferredType.EMPTY: ColumnType.TEXT,
}


@dataclass(frozen=True)
class BaselineInputs:
    """Everything the baseline compiler reads."""

    table_set: TableSet
    tables: Mapping[str, TableMeta]
    profiles: Mapping[str, TableProfile]
    profile_versions: Mapping[str, str]
    semantic_map: SemanticMap | None
    row_targets: Mapping[str, int]
    seed: int


def compile_baseline(inputs: BaselineInputs, settings: GenerationSettings) -> GenerationSpec:
    """One table spec per table in generation order; no scenarios yet."""
    return GenerationSpec(
        version=SPEC_FORMAT_VERSION,
        seed=inputs.seed,
        profile_versions=[
            ProfileRef(table=t, version=inputs.profile_versions[t]) for t in inputs.table_set.order
        ],
        tables=[_table(name, inputs, settings) for name in inputs.table_set.order],
        scenarios=[],
        output=OutputSpec(compression="zstd", chunk_rows=settings.chunk_rows),
    )


def _table(name: str, inputs: BaselineInputs, settings: GenerationSettings) -> TableSpec:
    profile = inputs.profiles[name]
    keys = {r.child_column: r for r in inputs.table_set.relationships if r.child_table == name}
    semantics = inputs.semantic_map.for_table(name) if inputs.semantic_map else None
    label = semantics.label_column if semantics else None
    columns = []
    for meta in inputs.tables[name].columns:
        column = profile.column(meta.name)
        if column is None:
            raise SpecCompilationError(f"profile of {name} has no column {meta.name}")
        columns.append(_column(meta, column, keys.get(meta.name), is_label=meta.name == label))
    return TableSpec(
        name=name,
        row_rule=_row_rule(name, inputs, settings),
        columns=columns,
        correlation=_correlation(profile, columns, settings),
    )


def _correlation(
    profile: TableProfile, columns: list[ColumnSpec], settings: GenerationSettings
) -> CorrelationSpec | None:
    """The profiled Spearman matrix over columns with quantile generators that correlate.

    A column joins when its strongest |rho| with another reaches `copula_min_rho`; wide tables
    keep the `max_copula_columns` columns with the largest total |rho|.
    """
    cross = profile.cross_column
    eligible = {c.name for c in columns if c.generator.type in COPULA_GENERATORS}
    names = [n for n in cross.spearman_columns if n in eligible]
    if len(names) < 2:  # noqa: PLR2004 - a correlation needs two columns
        return None
    positions = [cross.spearman_columns.index(n) for n in names]
    matrix = as_matrix(cross.spearman)[np.ix_(positions, positions)]
    strength = np.abs(matrix - np.eye(len(names)))
    joined = [i for i in range(len(names)) if strength[i].max() >= settings.copula_min_rho]
    joined = sorted(sorted(joined, key=lambda i: -strength[i].sum())[: settings.max_copula_columns])
    if len(joined) < 2:  # noqa: PLR2004
        return None
    kept = nearest_correlation(matrix[np.ix_(joined, joined)])
    return CorrelationSpec(columns=[names[i] for i in joined], spearman=kept.tolist())


def _row_rule(name: str, inputs: BaselineInputs, settings: GenerationSettings) -> RowRule:
    driver = next(
        (
            r
            for r in inputs.table_set.relationships
            if r.child_table == name and r.drives_cardinality and not r.is_self_reference
        ),
        None,
    )
    target = inputs.row_targets.get(name)
    if driver is None:
        return FixedRows(kind="fixed", rows=target or settings.default_root_rows)
    shape = next(
        s for s in inputs.profiles[name].relationships if s.child_column == driver.child_column
    )
    return PerParentRows(
        kind="per_parent",
        parent_table=driver.parent_table,
        parent_column=driver.parent_column,
        child_column=driver.child_column,
        counts=[ChildCount(children=c, weight=float(n)) for c, n in shape.counts.value],
        target_rows=target,
    )


def _column(
    meta: ColumnMeta, column: ColumnProfile, key: Relationship | None, *, is_label: bool
) -> ColumnSpec:
    sensitive = column.sensitivity is Sensitivity.SENSITIVE
    column_type = _TYPES[column.inferred_type]
    if sensitive and not (
        column_type is ColumnType.DATE and _sensitive_kind(meta, column) is K.DATE_OF_BIRTH
    ):
        column_type = ColumnType.TEXT
    return ColumnSpec(
        name=meta.name,
        type=column_type,
        nullable=meta.nullable,
        null_rate=0.0 if meta.primary_key or is_label else column.null_rate.value,
        primary_key=meta.primary_key,
        unique=meta.unique and not meta.primary_key,
        sensitive=sensitive,
        generator=_generator(meta, column, key, is_label=is_label),
    )


def _generator(  # noqa: PLR0911 - one branch per kind of column
    meta: ColumnMeta, column: ColumnProfile, key: Relationship | None, *, is_label: bool
) -> Generator:
    if meta.primary_key:
        if column.inferred_type is InferredType.INTEGER:
            return SequenceGen(type="sequence", start=1, step=1)
        return UuidGen(type="uuid")
    if key is not None:
        return ForeignKeyGen(
            type="foreign_key",
            parent_table=key.parent_table,
            parent_column=key.parent_column,
            distribution="uniform",
            exponent=None,
            hot_keys=None,
            hot_share=None,
            group_size=None,
        )
    if is_label and column.top_value is not None:
        return ConstantGen(type="constant", value=column.top_value.value)
    if meta.unique and column.inferred_type is InferredType.TEXT:
        return _pattern(column)  # observed values cannot stay unique once more rows are asked for
    if column.sensitivity is Sensitivity.SENSITIVE:
        kind = _sensitive_kind(meta, column)
        return SemanticGen(type="semantic", kind=kind) if kind else _pattern(column)
    if column.inferred_type is InferredType.EMPTY:
        return _pattern(column)
    if column.numeric is not None:
        decimals = (
            0
            if column.inferred_type is InferredType.INTEGER
            else column.numeric.decimal_places.value
        )
        return EmpiricalGen(
            type="empirical", quantiles=column.numeric.quantiles.value, decimals=decimals
        )
    if column.temporal is not None:
        return _datetime(column)
    return _categorical_or_pattern(column)


def _normalised(weights: list[float]) -> list[float]:
    total = sum(weights)
    return [w / total for w in weights] if total else [1.0 / len(weights)] * len(weights)


def _sensitive_kind(meta: ColumnMeta, column: ColumnProfile) -> SemanticKind | None:
    if column.sensitivity is not Sensitivity.SENSITIVE:
        return None
    return semantic_kind(meta.name, column.semantic_type)


def _pattern(column: ColumnProfile) -> Generator:
    masks = column.text.masks.value if column.text else []
    if not masks:
        return ConstantGen(type="constant", value=None)
    luhn = column.semantic_type == "card_number" and all("9" in m.mask for m in masks)
    return PatternGen(
        type="pattern",
        masks=[m.mask for m in masks],
        weights=_normalised([m.share for m in masks]),
        check_digit="luhn" if luhn else None,
    )


def _datetime(column: ColumnProfile) -> Generator:
    temporal = column.temporal
    if temporal is None:
        raise SpecCompilationError(f"{column.name} has no temporal profile")
    is_datetime = column.inferred_type is InferredType.DATETIME
    return DatetimeGen(
        type="datetime",
        start=temporal.minimum.value,
        end=temporal.maximum.value,
        granularity="datetime" if is_datetime else "date",
        weekday_weights=_normalised([float(c) for c in temporal.weekday_counts.value]),
        hour_weights=_normalised([float(c) for c in temporal.hour_counts.value])
        if is_datetime
        else None,
    )


def _categorical_or_pattern(column: ColumnProfile) -> Generator:
    """Stored frequencies when they cover most rows (rare-value mass is redistributed, PLAN A8)."""
    categorical = column.categorical
    if categorical is None or not categorical.frequencies.value:
        return _pattern(column)
    kept = sum(v.count for v in categorical.frequencies.value)
    if categorical.other_count.value > PATTERN_OTHER_SHARE * (kept + categorical.other_count.value):
        return _pattern(column)
    return CategoricalGen(
        type="categorical",
        values=[v.value for v in categorical.frequencies.value],
        weights=_normalised([float(v.count) for v in categorical.frequencies.value]),
    )
