"""The fourteen primitives (SPEC 7.2): the spec construct each needs and the columns it acts on.

Which constructs exist is read from the spec models themselves, so the feasibility check follows
the engine as it grows instead of a hard-coded list.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from enum import StrEnum

from synthgen.domain.plan import Primitive
from synthgen.domain.semantic import ColumnRole
from synthgen.spec import generators
from synthgen.spec.models import ScenarioSpec


class Construct(StrEnum):
    """Spec constructs that carry primitives (SPEC 7.2, third column)."""

    OVERRIDE = "generator override"
    CATEGORICAL = "categorical override"
    DATETIME = "datetime override"
    CONDITIONAL = "conditional or derived generator"
    SEQUENCE = "SequenceSpec"
    CARDINALITY = "cardinality override"
    FK_WEIGHTS = "weighted foreign keys"
    SHARED_GROUP = "foreign keys from a shared group"
    CORRELATION = "correlation override"
    BOUNDARY = "boundary generator"
    TEXT_POOL = "text_pool generator"
    DEFECTS = "defects"


@dataclass(frozen=True)
class PrimitiveRule:
    """What a primitive needs: a construct, and a lever column with one of these roles."""

    construct: Construct
    roles: frozenset[ColumnRole]
    alternative: Primitive | None


_ANY = frozenset(ColumnRole) - {ColumnRole.IDENTIFIER, ColumnRole.SENSITIVE_IDENTIFIER}
_VALUES = frozenset(
    {
        ColumnRole.MEASURE,
        ColumnRole.CATEGORY,
        ColumnRole.STATUS,
        ColumnRole.LOCATION,
        ColumnRole.OTHER,
    }
)
_CATEGORIES = frozenset(
    {ColumnRole.CATEGORY, ColumnRole.STATUS, ColumnRole.LOCATION, ColumnRole.LABEL}
)
_KEYS = frozenset({ColumnRole.FOREIGN_KEY, ColumnRole.ENTITY_KEY})
_TIME = frozenset({ColumnRole.EVENT_TIME})
P = Primitive
RULES: dict[Primitive, PrimitiveRule] = {
    P.VALUE_SHIFT: PrimitiveRule(Construct.OVERRIDE, _VALUES, None),
    P.MIX_SHIFT: PrimitiveRule(Construct.CATEGORICAL, _CATEGORIES, P.VALUE_SHIFT),
    P.CONDITIONAL_RULE: PrimitiveRule(Construct.CONDITIONAL, _VALUES, P.VALUE_SHIFT),
    P.SEQUENCE: PrimitiveRule(Construct.SEQUENCE, _TIME | _KEYS, P.TIMING_SHIFT),
    P.TIMING_SHIFT: PrimitiveRule(Construct.DATETIME, _TIME, P.VALUE_SHIFT),
    P.GAP_OR_ABSENCE: PrimitiveRule(Construct.SEQUENCE, _TIME | _KEYS, P.TIMING_SHIFT),
    P.RELATIONSHIP_SHAPE: PrimitiveRule(Construct.CARDINALITY, _KEYS, P.VOLUME_SKEW),
    P.CROSS_ENTITY_LINK: PrimitiveRule(Construct.SHARED_GROUP, _KEYS, P.VALUE_SHIFT),
    P.STATE_PATH: PrimitiveRule(
        Construct.SEQUENCE, frozenset({ColumnRole.STATUS}) | _TIME, P.MIX_SHIFT
    ),
    P.CORRELATION_CHANGE: PrimitiveRule(
        Construct.CORRELATION, frozenset({ColumnRole.MEASURE}), P.VALUE_SHIFT
    ),
    P.BOUNDARY_VALUE: PrimitiveRule(Construct.BOUNDARY, _ANY, P.VALUE_SHIFT),
    P.VOLUME_SKEW: PrimitiveRule(Construct.FK_WEIGHTS, _KEYS, P.MIX_SHIFT),
    P.TEXT_CONTENT: PrimitiveRule(
        Construct.TEXT_POOL, frozenset({ColumnRole.FREE_TEXT}), P.MIX_SHIFT
    ),
    P.RULE_BREAK: PrimitiveRule(Construct.DEFECTS, _ANY, P.BOUNDARY_VALUE),
}


def _generator_types() -> set[str]:
    union = typing.get_args(generators.Generator)[0]
    return {typing.get_args(m.model_fields["type"].annotation)[0] for m in typing.get_args(union)}


def _fk_distributions() -> set[str]:
    return set(typing.get_args(generators.ForeignKeyGen.model_fields["distribution"].annotation))


def available_constructs() -> frozenset[Construct]:
    """Constructs the current spec models can express."""
    types = _generator_types()
    found = {Construct.OVERRIDE}
    checks = {
        Construct.CATEGORICAL: "categorical" in types,
        Construct.DATETIME: "datetime" in types,
        Construct.CONDITIONAL: bool({"conditional", "derived"} & types),
        Construct.BOUNDARY: "boundary" in types,
        Construct.TEXT_POOL: "text_pool" in types,
        Construct.FK_WEIGHTS: bool({"zipf", "weighted"} & _fk_distributions()),
        Construct.SHARED_GROUP: "shared_group" in _fk_distributions(),
        Construct.SEQUENCE: "sequence" in ScenarioSpec.model_fields,
        Construct.DEFECTS: "defects" in ScenarioSpec.model_fields,
        Construct.CARDINALITY: "cardinality" in ScenarioSpec.model_fields,
        Construct.CORRELATION: "correlations" in ScenarioSpec.model_fields,
    }
    return frozenset(found | {c for c, ok in checks.items() if ok})
