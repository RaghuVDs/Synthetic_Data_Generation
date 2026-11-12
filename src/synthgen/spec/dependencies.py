"""Which columns a generator reads, and an order in which a table's columns can be generated."""

from __future__ import annotations

from collections.abc import Sequence, Set

from synthgen.domain.errors import ExpressionError
from synthgen.spec.constraints import (
    Constraint,
    ExpressionConstraint,
    OrderConstraint,
    UniqueConstraint,
)
from synthgen.spec.expressions import parse, references
from synthgen.spec.generators import ConditionalGen, DerivedGen, Generator
from synthgen.spec.models import ColumnSpec


def reads(gen: Generator) -> set[str]:
    """Columns the generator reads: own-table names, and `table.column` for the driving parent."""
    if isinstance(gen, DerivedGen):
        return references(parse(gen.expression))
    if isinstance(gen, ConditionalGen):
        inner = [reads(c.generator) for c in gen.cases] + [reads(gen.otherwise)]
        return {gen.column}.union(*inner)
    return set()


def constraint_reads(constraint: Constraint) -> set[str]:
    """Columns a constraint reads, including `table.column` names of the driving parent."""
    if isinstance(constraint, OrderConstraint):
        return {constraint.lower, constraint.upper}
    if isinstance(constraint, ExpressionConstraint):
        return references(parse(constraint.expression))
    if isinstance(constraint, UniqueConstraint):
        return set(constraint.columns)
    return {constraint.column}


def own_reads(gen: Generator) -> set[str]:
    """Columns of the same table the generator reads."""
    return {name for name in reads(gen) if "." not in name}


def parent_reads(gen: Generator) -> set[str]:
    """`table.column` names of the driving parent the generator reads."""
    return {name for name in reads(gen) if "." in name}


def generation_order(
    columns: Sequence[ColumnSpec], given: Set[str] = frozenset(), *, overlay: bool = False
) -> list[ColumnSpec]:
    """Columns in spec order, except that each comes after every column it reads.

    `given` names columns that already have values. In an overlay a column may read itself (its
    baseline value). Raises ExpressionError when columns read each other in a cycle.
    """
    pending = list(columns)
    done: set[str] = set(given)
    ordered: list[ColumnSpec] = []
    while pending:
        ready = next(
            (
                c
                for c in pending
                if own_reads(c.generator) - ({c.name} if overlay else set()) <= done
            ),
            None,
        )
        if ready is None:
            names = ", ".join(c.name for c in pending)
            raise ExpressionError(f"columns read each other in a cycle: {names}")
        pending.remove(ready)
        done.add(ready.name)
        ordered.append(ready)
    return ordered


def downstream(columns: Sequence[ColumnSpec], changed: set[str]) -> list[ColumnSpec]:
    """Columns (in generation order) that read a changed column, directly or through others."""
    affected = set(changed)
    found: list[ColumnSpec] = []
    for column in generation_order(columns):
        if column.name not in changed and own_reads(column.generator) & affected:
            affected.add(column.name)
            found.append(column)
    return found
