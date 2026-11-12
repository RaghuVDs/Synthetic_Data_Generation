"""Does a generator suit its column? One reason why not, or None (SPEC 8.4 spec validation)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from synthgen.config import GenerationSettings
from synthgen.domain.errors import ExpressionError
from synthgen.spec.expressions import ExprType, parse, type_of
from synthgen.spec.generators import (
    BoundaryGen,
    ConditionalGen,
    DerivedGen,
    ForeignKeyGen,
    Generator,
    SemanticGen,
    SemanticKind,
)
from synthgen.spec.models import EXPR_TYPES, ColumnSpec, ColumnType, GenerationSpec, TableSpec

_NUMERIC = (ColumnType.INTEGER, ColumnType.FLOAT, ColumnType.BOOLEAN)
_TIMES = (ColumnType.DATE, ColumnType.DATETIME)
SUITS: dict[str, tuple[ColumnType, ...]] = {
    "sequence": (ColumnType.INTEGER,),
    "uuid": (ColumnType.TEXT,),
    "empirical": _NUMERIC,
    "distribution": _NUMERIC,
    "datetime": _TIMES,
    "pattern": (ColumnType.TEXT,),
    "text_pool": (ColumnType.TEXT,),
}
SENSITIVE_OK = ("pattern", "semantic", "constant")
_RESULTS: dict[ExprType, tuple[ColumnType, ...]] = {
    ExprType.NUMBER: _NUMERIC,
    ExprType.BOOLEAN: (ColumnType.BOOLEAN, ColumnType.INTEGER),
    ExprType.TEXT: (ColumnType.TEXT,),
    ExprType.DATE: (ColumnType.DATE,),
    ExprType.DATETIME: (ColumnType.DATETIME,),
}


class GeneratorChecker:
    """Checks generators against one spec, the tables generated before a table, and settings."""

    def __init__(self, spec: GenerationSpec, settings: GenerationSettings) -> None:
        """Bind the spec whose tables and keys the checks look up."""
        self._spec = spec
        self._settings = settings

    def problem(
        self, table: TableSpec, column: ColumnSpec, gen: Generator, earlier: Sequence[str]
    ) -> str | None:
        """Why this generator cannot fill this column, or None when it can."""
        found = self._shape_problem(column, gen)
        if found:
            return found
        if isinstance(gen, SemanticGen | BoundaryGen):
            return _value_problem(gen, column)
        if isinstance(gen, ForeignKeyGen):
            return self._foreign_key(gen, earlier)
        if isinstance(gen, DerivedGen):
            return self._derived(gen, table, column)
        if isinstance(gen, ConditionalGen):
            return self._conditional(gen, table, column, earlier)
        return None

    def _shape_problem(self, column: ColumnSpec, gen: Generator) -> str | None:
        """Checks that need only the generator and the column."""
        allowed = SUITS.get(gen.type)
        if allowed is not None and column.type not in allowed:
            return f"{gen.type} generator does not suit {column.type} column {column.name}"
        if column.sensitive and gen.type not in SENSITIVE_OK and not column.primary_key:
            return f"sensitive column {column.name} must use a pattern or semantic generator"
        if gen.type == "datetime" and (gen.granularity == "datetime") != (
            column.type is ColumnType.DATETIME
        ):
            return f"datetime granularity {gen.granularity} does not suit {column.type} column"
        if gen.type == "text_pool" and len(gen.values) > self._settings.text_pool_max_values:
            return f"text pool has more than {self._settings.text_pool_max_values} values"
        return None

    def _foreign_key(self, gen: ForeignKeyGen, earlier: Sequence[str]) -> str | None:
        parent = self._spec.table(gen.parent_table)
        if gen.parent_table not in earlier or parent is None:
            return f"foreign key to {gen.parent_table}, which is not generated earlier"
        key = parent.column(gen.parent_column)
        if key is None or not (key.primary_key or key.unique):
            return f"foreign key target {gen.parent_table}.{gen.parent_column} is not a unique key"
        return None

    def expression_types(self, table: TableSpec) -> dict[str, ExprType]:
        """Names an expression in this table may read, with their types."""
        types = {c.name: EXPR_TYPES[c.type] for c in table.columns}
        rule = table.row_rule
        parent = self._spec.table(rule.parent_table) if rule.kind == "per_parent" else None
        if parent is not None:
            types |= {f"{parent.name}.{c.name}": EXPR_TYPES[c.type] for c in parent.columns}
        return types

    def _derived(self, gen: DerivedGen, table: TableSpec, column: ColumnSpec) -> str | None:
        try:
            result = type_of(parse(gen.expression), self.expression_types(table))
        except ExpressionError as error:
            return f"expression for {column.name}: {error}"
        if column.type not in _RESULTS[result]:
            return f"expression gives {result.value}, which does not suit {column.type} column"
        return None

    def _conditional(
        self, gen: ConditionalGen, table: TableSpec, column: ColumnSpec, earlier: Sequence[str]
    ) -> str | None:
        if gen.column == column.name or table.column(gen.column) is None:
            return f"conditional on {gen.column}, which is not another column of {table.name}"
        for inner in [*(c.generator for c in gen.cases), gen.otherwise]:
            found = self.problem(table, column, inner, earlier)
            if found:
                return f"conditional case: {found}"
        return None


def _value_problem(gen: SemanticGen | BoundaryGen, column: ColumnSpec) -> str | None:
    return _semantic(gen, column) if isinstance(gen, SemanticGen) else _boundary(gen, column)


def _semantic(gen: SemanticGen, column: ColumnSpec) -> str | None:
    wanted = (
        (ColumnType.DATE, ColumnType.TEXT)
        if gen.kind is SemanticKind.DATE_OF_BIRTH
        else (ColumnType.TEXT,)
    )
    if column.type not in wanted:
        return f"semantic {gen.kind} does not suit {column.type} column {column.name}"
    return None


def _boundary(gen: BoundaryGen, column: ColumnSpec) -> str | None:
    numeric = column.type in _NUMERIC
    for point in gen.points:
        if numeric != isinstance(point, (int, float)) or isinstance(point, bool):
            return f"boundary point {point!r} does not suit {column.type} column {column.name}"
        if column.type in _TIMES and not _is_time(str(point), column.type):
            return f"boundary point {point!r} is not an ISO {column.type}"
    return None


def _is_time(text: str, kind: ColumnType) -> bool:
    try:
        np.datetime64(text.replace(" ", "T"), "s")
    except ValueError:
        return False
    return kind is ColumnType.DATETIME or len(text) == len("2000-01-01")
