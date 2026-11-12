"""Ask the source which values it holds, in batches of bound parameters (never SQL text)."""

from __future__ import annotations

from collections.abc import Iterable

from synthgen.domain.ports import Connector
from synthgen.domain.query import AggExpr, AggFunc, Condition, Filter, Operator, Scalar


def source_counts(
    source: Connector, table: str, column: str, values: Iterable[Scalar], batch: int
) -> dict[str, int]:
    """How many source rows hold each of `values` (as text); values the source lacks are absent."""
    wanted = sorted({v for v in values if v is not None}, key=str)
    found: dict[str, int] = {}
    count = AggExpr(func=AggFunc.COUNT, column=None, alias="n")
    for start in range(0, len(wanted), batch):
        chunk = tuple(wanted[start : start + batch])
        where = Filter(conditions=(Condition(column=column, op=Operator.IN, value=chunk),))
        for row in source.aggregate(table, [count], group_by=[column], where=where):
            found[str(row[column])] = int(row["n"])
    return found
