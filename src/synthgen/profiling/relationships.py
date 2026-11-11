"""Cross-table statistics per relationship (SPEC 8.3 G), computed through SQL."""

from __future__ import annotations

from collections import Counter

from synthgen.domain.metadata import Relationship
from synthgen.domain.ports import Connector
from synthgen.domain.query import AggExpr, AggFunc
from synthgen.profiling.models import ChildrenPerParent, Measured, StatSource


def children_per_parent(connector: Connector, rel: Relationship) -> ChildrenPerParent:
    """Children per parent row (including parents with none) and orphan child rows, via SQL."""
    n = AggExpr(func=AggFunc.COUNT, column=None, alias="n")
    children = {
        r[rel.child_column]: int(r["n"])
        for r in connector.aggregate(rel.child_table, [n], group_by=[rel.child_column])
        if r[rel.child_column] is not None
    }
    parents = [
        r[rel.parent_column]
        for r in connector.aggregate(rel.parent_table, [n], group_by=[rel.parent_column])
    ]
    known = set(parents)
    per_parent = Counter(children.get(key, 0) for key in parents)
    return ChildrenPerParent(
        child_column=rel.child_column,
        parent_table=rel.parent_table,
        parent_column=rel.parent_column,
        counts=Measured[list[tuple[int, int]]](
            value=sorted(per_parent.items()), source=StatSource.SQL
        ),
        orphan_rows=Measured[int](
            value=sum(c for key, c in children.items() if key not in known), source=StatSource.SQL
        ),
    )
