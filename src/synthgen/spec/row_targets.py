"""Root sizes implied by row targets on child tables (PLAN Q3, revised in M7).

A target on a child table walks up its driving relationships, dividing by the profiled mean
children per parent at each step, so the root gets enough rows for the child to keep its
profiled shape. A root named by several targets takes the largest. The child keeps its own
target, which then only fine-tunes the count.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from synthgen.domain.metadata import Relationship, TableSet
from synthgen.profiling.models import TableProfile


def implied_root_rows(
    table_set: TableSet, profiles: Mapping[str, TableProfile], targets: Mapping[str, int]
) -> dict[str, int]:
    """Rows for each root table that a child target implies; explicit root targets are kept."""
    drivers = {
        r.child_table: r
        for r in table_set.relationships
        if r.drives_cardinality and not r.is_self_reference
    }
    found: dict[str, int] = {}
    for table, rows in targets.items():
        if table not in drivers:
            continue
        root, needed = _walk_up(table, float(rows), drivers, profiles)
        if root not in targets:
            found[root] = max(found.get(root, 1), needed)
    return found


def _walk_up(
    table: str,
    rows: float,
    drivers: Mapping[str, Relationship],
    profiles: Mapping[str, TableProfile],
) -> tuple[str, int]:
    seen = {table}
    while table in drivers:
        driver = drivers[table]
        mean = _mean_children(profiles[table], driver)
        if mean > 0:
            rows /= mean
        table = driver.parent_table
        if table in seen:
            break
        seen.add(table)
    return table, max(1, math.ceil(rows))


def _mean_children(profile: TableProfile, driver: Relationship) -> float:
    """Profiled children per parent row, parents without children included; 0 when unknown."""
    shape = next(
        (
            s
            for s in profile.relationships
            if s.child_column == driver.child_column and s.parent_table == driver.parent_table
        ),
        None,
    )
    pairs = shape.counts.value if shape is not None else []
    parents = sum(n for _, n in pairs)
    return sum(c * n for c, n in pairs) / parents if parents else 0.0
