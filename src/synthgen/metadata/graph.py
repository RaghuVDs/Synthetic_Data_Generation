"""Table dependency graph: which tables a job needs and the order to generate them (SPEC 8.1).

Edges run parent -> child. Unselected parents join as support tables. Self-references are filled
in a second pass. A mutual reference is broken at its nullable key, which is also deferred.
"""

from __future__ import annotations

import heapq
from collections.abc import Sequence

from synthgen.domain.errors import DependencyCycleError, UnknownIdentifierError
from synthgen.domain.metadata import Relationship, TableSet


def build_table_set(
    selected: Sequence[str], known_tables: Sequence[str], relationships: Sequence[Relationship]
) -> TableSet:
    """Add required parents, defer what must be filled later and order the tables."""
    unknown = sorted(set(selected) - set(known_tables))
    if unknown:
        raise UnknownIdentifierError(f"unknown tables selected: {unknown}")
    included = _with_parents(selected, relationships)
    in_scope = [
        r for r in relationships if r.child_table in included and r.parent_table in included
    ]
    deferred = [r for r in in_scope if r.is_self_reference]
    edges = [r for r in in_scope if not r.is_self_reference]
    while cycle := _find_cycle(sorted(included), edges):
        breakable = sorted((r for r in cycle if r.nullable), key=Relationship.label)
        if not breakable:
            labels = [r.label() for r in cycle]
            raise DependencyCycleError(f"cycle with no nullable key to break: {labels}")
        edges.remove(breakable[0])
        deferred.append(breakable[0])
    order = _topological_order(included, edges)
    return TableSet(
        selected=tuple(selected),
        support=tuple(t for t in order if t not in selected),
        order=tuple(order),
        relationships=tuple(in_scope),
        deferred=tuple(deferred),
    )


def _with_parents(selected: Sequence[str], relationships: Sequence[Relationship]) -> set[str]:
    included = set(selected)
    frontier = list(selected)
    while frontier:
        child = frontier.pop()
        for rel in relationships:
            if rel.child_table == child and rel.parent_table not in included:
                included.add(rel.parent_table)
                frontier.append(rel.parent_table)
    return included


def _find_cycle(nodes: list[str], edges: list[Relationship]) -> list[Relationship]:
    """Edges of one cycle, found by depth-first search; empty when the graph is acyclic."""
    outgoing: dict[str, list[Relationship]] = {n: [] for n in nodes}
    for edge in sorted(edges, key=Relationship.label):
        outgoing[edge.parent_table].append(edge)
    state: dict[str, int] = dict.fromkeys(nodes, 0)  # 0 new, 1 on path, 2 done
    path: list[Relationship] = []

    def visit(node: str) -> list[Relationship]:
        state[node] = 1
        for edge in outgoing[node]:
            nxt = edge.child_table
            if state[nxt] == 1:
                # Self-loops are excluded, so nxt is the parent of an edge on the current path.
                start = next(i for i, e in enumerate(path) if e.parent_table == nxt)
                return [*path[start:], edge]
            if state[nxt] == 0:
                path.append(edge)
                found = visit(nxt)
                if found:
                    return found
                path.pop()
        state[node] = 2
        return []

    for node in nodes:
        if state[node] == 0:
            found = visit(node)
            if found:
                return found
    return []


def _topological_order(nodes: set[str], edges: list[Relationship]) -> list[str]:
    """Kahn's algorithm; ties broken by name so the order is stable."""
    indegree = dict.fromkeys(nodes, 0)
    for edge in edges:
        indegree[edge.child_table] += 1
    ready = [n for n, d in indegree.items() if d == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        node = heapq.heappop(ready)
        order.append(node)
        for edge in edges:
            if edge.parent_table == node:
                indegree[edge.child_table] -= 1
                if indegree[edge.child_table] == 0:
                    heapq.heappush(ready, edge.child_table)
    return order
