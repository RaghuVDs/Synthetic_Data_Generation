"""Node 2, code: load metadata, add required parent tables and order them (SPEC 3.1)."""

from __future__ import annotations

from synthgen.agents.nodes.deps import NodeDeps, StateUpdate
from synthgen.agents.state import JobState
from synthgen.metadata.graph import build_table_set


def run(state: JobState, deps: NodeDeps) -> StateUpdate:
    """Resolve the selected tables into an ordered table set."""
    request = state["request"]
    with deps.open_source(request.source) as source:
        names = [t.name for t in source.metadata.tables()]
        table_set = build_table_set(request.tables, names, source.metadata.relationships())
    return {"table_set": table_set}
