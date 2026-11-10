"""Bookkeeping step after the last work node: record why the job stopped.

It does no work; the routing already chose the terminal status (PLAN decision, M2).
"""

from __future__ import annotations

from synthgen.agents.nodes.deps import NodeDeps, StateUpdate
from synthgen.agents.state import JobState


def run(state: JobState, deps: NodeDeps) -> StateUpdate:
    """Copy the route's reason into the job's error field."""
    del deps
    route = state["route"]
    return {"error": route.reason if route else state["error"]}
