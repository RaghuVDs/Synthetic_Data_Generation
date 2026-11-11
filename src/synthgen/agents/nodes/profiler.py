"""Node 3, code: compute profiles or reuse cached ones (SPEC 3.1, 8.3)."""

from __future__ import annotations

from synthgen.agents.nodes.deps import NodeDeps, StateUpdate
from synthgen.agents.state import JobState
from synthgen.domain.errors import SynthgenError
from synthgen.profiling.cache import profile_request, profile_version


def run(state: JobState, deps: NodeDeps) -> StateUpdate:
    """One profile version per table; cached while the schema and age allow."""
    table_set = state["table_set"]
    if table_set is None:
        raise SynthgenError("profiler ran before the table set was built")
    request = state["request"]
    with deps.open_source(request.source) as source:
        refs = {
            name: profile_version(
                profile_request(
                    request.source,
                    source.metadata,
                    name,
                    refresh=request.refresh_profiles,
                    now=deps.clock(),
                ),
                source.connector,
                deps.profiles,
                deps.settings.profiling,
            )
            for name in table_set.order
        }
    return {"profile_refs": refs}
