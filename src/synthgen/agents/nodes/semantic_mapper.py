"""Node 4, LLM: profile cards to a SemanticMap of column roles and business rules."""

from __future__ import annotations

from synthgen.agents.nodes.common import as_json, cards
from synthgen.agents.nodes.deps import NodeDeps, StateUpdate
from synthgen.agents.prompts.loader import render
from synthgen.agents.state import JobState
from synthgen.domain.job import ArtifactKind
from synthgen.domain.semantic import SemanticMap

ROLE = "semantic_mapper"


def run(state: JobState, deps: NodeDeps) -> StateUpdate:
    """Map every table's columns to roles."""
    _, messages = render(ROLE, {"PROFILE_CARDS": as_json(cards(state, deps))})
    semantic_map = deps.llm.complete(messages, SemanticMap)
    deps.artifacts.put(state["job_id"], ArtifactKind.SEMANTIC_MAP, semantic_map)
    return {"semantic_map": semantic_map}
