"""Node 4, LLM: profile cards to a SemanticMap of column roles and business rules.

Above `mapper_batch_tables` tables the cards go to the model in batches (SPEC 4.4).
"""

from __future__ import annotations

from functools import partial

from synthgen.agents.meaning import check_semantic_map
from synthgen.agents.nodes.common import ask, cards, job_columns
from synthgen.agents.nodes.deps import NodeDeps, StateUpdate
from synthgen.agents.state import JobState
from synthgen.domain.job import ArtifactKind
from synthgen.domain.semantic import SemanticMap
from synthgen.profiling.card import card_text

ROLE = "semantic_mapper"


def run(state: JobState, deps: NodeDeps) -> StateUpdate:
    """Map every table's columns to roles, batch by batch."""
    all_cards = cards(state, deps)
    columns = job_columns(state, deps)
    size = deps.settings.profiling.mapper_batch_tables
    tables: list[SemanticMap] = []
    for start in range(0, len(all_cards), size):
        batch = all_cards[start : start + size]
        batch_columns = {c.table: columns[c.table] for c in batch}
        tables.append(
            ask(
                state,
                deps,
                ROLE,
                {"PROFILE_CARDS": "\n\n".join(card_text(c) for c in batch)},
                SemanticMap,
                meaning=partial(check_semantic_map, columns=batch_columns),
            )
        )
    merged = SemanticMap(
        tables=[t for m in tables for t in m.tables],
        business_rules=[r for m in tables for r in m.business_rules],
    )
    deps.artifacts.put(state["job_id"], ArtifactKind.SEMANTIC_MAP, merged)
    return {"semantic_map": merged}
