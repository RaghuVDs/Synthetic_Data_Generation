"""Node 4, LLM: profile cards to a SemanticMap of column roles and business rules.

Cards go to the model in batches (SPEC 4.4): at most `mapper_batch_tables` tables and
`mapper_batch_columns` columns per call. A wider table is split into column slices whose maps
are merged afterwards, so tables of thousands of columns map in bounded calls.
"""

from __future__ import annotations

from functools import partial

from synthgen.agents.meaning import check_semantic_map
from synthgen.agents.nodes.common import ask, cards, job_columns
from synthgen.agents.nodes.deps import NodeDeps, StateUpdate
from synthgen.agents.state import JobState
from synthgen.domain.job import ArtifactKind
from synthgen.domain.semantic import SemanticMap, TableSemantics
from synthgen.profiling.card import LlmSafeProfile, card_text

ROLE = "semantic_mapper"


def mapper_batches(
    all_cards: list[LlmSafeProfile], max_tables: int, max_columns: int
) -> list[list[LlmSafeProfile]]:
    """Cards cut into column slices of at most `max_columns`, packed into bounded batches."""
    slices = [
        card.model_copy(update={"columns": card.columns[start : start + max_columns]})
        for card in all_cards
        for start in range(0, max(1, len(card.columns)), max_columns)
    ]
    batches: list[list[LlmSafeProfile]] = []
    for piece in slices:
        current = batches[-1] if batches else None
        fits = (
            current is not None
            and len(current) < max_tables
            and sum(len(c.columns) for c in current) + len(piece.columns) <= max_columns
        )
        if current is not None and fits:
            current.append(piece)
        else:
            batches.append([piece])
    return batches


def merge_maps(maps: list[SemanticMap]) -> SemanticMap:
    """One map: each table once, columns in slice order, the first stated key of each kind."""
    tables: dict[str, TableSemantics] = {}
    for table in (t for m in maps for t in m.tables):
        seen = tables.get(table.table)
        if seen is None:
            tables[table.table] = table
            continue
        tables[table.table] = seen.model_copy(
            update={
                "columns": [*seen.columns, *table.columns],
                "entity_key": seen.entity_key or table.entity_key,
                "event_time": seen.event_time or table.event_time,
                "label_column": seen.label_column or table.label_column,
            }
        )
    rules = {r.rule: r for m in maps for r in m.business_rules}
    return SemanticMap(tables=list(tables.values()), business_rules=list(rules.values()))


def run(state: JobState, deps: NodeDeps) -> StateUpdate:
    """Map every table's columns to roles, batch by batch."""
    columns = job_columns(state, deps)
    settings = deps.settings.profiling
    maps = []
    for batch in mapper_batches(
        cards(state, deps), settings.mapper_batch_tables, settings.mapper_batch_columns
    ):
        batch_columns = {c.table: [col.name for col in c.columns] for c in batch}
        maps.append(
            ask(
                state,
                deps,
                ROLE,
                {"PROFILE_CARDS": "\n\n".join(card_text(c) for c in batch)},
                SemanticMap,
                meaning=partial(check_semantic_map, columns=batch_columns, key_columns=columns),
            )
        )
    merged = merge_maps(maps)
    deps.artifacts.put(state["job_id"], ArtifactKind.SEMANTIC_MAP, merged)
    return {"semantic_map": merged}
