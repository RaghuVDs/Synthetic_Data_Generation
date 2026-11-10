"""Node 7, code: compile the baseline spec from profiles. No LLM (SPEC 3.1, 8.4)."""

from __future__ import annotations

from synthgen.agents.nodes.common import load_profile
from synthgen.agents.nodes.deps import NodeDeps, StateUpdate
from synthgen.agents.state import JobState
from synthgen.domain.errors import SynthgenError
from synthgen.domain.job import ArtifactKind
from synthgen.domain.metadata import TableSet
from synthgen.spec.baseline import BaselineInputs, compile_baseline


def row_targets(state: JobState, table_set: TableSet) -> dict[str, int]:
    """`--rows` targets, then the intent's volume hint for root tables (PLAN Q3)."""
    targets = {t.table: t.rows for t in state["request"].row_targets}
    intent = state["intent"]
    hint = intent.volume_hint if intent else None
    children = {
        r.child_table
        for r in table_set.relationships
        if r.drives_cardinality and not r.is_self_reference
    }
    if hint is not None:
        for name in table_set.order:
            if name not in children:
                targets.setdefault(name, hint)
    return targets


def run(state: JobState, deps: NodeDeps) -> StateUpdate:
    """Store the baseline; earlier overlays and merged specs no longer apply."""
    table_set = state["table_set"]
    if table_set is None:
        raise SynthgenError("baseline compiler ran before the table set was built")
    with deps.open_source(state["request"].source) as source:
        tables = {name: source.metadata.table(name) for name in table_set.order}
    inputs = BaselineInputs(
        table_set=table_set,
        tables=tables,
        profiles={name: load_profile(state, deps, name) for name in table_set.order},
        profile_versions=state["profile_refs"],
        semantic_map=state["semantic_map"],
        row_targets=row_targets(state, table_set),
        seed=state["request"].seed,
    )
    spec = compile_baseline(inputs, deps.settings.generation)
    deps.artifacts.put(state["job_id"], ArtifactKind.BASELINE_SPEC, spec)
    return {"overlays": [], "spec_ref": None, "spec_errors": []}
