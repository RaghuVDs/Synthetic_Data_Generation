"""Node 12, LLM: failed checks to a SpecPatch, applied by code. M3 adds the path allow-list."""

from __future__ import annotations

from synthgen.agents.nodes.common import as_json, cards, load_spec, spec_ref
from synthgen.agents.nodes.deps import NodeDeps, StateUpdate
from synthgen.agents.prompts.loader import render
from synthgen.agents.state import JobState
from synthgen.domain.errors import LlmOutputError, PatchError, SynthgenError
from synthgen.domain.job import ArtifactKind
from synthgen.domain.patch import SpecPatch
from synthgen.spec.patch import apply_patch

ROLE = "repair_analyst"


def run(state: JobState, deps: NodeDeps) -> StateUpdate:
    """Patch the spec, or hand the plan back to the user when a patch cannot fix it."""
    report = state["dry_run_report"]
    if report is None:
        raise SynthgenError("repair analyst ran without a dry-run report")
    counters = state["counters"]
    counters = counters.model_copy(update={"validation_repair": counters.validation_repair + 1})
    spec = load_spec(state, deps)
    _, messages = render(
        ROLE,
        {
            "FAILED_CHECKS": as_json(report.failures()),
            "OVERLAYS": as_json(spec.scenarios),
            "PROFILE_CARDS": as_json(cards(state, deps)),
        },
    )
    patch = deps.llm.complete(messages, SpecPatch)
    update: StateUpdate = {"patch": patch, "counters": counters}
    if patch.plan_change_required:
        return update
    try:
        patched = apply_patch(spec, patch)
    except PatchError as exc:
        raise LlmOutputError(f"repair patch cannot be applied: {exc}") from exc
    version = deps.artifacts.put(state["job_id"], ArtifactKind.SPEC, patched)
    return {**update, "spec_ref": spec_ref(version)}
