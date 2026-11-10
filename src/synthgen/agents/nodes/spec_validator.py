"""Node 9, code: merge overlays into the baseline (or take a patched spec) and validate it."""

from __future__ import annotations

from synthgen.agents.nodes.common import load_spec, spec_ref
from synthgen.agents.nodes.deps import NodeDeps, StateUpdate
from synthgen.agents.state import JobState
from synthgen.domain.job import ArtifactKind
from synthgen.spec.merge import merge_overlays
from synthgen.spec.models import GenerationSpec
from synthgen.spec.validator import validate_spec


def run(state: JobState, deps: NodeDeps) -> StateUpdate:
    """Store a valid spec and point at it; otherwise return the errors."""
    job_id = state["job_id"]
    if state["spec_ref"] is None:
        baseline = deps.artifacts.get(job_id, ArtifactKind.BASELINE_SPEC, GenerationSpec)
        spec = merge_overlays(baseline, state["overlays"])
    else:
        spec = load_spec(state, deps)
    errors = validate_spec(spec, deps.settings.generation)
    if errors:
        return {"spec_errors": errors, "spec_ref": None}
    version = deps.artifacts.put(job_id, ArtifactKind.SPEC, spec)
    return {"spec_errors": [], "spec_ref": spec_ref(version)}
