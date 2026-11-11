"""Node 12, LLM: failed checks to a SpecPatch on allow-listed paths, applied by code."""

from __future__ import annotations

from synthgen.agents.meaning import check_patch
from synthgen.agents.nodes.common import as_json, ask, cards_text, load_spec, spec_ref
from synthgen.agents.nodes.deps import NodeDeps, StateUpdate
from synthgen.agents.state import JobState
from synthgen.domain.errors import SynthgenError
from synthgen.domain.job import ArtifactKind
from synthgen.domain.patch import SpecPatch
from synthgen.domain.report import ValidationReport
from synthgen.spec.models import GenerationSpec
from synthgen.spec.patch import apply_patch

ROLE = "repair_analyst"


def repair_columns(spec: GenerationSpec, report: ValidationReport) -> set[tuple[str, str]]:
    """Columns the overlays change or label, and the columns of failed checks."""
    found = {(o.table, o.column) for s in spec.scenarios for o in s.overrides}
    found |= {(lb.table, lb.column) for s in spec.scenarios for lb in s.labels}
    return found | {(f.table, f.column) for f in report.failures() if f.column}


def run(state: JobState, deps: NodeDeps) -> StateUpdate:
    """Patch the spec, or hand the plan back to the user when a patch cannot fix it."""
    report = state["dry_run_report"]
    if report is None:
        raise SynthgenError("repair analyst ran without a dry-run report")
    counters = state["counters"]
    counters = counters.model_copy(update={"validation_repair": counters.validation_repair + 1})
    spec = load_spec(state, deps)
    patch = ask(
        state,
        deps,
        ROLE,
        {
            "FAILED_CHECKS": as_json(report.failures()),
            "OVERLAYS": as_json(spec.scenarios),
            "PROFILE_CARDS": cards_text(
                state, deps, state["semantic_map"], repair_columns(spec, report)
            ),
        },
        SpecPatch,
        meaning=lambda value: check_patch(value, spec),
    )
    update: StateUpdate = {"patch": patch, "counters": counters}
    if patch.plan_change_required:
        return update
    version = deps.artifacts.put(state["job_id"], ArtifactKind.SPEC, apply_patch(spec, patch))
    return {**update, "spec_ref": spec_ref(version)}
