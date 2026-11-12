"""Build the inputs of the validation nodes: the spec, profiles, plan and run options."""

from __future__ import annotations

from synthgen.agents.nodes.common import load_profile, load_spec
from synthgen.agents.nodes.deps import NodeDeps
from synthgen.agents.state import JobState
from synthgen.engine.runner import RunOptions
from synthgen.validation.output import ValidationInputs


def validation_inputs(state: JobState, deps: NodeDeps, options: RunOptions) -> ValidationInputs:
    """What the checks compare the output with; the source is not opened here."""
    spec = load_spec(state, deps)
    return ValidationInputs(
        spec=spec,
        options=options,
        settings=deps.settings.validation,
        profiles={t.name: load_profile(state, deps, t.name) for t in spec.tables},
        plan=state["plan"],
        source=None,
    )
