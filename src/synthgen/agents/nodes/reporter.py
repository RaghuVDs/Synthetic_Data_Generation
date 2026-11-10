"""Node 15, LLM: a short summary. The model names metrics; code inserts every number."""

from __future__ import annotations

import json

from synthgen.agents.meaning import PLACEHOLDER, check_summary
from synthgen.agents.nodes.common import ask
from synthgen.agents.nodes.deps import NodeDeps, StateUpdate
from synthgen.agents.state import JobState
from synthgen.domain.feasibility import FeasibilityStatus
from synthgen.domain.job import ArtifactKind
from synthgen.domain.summary import Summary
from synthgen.engine.writer import Manifest, read_manifest

ROLE = "reporter"


def metrics(state: JobState, manifest: Manifest) -> dict[str, str]:
    """Every number the summary may mention, keyed by placeholder name."""
    report = state["final_report"]
    values: dict[str, str] = {
        "total_rows": f"{sum(t.rows for t in manifest.tables):,}",
        "table_count": str(len(manifest.tables)),
        "checks_run": str(len(report.results)) if report else "0",
        "checks_failed": str(len(report.failures())) if report else "0",
    }
    for table in manifest.tables:
        values[f"rows_{table.name}"] = f"{table.rows:,}"
        for count in table.scenario_counts:
            values[f"scenario_rows_{table.name}_{count.scenario}"] = f"{count.rows:,}"
    return values


def fill(text: str, values: dict[str, str]) -> str:
    """Replace every {placeholder}; the meaning gate has already rejected unknown names."""
    return PLACEHOLDER.sub(lambda m: values[m.group(1)], text)


def run(state: JobState, deps: NodeDeps) -> StateUpdate:
    """Write the summary with real numbers in place of placeholders."""
    values = metrics(state, read_manifest(deps.job_dir(state["job_id"])))
    intent = state["intent"]
    draft = ask(
        state,
        deps,
        ROLE,
        {
            "METRICS": json.dumps(sorted(values)),
            "INTENT": intent.model_dump_json() if intent else "none",
        },
        Summary,
        meaning=lambda value: check_summary(value, set(values)),
    )
    skipped = [
        f"Scenario {f.scenario} was not generated: {f.reason}"
        for f in state["feasibility"]
        if f.status is FeasibilityStatus.NOT_EXPRESSIBLE
    ]
    summary = Summary(
        text=fill(draft.text, values), caveats=[*(fill(c, values) for c in draft.caveats), *skipped]
    )
    deps.artifacts.put(state["job_id"], ArtifactKind.SUMMARY, summary)
    return {}
