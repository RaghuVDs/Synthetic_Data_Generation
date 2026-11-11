"""Gate 3: check that an LLM answer makes sense against metadata and the plan (SPEC 4.3).

Each check returns plain-language problems; an empty list passes. They are sent back to the
model as a numbered list.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence, Set

from synthgen.domain.intent import TestIntent
from synthgen.domain.patch import SpecPatch
from synthgen.domain.plan import ScenarioCategory, ScenarioPlan
from synthgen.domain.semantic import SemanticMap
from synthgen.domain.summary import Summary
from synthgen.spec.models import GenerationSpec, OverlaySet
from synthgen.spec.patch import disallowed_paths, try_apply

Columns = Mapping[str, Sequence[str]]
"""Table name -> its column names, in order."""
PLACEHOLDER = re.compile(r"\{([a-z0-9_]+)\}")
DIGIT = re.compile(r"\d")
TYPICAL_IS_REMAINDER = "typical data is the remainder after all shares; do not plan it"
LABELS_FOLLOW_DATA = "change the data columns that make the label true"


def _missing(columns: Columns, table: str, column: str | None) -> str | None:
    if table not in columns:
        return f"unknown table {table!r}"
    if column is not None and column not in columns[table]:
        return f"unknown column {table}.{column}"
    return None


def check_intent(intent: TestIntent, max_rows: int) -> list[str]:
    """A volume hint must fit the per-table row limit."""
    if intent.volume_hint is not None and intent.volume_hint > max_rows:
        return [f"volume_hint {intent.volume_hint} exceeds the limit of {max_rows} rows"]
    return []


def check_semantic_map(
    semantic: SemanticMap, columns: Columns, key_columns: Columns | None = None
) -> list[str]:
    """Every table once, every given column of it once, and keys that exist.

    `columns` are the columns this answer must map (a slice of a wide table); `key_columns`, when
    given, are all of each table's columns, since a key may sit in another slice.
    """
    keys = key_columns or columns
    problems: list[str] = []
    seen = [t.table for t in semantic.tables]
    problems += [f"table {t} is not mapped" for t in columns if t not in seen]
    problems += [f"table {t} is mapped more than once" for t in set(seen) if seen.count(t) > 1]
    for table in semantic.tables:
        if table.table not in columns:
            problems.append(f"unknown table {table.table!r}")
            continue
        named = [c.name for c in table.columns]
        known = columns[table.table]
        problems += [f"column {table.table}.{c} is not mapped" for c in known if c not in named]
        problems += [f"unknown column {table.table}.{c}" for c in named if c not in known]
        for key in (table.entity_key, table.event_time, table.label_column):
            if key is not None and key not in keys.get(table.table, known):
                problems.append(f"unknown key column {table.table}.{key}")
    every = {c for cols in keys.values() for c in cols}
    for rule in semantic.business_rules:
        problems += [
            f"business rule names unknown column {c!r}"
            for c in rule.columns
            if c.split(".")[-1] not in every
        ]
    return problems


def check_plan(
    plan: ScenarioPlan, columns: Columns, template_ids: Set[str], labels: Set[tuple[str, str]]
) -> list[str]:
    """Real tables and columns; no typical scenarios; every scenario changes some data column."""
    problems: list[str] = []
    if not plan.scenarios:
        problems.append("the plan has no scenarios")
    for s in plan.scenarios:
        if s.category is ScenarioCategory.TYPICAL:
            problems.append(
                f"{s.name}: typical data is the remainder after all shares; do not plan it"
            )
        if s.levers and all((lv.table, lv.column) in labels for lv in s.levers):
            problems.append(f"{s.name}: only label columns change; {LABELS_FOLLOW_DATA}")
        problems += [f"{s.name}: {p}" for t in s.tables if (p := _missing(columns, t, None))]
        problems += [
            f"{s.name}: {p}" for lv in s.levers if (p := _missing(columns, lv.table, lv.column))
        ]
        problems += [
            f"{s.name}: {p}" for a in s.signature if (p := _missing(columns, a.table, a.column))
        ]
        if s.template_ref is not None and s.template_ref not in template_ids:
            problems.append(f"{s.name}: unknown template {s.template_ref!r}")
    return problems


def check_overlays(overlays: OverlaySet, expected: Sequence[str], columns: Columns) -> list[str]:
    """Exactly one overlay per approved scenario, each naming real tables and columns."""
    names = [o.name for o in overlays.scenarios]
    problems = [f"missing overlay for scenario {n!r}" for n in expected if n not in names]
    problems += [f"overlay {n!r} is not an approved scenario" for n in names if n not in expected]
    problems += [f"overlay {n!r} appears more than once" for n in set(names) if names.count(n) > 1]
    for o in overlays.scenarios:
        problems += [f"{o.name}: {p}" for p in [_missing(columns, o.entity, None)] if p]
        problems += [
            f"{o.name}: {p}" for lb in o.labels if (p := _missing(columns, lb.table, lb.column))
        ]
        problems += [
            f"{o.name}: {p}" for ov in o.overrides if (p := _missing(columns, ov.table, ov.column))
        ]
    return problems


def check_patch(patch: SpecPatch, spec: GenerationSpec) -> list[str]:
    """Only allow-listed paths, and a patch that applies to the current spec (SPEC 5)."""
    if patch.plan_change_required:
        return []
    if not patch.patches:
        return ["no patches given; set plan_change_required if the plan itself must change"]
    problems = [
        f"path {p} is not patchable; only generator parameters may change,"
        " otherwise set plan_change_required"
        for p in disallowed_paths(patch)
    ]
    return problems or try_apply(spec, patch)


def check_summary(summary: Summary, metric_names: Set[str]) -> list[str]:
    """Numbers only through known {placeholders}, never typed by the model."""
    problems: list[str] = []
    for text in (summary.text, *summary.caveats):
        problems += [
            f"unknown metric {{{n}}}" for n in PLACEHOLDER.findall(text) if n not in metric_names
        ]
        if DIGIT.search(PLACEHOLDER.sub("", text)):
            problems.append(f"write numbers only as {{placeholders}}: {text!r}")
    return problems
