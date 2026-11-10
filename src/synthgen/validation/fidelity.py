"""Fidelity checks (SPEC 8.6). M2 has the row-count check; M6 adds the distribution checks."""

from __future__ import annotations

from synthgen.config import ValidationSettings
from synthgen.domain.report import CheckFamily, CheckResult, CheckStatus
from synthgen.engine.keys import expected_children
from synthgen.engine.runner import RunOptions
from synthgen.engine.writer import Manifest
from synthgen.spec.models import GenerationSpec


def check_row_counts(
    spec: GenerationSpec, manifest: Manifest, options: RunOptions, settings: ValidationSettings
) -> list[CheckResult]:
    """Each table's row count against the target the spec implies, within the tolerance."""
    actual = {t.name: t.rows for t in manifest.tables}
    results: list[CheckResult] = []
    for table in spec.tables:
        rule = table.row_rule
        if rule.kind == "fixed":
            cap = options.max_root_rows
            target = min(rule.rows, cap) if cap else rule.rows
        else:
            parents = actual.get(rule.parent_table, 0)
            target = expected_children(rule, parents, use_target=not options.is_sample)
        rows = actual.get(table.name, 0)
        error = abs(rows - target) / max(target, 1)
        results.append(
            CheckResult(
                family=CheckFamily.FIDELITY,
                check="row_count",
                table=table.name,
                metric="relative_error",
                value=error,
                threshold=settings.row_count_tolerance,
                status=CheckStatus.PASS
                if error <= settings.row_count_tolerance
                else CheckStatus.FAIL,
                detail=f"{rows} rows, target {target}",
            )
        )
    return results
