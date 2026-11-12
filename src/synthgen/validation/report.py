"""Run every validation family on an output folder and write `validation_report.json`."""

from __future__ import annotations

from pathlib import Path

from synthgen.domain.report import CheckResult, ValidationReport
from synthgen.engine.writer import read_manifest
from synthgen.validation.fidelity import check_row_counts, column_checks
from synthgen.validation.fidelity_tables import children_check, spearman_check
from synthgen.validation.integrity import check_integrity
from synthgen.validation.output import OutputData, ValidationInputs

REPORT_FILE = "validation_report.json"


def validate_output(inputs: ValidationInputs, out_dir: Path) -> ValidationReport:
    """Validate what the engine wrote to `out_dir` and save the report beside it."""
    manifest = read_manifest(out_dir)
    data = OutputData(out_dir, manifest)
    spec = inputs.spec
    results: list[CheckResult] = [
        *check_integrity(spec, out_dir, manifest),
        *check_row_counts(spec, manifest, inputs.options, inputs.settings),
    ]
    for table in spec.tables:
        profile = inputs.profiles.get(table.name)
        if profile is None or data.entry(table.name) is None:
            continue
        results += column_checks(inputs, data, table, profile)
        results += spearman_check(inputs, data, table, profile)
        results += children_check(inputs, data, table, profile)
    report = ValidationReport(results=tuple(results))
    (out_dir / REPORT_FILE).write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report
