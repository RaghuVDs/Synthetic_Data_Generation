"""Run every validation family on an output folder and write `validation_report.json`."""

from __future__ import annotations

from pathlib import Path

from synthgen.config import ValidationSettings
from synthgen.domain.report import ValidationReport
from synthgen.engine.runner import RunOptions
from synthgen.engine.writer import read_manifest
from synthgen.spec.models import GenerationSpec
from synthgen.validation.fidelity import check_row_counts
from synthgen.validation.integrity import check_integrity

REPORT_FILE = "validation_report.json"


def validate_output(
    spec: GenerationSpec, out_dir: Path, options: RunOptions, settings: ValidationSettings
) -> ValidationReport:
    """Validate what the engine wrote to `out_dir` and save the report beside it."""
    manifest = read_manifest(out_dir)
    report = ValidationReport(
        results=(
            *check_integrity(spec, out_dir, manifest),
            *check_row_counts(spec, manifest, options, settings),
        )
    )
    (out_dir / REPORT_FILE).write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report
