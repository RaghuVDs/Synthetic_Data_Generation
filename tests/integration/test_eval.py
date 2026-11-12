"""`eval` end to end on the stub: pauses are answered, jobs are scored, the report is saved."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from typer.testing import CliRunner

from synthgen.app import Application
from synthgen.cli import app as cli
from synthgen.domain.job import JobStatus
from synthgen.evaluation.scoring import JobScore

from .helpers import fixture_set, skeleton

EnvFactory = Callable[..., dict[str, str]]
AppFactory = Callable[[dict[str, str]], Application]


def _request(answer: str | None, test_type: str) -> dict[str, Any]:
    return {
        "source": "finance",
        "tables": ["customers", "accounts"],
        "target_table": "accounts",
        "request": "customers with unusually high balances",
        "answer": answer,
        "expected": {
            "test_type": test_type,
            "include_invalid": False,
            "must_have": [
                {
                    "what": "high balances",
                    "categories": ["target_scenario"],
                    "tables": ["accounts"],
                    "columns": ["balance"],
                }
            ],
        },
    }


def _golden(root: Path) -> Path:
    (root / "requests").mkdir(parents=True)
    (root / "roles").mkdir()
    requests = {
        "answered": _request("p99", "rule_testing"),
        "unanswered": _request(None, "ui_testing"),
    }
    for name, body in requests.items():
        (root / "requests" / f"{name}.yaml").write_text(yaml.safe_dump(body))
    roles = {"accounts": {"balance": ["measure"], "status": ["status"]}}
    (root / "roles" / "finance.yaml").write_text(yaml.safe_dump(roles))
    return root


def _env(job_env: EnvFactory, tmp_path: Path) -> dict[str, str]:
    blocking = skeleton("TestIntent")
    blocking["ambiguities"] = [
        {"question": "Which balance counts as high?", "options": ["p95", "p99"], "blocking": True}
    ]
    fixtures = fixture_set(
        tmp_path / "fx",
        TestIntent=[(None, blocking), ("ANSWER: p99", skeleton("TestIntent"))],
    )
    return job_env(fixtures, SYNTHGEN_EVALUATION__GOLDEN_DIR=str(_golden(tmp_path / "golden")))


def test_eval_drives_scores_and_saves(
    make_app: AppFactory, job_env: EnvFactory, tmp_path: Path
) -> None:
    seen: list[JobScore] = []
    report, path = make_app(_env(job_env, tmp_path)).run_eval([100], [], [], seen.append)
    answered, unanswered = report.scores
    assert [s.name for s in seen] == ["answered", "unanswered"]
    assert answered.status is JobStatus.COMPLETED
    assert (answered.intent_correct, answered.plan_coverage) == (True, 1.0)
    assert answered.role_accuracy is not None
    assert (answered.first_pass_valid, answered.dry_run_passed, answered.final_passed) == (
        True,
        True,
        True,
    )
    assert unanswered.status is JobStatus.NEEDS_REVIEW  # the default answer resolves nothing
    assert (unanswered.intent_correct, unanswered.plan_coverage) == (False, None)
    assert report.summary.completed == 1 and report.by_volume[100].jobs == 2
    assert json.loads(path.read_text())["summary"]["jobs"] == 2


def test_eval_command_prints_metrics(job_env: EnvFactory, tmp_path: Path) -> None:
    env = {**_env(job_env, tmp_path), "COLUMNS": "200"}
    result = CliRunner().invoke(cli, ["eval", "--volumes", "100", "--only", "answered"], env=env)
    assert result.exit_code == 0, result.output
    assert "answered @ 100 rows: completed" in result.output
    assert "plan coverage" in result.output and "100%" in result.output
    assert CliRunner().invoke(cli, ["eval", "--volumes", "lots"], env=env).exit_code != 0
