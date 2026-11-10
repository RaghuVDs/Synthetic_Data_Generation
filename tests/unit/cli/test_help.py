from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from synthgen.cli import app

ROOT = Path(__file__).resolve().parents[3]


def test_help_exits_zero_and_describes_tool() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Synthetic Data Agent" in result.output


def test_main_py_help_runs_as_script() -> None:
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, str(ROOT / "main.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Synthetic Data Agent" in completed.stdout
