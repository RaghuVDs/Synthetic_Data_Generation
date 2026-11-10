"""Helpers for integration tests: stub fixture sets and settings paths."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "llm"
SETTINGS_FILE = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


def write_fixture(
    folder: Path, schema: str, entries: list[tuple[str | None, dict[str, Any]]]
) -> None:
    """Record stub responses for one schema: (match text, response) pairs."""
    folder.mkdir(parents=True, exist_ok=True)
    payload = [{"match": m, "response": r} for m, r in entries]
    (folder / f"{schema}.json").write_text(json.dumps(payload), encoding="utf-8")


def skeleton(schema: str) -> dict[str, Any]:
    """The skeleton fixture's single response for a schema."""
    entries = json.loads((FIXTURES / "skeleton" / f"{schema}.json").read_text(encoding="utf-8"))
    return dict(entries[0]["response"])


def fixture_set(folder: Path, **overrides: list[tuple[str | None, dict[str, Any]]]) -> Path:
    """The skeleton fixtures with some schemas replaced."""
    folder.mkdir(parents=True, exist_ok=True)
    for path in (FIXTURES / "skeleton").glob("*.json"):
        shutil.copy(path, folder / path.name)
    for schema, entries in overrides.items():
        write_fixture(folder, schema, entries)
    return folder
