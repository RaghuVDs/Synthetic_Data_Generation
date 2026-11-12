"""The golden set (SPEC 11): requests, expected intents, must-have scenarios and role labels.

Layout and matching rules are in `tests/golden/README.md`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import Field

from synthgen.domain.base import StrictModel
from synthgen.domain.errors import ConfigError
from synthgen.domain.intent import TestType
from synthgen.domain.plan import ScenarioCategory
from synthgen.domain.semantic import ColumnRole

REQUESTS_DIR = "requests"
ROLES_DIR = "roles"


class MustHave(StrictModel):
    """A scenario the plan must contain, matched by category, tables and touched columns."""

    what: str = Field(min_length=1)
    categories: list[ScenarioCategory] = Field(min_length=1)
    tables: list[str]
    columns: list[str]


class ExpectedIntent(StrictModel):
    """What the intent analyst should read from the request."""

    test_type: TestType
    include_invalid: bool
    must_have: list[MustHave] = Field(min_length=1)


class GoldenRequest(StrictModel):
    """One golden request, named after its file."""

    name: str
    source: str
    tables: list[str] = Field(min_length=1)
    target_table: str
    request: str = Field(min_length=1)
    answer: str | None
    expected: ExpectedIntent


RoleLabels = dict[str, dict[str, list[ColumnRole]]]
"""table -> column -> roles a correct mapping may give."""


class GoldenSet(StrictModel):
    """Every golden request and the role labels of each source."""

    requests: list[GoldenRequest]
    roles: dict[str, RoleLabels]

    def select(self, names: Sequence[str], sources: Sequence[str]) -> GoldenSet:
        """Only the named requests and sources; an empty filter keeps everything."""
        unknown = set(names) - {r.name for r in self.requests}
        if unknown:
            raise ConfigError(f"no golden request named {sorted(unknown)[0]!r}")
        kept = [
            r
            for r in self.requests
            if (not names or r.name in names) and (not sources or r.source in sources)
        ]
        if not kept:
            raise ConfigError("no golden request matches the filter")
        return self.model_copy(update={"requests": kept})


def load_golden(root: Path) -> GoldenSet:
    """Read `requests/*.yaml` and `roles/*.yaml` under `root`; a bad file is a ConfigError."""
    requests = [
        GoldenRequest.model_validate({"name": path.stem, **_yaml(path)})
        for path in sorted((root / REQUESTS_DIR).glob("*.yaml"))
    ]
    if not requests:
        raise ConfigError(f"no golden requests under {root / REQUESTS_DIR}")
    roles = {path.stem: _yaml(path) for path in sorted((root / ROLES_DIR).glob("*.yaml"))}
    return GoldenSet.model_validate({"requests": requests, "roles": roles})


def _yaml(path: Path) -> dict[str, object]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} is not a YAML mapping")
    return loaded
