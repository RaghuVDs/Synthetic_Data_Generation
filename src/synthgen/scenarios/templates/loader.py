"""Scenario template library (SPEC 7.5). Optional: the system works with zero templates."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field, ValidationError

from synthgen.domain.base import StrictModel
from synthgen.domain.errors import MetadataError
from synthgen.domain.plan import AssertionMetric, AssertionOperator, Primitive, ScenarioCategory
from synthgen.domain.semantic import ColumnRole

BUILT_IN = Path(__file__).parent


class TemplateLever(StrictModel):
    """A change the template makes, on whichever column has this role."""

    role: ColumnRole
    primitive: Primitive
    change: str


class TemplateAssertion(StrictModel):
    """A signature assertion stated on a role; the planner quantifies it from the profile."""

    role: ColumnRole
    metric: AssertionMetric
    operator: AssertionOperator
    value_hint: str


class ScenarioTemplate(StrictModel):
    """One reusable scenario recipe."""

    id: str
    domain: str
    name: str
    version: int
    category: ScenarioCategory
    description: str
    required_roles: list[ColumnRole]
    primitives: list[Primitive]
    levers: list[TemplateLever]
    signature: list[TemplateAssertion]
    parameters: dict[str, float | str] = Field(default_factory=dict)
    default_share: float = Field(ge=0.0, le=1.0)


class TemplateSummary(StrictModel):
    """What a search returns: enough to decide whether to fetch the template."""

    id: str
    domain: str
    name: str
    category: ScenarioCategory
    description: str


class TemplateLibrary:
    """Every template under one folder, loaded once."""

    def __init__(self, folder: Path) -> None:
        """Load and validate every `*.yaml` under `folder`; a missing folder means no templates."""
        self._templates: dict[str, ScenarioTemplate] = {}
        for path in sorted(folder.rglob("*.yaml")) if folder.is_dir() else []:
            try:
                template = ScenarioTemplate.model_validate(yaml.safe_load(path.read_text("utf-8")))
            except (OSError, yaml.YAMLError, ValidationError) as exc:
                raise MetadataError(f"invalid scenario template {path}: {exc}") from exc
            self._templates[template.id] = template

    @property
    def ids(self) -> frozenset[str]:
        """IDs of every loaded template."""
        return frozenset(self._templates)

    def search(self, domain: str, keywords: list[str]) -> list[TemplateSummary]:
        """Templates ranked by how many of the words appear in their domain, name or description."""
        words = [w.lower() for w in [domain, *keywords] if w.strip()]
        scored = []
        for t in self._templates.values():
            text = f"{t.domain} {t.name} {t.description}".lower()
            score = sum(w in text for w in words)
            if score:
                scored.append((-score, t.id, t))
        return [
            TemplateSummary(
                id=t.id,
                domain=t.domain,
                name=t.name,
                category=t.category,
                description=t.description,
            )
            for _, _, t in sorted(scored)
        ]

    def get(self, template_id: str) -> ScenarioTemplate | None:
        """One template, or None when no template has this ID."""
        return self._templates.get(template_id)
