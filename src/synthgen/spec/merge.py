"""Merge scenario overlays into the baseline (SPEC 8.4 layers).

An overlay replaces only the generators it lists, and only for its own rows; every other column
keeps its baseline generator. The engine applies overrides, so merging attaches the overlays.
"""

from __future__ import annotations

from collections.abc import Sequence

from synthgen.spec.models import GenerationSpec, ScenarioSpec


def merge_overlays(baseline: GenerationSpec, overlays: Sequence[ScenarioSpec]) -> GenerationSpec:
    """The baseline with these overlays as its scenarios; earlier scenarios are replaced."""
    return baseline.model_copy(update={"scenarios": list(overlays)})
