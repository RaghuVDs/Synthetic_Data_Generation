"""Apply a SpecPatch: replace values at JSON pointers (RFC 6901), then re-validate the spec.

Only allow-listed paths may be patched (SPEC 5); anything else needs a plan change.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from synthgen.domain.errors import PatchError
from synthgen.domain.patch import SpecPatch
from synthgen.spec.models import GenerationSpec

Json = Any  # Any: a decoded JSON document (dict, list or scalar)


def _tokens(pointer: str) -> list[str]:
    return [t.replace("~1", "/").replace("~0", "~") for t in pointer.split("/")[1:]]


def set_pointer(document: Json, pointer: str, value: Json) -> None:
    """Replace the existing value at `pointer` in place."""
    *parents, last = _tokens(pointer)
    node = document
    for token in parents:
        node = _child(node, token, pointer)
    if isinstance(node, dict):
        if last not in node:
            raise PatchError(f"{pointer}: no key {last!r}")
        node[last] = value
    elif isinstance(node, list):
        node[_index(node, last, pointer)] = value
    else:
        raise PatchError(f"{pointer}: cannot descend into a scalar")


def _child(node: Json, token: str, pointer: str) -> Json:
    if isinstance(node, dict):
        if token not in node:
            raise PatchError(f"{pointer}: no key {token!r}")
        return node[token]
    if isinstance(node, list):
        return node[_index(node, token, pointer)]
    raise PatchError(f"{pointer}: cannot descend into a scalar")


def _index(node: list[Json], token: str, pointer: str) -> int:
    if not token.isdigit() or int(token) >= len(node):
        raise PatchError(f"{pointer}: {token!r} is not an index of a list of {len(node)}")
    return int(token)


def apply_patch(spec: GenerationSpec, patch: SpecPatch) -> GenerationSpec:
    """The spec with every patch applied, validated again as a whole."""
    document = spec.model_dump(mode="json")
    for op in patch.patches:
        set_pointer(document, op.path, op.value)
    try:
        return GenerationSpec.model_validate(document)
    except ValidationError as exc:
        raise PatchError(f"patched spec is invalid: {exc}") from exc


_PARAMETER = r"generator/(?!type$)[^/]+(/.*)?$"
ALLOWED_PATHS = (
    re.compile(rf"^/scenarios/\d+/overrides/\d+/{_PARAMETER}"),
    re.compile(rf"^/tables/\d+/columns/\d+/{_PARAMETER}"),
    re.compile(r"^/scenarios/\d+/sequences/\d+/steps/\d+/(count|gap_min_seconds|gap_max_seconds)$"),
    re.compile(rf"^/scenarios/\d+/sequences/\d+/steps/\d+/columns/\d+/{_PARAMETER}"),
)
"""SPEC 5: generator parameters (clip bounds and conditional cases included) of overrides,
baseline columns and sequence steps, and sequence step counts and gaps; never a generator's
type, a share, a name or a table's structure."""


def disallowed_paths(patch: SpecPatch) -> list[str]:
    """Patch paths outside the allow-list."""
    return [op.path for op in patch.patches if not any(p.match(op.path) for p in ALLOWED_PATHS)]


def try_apply(spec: GenerationSpec, patch: SpecPatch) -> list[str]:
    """Why the patch cannot be applied; empty when it applies."""
    try:
        apply_patch(spec, patch)
    except PatchError as exc:
        return [str(exc)]
    return []
