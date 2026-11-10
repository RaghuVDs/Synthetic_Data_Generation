"""Apply a SpecPatch: replace values at JSON pointers (RFC 6901), then re-validate the spec.

M3 restricts patches to an allow-list of paths; M2 applies any path that exists.
"""

from __future__ import annotations

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
