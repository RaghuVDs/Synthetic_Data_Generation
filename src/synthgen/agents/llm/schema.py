"""Strict-mode JSON schemas for structured output (SPEC 4.2).

Pydantic's schema is reduced to what strict mode accepts: every property required, closed
objects, `anyOf` instead of `oneOf`, enums instead of consts, and no validation keywords the
provider does not document. Pydantic still enforces those constraints at the parsing gate.
"""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel

from synthgen.domain.errors import ConfigError

DROPPED_KEYWORDS = frozenset(
    {
        "title",
        "default",
        "discriminator",
        "pattern",
        "format",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minItems",
        "maxItems",
    }
)
Json = Any  # Any: a JSON schema fragment


def strict_schema(model: type[BaseModel]) -> dict[str, Json]:
    """The model's JSON schema rewritten for strict mode; fails if the model cannot comply."""
    schema = _rewrite(copy.deepcopy(model.model_json_schema()))
    _check(schema, model.__name__)
    return dict(schema)


def _rewrite(node: Json) -> Json:
    if isinstance(node, list):
        return [_rewrite(item) for item in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Json] = {}
    for key, value in node.items():
        if key in DROPPED_KEYWORDS:
            continue
        if key == "oneOf":
            out["anyOf"] = _rewrite(value)
        elif key == "const":
            out["enum"] = [value]
        elif key in ("properties", "$defs"):
            out[key] = {name: _rewrite(sub) for name, sub in value.items()}
        else:
            out[key] = _rewrite(value)
    return out


def _check(node: Json, where: str) -> None:
    if isinstance(node, list):
        for item in node:
            _check(item, where)
        return
    if not isinstance(node, dict):
        return
    if node.get("type") == "object" or "properties" in node:
        properties = node.get("properties", {})
        if node.get("additionalProperties") is not False:
            raise ConfigError(f"{where}: an object allows extra properties")
        if set(node.get("required", [])) != set(properties):
            raise ConfigError(f"{where}: not every property is required")
    for value in node.values():
        _check(value, where)
