"""SpecPatch: targeted spec changes proposed by the repair_analyst (SPEC 5)."""

from __future__ import annotations

import json

from pydantic import Field, field_validator

from synthgen.domain.base import StrictModel


class PatchOp(StrictModel):
    """Replace the value at a JSON pointer. The value is JSON text so strict mode can carry it."""

    path: str = Field(pattern=r"^(/[^/]*)+$")
    value_json: str
    reason: str = Field(min_length=1)

    @field_validator("value_json")
    @classmethod
    def _must_parse(cls, text: str) -> str:
        json.loads(text)
        return text

    @property
    def value(self) -> object:
        """The decoded replacement value."""
        return json.loads(self.value_json)


class SpecPatch(StrictModel):
    """Patches to apply, or a request to go back to the user with a changed plan."""

    patches: list[PatchOp]
    plan_change_required: bool
    explanation: str = Field(min_length=1)
