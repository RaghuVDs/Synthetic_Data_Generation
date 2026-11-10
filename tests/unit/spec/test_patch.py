from __future__ import annotations

import pytest

from synthgen.domain.errors import PatchError
from synthgen.domain.patch import SpecPatch
from synthgen.spec.patch import apply_patch, set_pointer

from ..engine.test_runner import BIG, make_spec


def patch(*ops: tuple[str, str]) -> SpecPatch:
    return SpecPatch.model_validate(
        {
            "patches": [{"path": p, "value_json": v, "reason": "r"} for p, v in ops],
            "plan_change_required": False,
            "explanation": "e",
        }
    )


def test_set_pointer_handles_escapes_and_lists() -> None:
    doc = {"a/b": [1, {"c~d": 2}]}
    set_pointer(doc, "/a~1b/1/c~0d", 3)
    assert doc == {"a/b": [1, {"c~d": 3}]}


@pytest.mark.parametrize("pointer", ["/missing", "/a~1b/9", "/a~1b/x", "/a~1b/0/deeper"])
def test_set_pointer_rejects_missing_paths(pointer: str) -> None:
    with pytest.raises(PatchError):
        set_pointer({"a/b": [1]}, pointer, 0)


def test_apply_patch_changes_scenario_share() -> None:
    spec = make_spec(scenarios=[BIG])
    patched = apply_patch(spec, patch(("/scenarios/0/share", "0.2")))
    assert patched.scenarios[0].share == 0.2
    assert spec.scenarios[0].share == 0.1


def test_apply_patch_rejects_invalid_result() -> None:
    with pytest.raises(PatchError):
        apply_patch(make_spec(scenarios=[BIG]), patch(("/scenarios/0/share", "7")))
