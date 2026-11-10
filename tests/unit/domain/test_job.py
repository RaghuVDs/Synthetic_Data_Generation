from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from synthgen.domain.job import (
    Approval,
    ApprovalDecision,
    JobStatus,
    LoopCounters,
    RowTarget,
    ShareEdit,
    UserRequest,
)

NOW = datetime(2026, 11, 10, tzinfo=UTC)


def _request(**overrides: object) -> UserRequest:
    fields: dict[str, object] = {
        "source": "finance",
        "tables": ("customers", "accounts"),
        "text": "test churn",
    }
    fields.update(overrides)
    return UserRequest.model_validate(fields)


def test_terminal_statuses() -> None:
    terminal = {s for s in JobStatus if s.is_terminal}
    assert terminal == {
        JobStatus.COMPLETED,
        JobStatus.NEEDS_REVIEW,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    }
    assert len(JobStatus) == 15


def test_request_defaults_and_round_trip() -> None:
    request = _request(row_targets=[{"table": "customers", "rows": 100}])
    assert request.seed == 0
    assert request.workers == 1
    assert UserRequest.model_validate_json(request.model_dump_json()) == request


@pytest.mark.parametrize(
    "overrides",
    [
        {"tables": ()},
        {"tables": ("a", "a")},
        {"text": ""},
        {"source": "bad name"},
        {"source": "../etc"},
        {"seed": -1},
        {"workers": 0},
        {"row_targets": [{"table": "a", "rows": 1}, {"table": "a", "rows": 2}]},
        {"unexpected": True},
    ],
)
def test_request_rejects_invalid(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _request(**overrides)


def test_row_target_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        RowTarget(table="customers", rows=0)


def test_counters_start_at_zero_and_reject_negative() -> None:
    assert LoopCounters() == LoopCounters(clarify=0, plan_rounds=0, spec_repair=0)
    with pytest.raises(ValidationError):
        LoopCounters(clarify=-1)


def test_models_are_frozen() -> None:
    counters = LoopCounters()
    with pytest.raises(ValidationError):
        counters.clarify = 1  # type: ignore[misc]


def test_edited_approval_needs_edits() -> None:
    with pytest.raises(ValidationError):
        Approval(decision=ApprovalDecision.EDITED, decided_at=NOW)
    edited = Approval(
        decision=ApprovalDecision.EDITED,
        share_edits=(ShareEdit(scenario="s1", share=0.1),),
        decided_at=NOW,
    )
    assert edited.share_edits[0].share == 0.1


def test_only_edited_approval_carries_edits() -> None:
    with pytest.raises(ValidationError):
        Approval(decision=ApprovalDecision.APPROVED, removed_scenarios=("s1",), decided_at=NOW)


@pytest.mark.parametrize(("feedback", "expected"), [(None, False), ("  ", False), ("more", True)])
def test_rejection_feedback(feedback: str | None, expected: bool) -> None:
    approval = Approval(decision=ApprovalDecision.REJECTED, feedback=feedback, decided_at=NOW)
    assert approval.has_feedback is expected
