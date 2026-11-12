"""Eval metrics (SPEC 11): pure functions from what a job produced to its scores and totals.

Intent accuracy: test type and include-invalid both as expected. Role accuracy: labelled
columns of the job's tables mapped to an acceptable role. Plan coverage: must-have scenarios
the plan contains. First-pass spec validity: the first merged spec needed no repair. Repair
loops: spec repairs plus dry-run repairs. Dry-run pass rate: the first dry run passed.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import mean

from synthgen.domain.base import StrictModel
from synthgen.domain.intent import TestIntent
from synthgen.domain.job import JobStatus, LoopCounters
from synthgen.domain.plan import PlannedScenario, ScenarioPlan
from synthgen.domain.semantic import SemanticMap
from synthgen.evaluation.golden import GoldenRequest, MustHave, RoleLabels


class JobFacts(StrictModel):
    """What one eval job produced, as far as it got."""

    job_id: str
    status: JobStatus
    error: str | None
    intent: TestIntent | None
    semantic_map: SemanticMap | None
    plan: ScenarioPlan | None
    counters: LoopCounters
    has_spec: bool
    first_dry_run_passed: bool | None
    final_passed: bool | None
    tokens: int
    llm_latency_ms: int
    seconds: float


class JobScore(StrictModel):
    """One golden request at one volume: the job's outcome and its metrics."""

    name: str
    source: str
    volume: int
    job_id: str
    status: JobStatus
    error: str | None
    intent_correct: bool | None
    role_accuracy: float | None
    plan_coverage: float | None
    first_pass_valid: bool | None
    repair_loops: int
    dry_run_passed: bool | None
    final_passed: bool | None
    tokens: int
    llm_latency_ms: int
    seconds: float


class EvalSummary(StrictModel):
    """Averages over jobs; a metric no job reached is None."""

    jobs: int
    completed: int
    intent_accuracy: float | None
    role_accuracy: float | None
    plan_coverage: float | None
    first_pass_spec_validity: float | None
    repair_loops_per_job: float | None
    dry_run_pass_rate: float | None
    tokens_per_job: float | None
    llm_latency_ms_per_job: float | None
    seconds_per_job: float | None


def score(golden: GoldenRequest, volume: int, facts: JobFacts, roles: RoleLabels) -> JobScore:
    """Every metric of one job against its golden request."""
    intent = facts.intent
    expected = golden.expected
    return JobScore(
        name=golden.name,
        source=golden.source,
        volume=volume,
        job_id=facts.job_id,
        status=facts.status,
        error=facts.error,
        intent_correct=None
        if intent is None
        else intent.test_type is expected.test_type
        and intent.include_invalid == expected.include_invalid,
        role_accuracy=role_accuracy(facts.semantic_map, roles, golden.tables),
        plan_coverage=None if facts.plan is None else coverage(facts.plan, expected.must_have),
        first_pass_valid=facts.counters.spec_repair == 0 if facts.has_spec else None,
        repair_loops=facts.counters.spec_repair + facts.counters.validation_repair,
        dry_run_passed=facts.first_dry_run_passed,
        final_passed=facts.final_passed,
        tokens=facts.tokens,
        llm_latency_ms=facts.llm_latency_ms,
        seconds=facts.seconds,
    )


def role_accuracy(
    semantic: SemanticMap | None, roles: RoleLabels, tables: Sequence[str]
) -> float | None:
    """Share of labelled columns of `tables` whose mapped role is one of the labels."""
    if semantic is None:
        return None
    mapped = {(t.table, c.name): c.role for t in semantic.tables for c in t.columns}
    checks = [
        mapped.get((table, column)) in allowed
        for table in tables
        for column, allowed in roles.get(table, {}).items()
    ]
    return mean(checks) if checks else None


def coverage(plan: ScenarioPlan, must_have: Sequence[MustHave]) -> float:
    """Share of must-have scenarios that some planned scenario covers."""
    return mean(any(_covers(s, m) for s in plan.scenarios) for m in must_have)


def _covers(scenario: PlannedScenario, must: MustHave) -> bool:
    if scenario.category not in must.categories or not set(must.tables) <= set(scenario.tables):
        return False
    if not must.columns:
        return True
    tables = set(must.tables or scenario.tables)
    touched = {(lv.table, lv.column) for lv in scenario.levers}
    touched |= {(a.table, a.column) for a in scenario.signature}
    return any(t in tables and c in must.columns for t, c in touched)


def summarise(scores: Sequence[JobScore]) -> EvalSummary:
    """Averages of each metric over the jobs that reached it."""

    def average(values: Sequence[float | bool | None]) -> float | None:
        kept = [float(v) for v in values if v is not None]
        return mean(kept) if kept else None

    return EvalSummary(
        jobs=len(scores),
        completed=sum(s.status is JobStatus.COMPLETED for s in scores),
        intent_accuracy=average([s.intent_correct for s in scores]),
        role_accuracy=average([s.role_accuracy for s in scores]),
        plan_coverage=average([s.plan_coverage for s in scores]),
        first_pass_spec_validity=average([s.first_pass_valid for s in scores]),
        repair_loops_per_job=average([s.repair_loops for s in scores]),
        dry_run_pass_rate=average([s.dry_run_passed for s in scores]),
        tokens_per_job=average([s.tokens for s in scores]),
        llm_latency_ms_per_job=average([s.llm_latency_ms for s in scores]),
        seconds_per_job=average([s.seconds for s in scores]),
    )
