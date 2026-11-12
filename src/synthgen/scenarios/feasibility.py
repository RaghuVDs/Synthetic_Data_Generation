"""Feasibility check, run by code after the planner (SPEC 7.4)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from synthgen.domain.feasibility import FeasibilityResult, FeasibilityStatus
from synthgen.domain.plan import Lever, PlannedScenario, Primitive, ScenarioPlan
from synthgen.domain.semantic import ColumnRole, SemanticMap
from synthgen.scenarios.primitives import RULES, Construct, available_constructs

Columns = Mapping[str, Sequence[str]]


def check_feasibility(
    plan: ScenarioPlan, semantic: SemanticMap, columns: Columns
) -> list[FeasibilityResult]:
    """One result per scenario, in plan order."""
    roles = {(t.table, c.name): c.role for t in semantic.tables for c in t.columns}
    return [_scenario(s, roles, columns) for s in plan.scenarios]


def _lever_problem(
    lever: Lever, roles: Mapping[tuple[str, str], ColumnRole], columns: Columns
) -> str | None:
    if lever.table not in columns or lever.column not in columns[lever.table]:
        return f"{lever.table}.{lever.column} does not exist"
    if roles.get((lever.table, lever.column)) is ColumnRole.LABEL:
        return None  # the overlay writes label values directly
    rule = RULES[lever.primitive]
    if rule.construct not in available_constructs():
        return _unavailable(lever.primitive)
    role = roles.get((lever.table, lever.column))
    if role is not None and role not in rule.roles:
        return f"{lever.primitive} does not apply to {lever.table}.{lever.column} ({role})"
    return None


def _scenario(
    scenario: PlannedScenario, roles: Mapping[tuple[str, str], ColumnRole], columns: Columns
) -> FeasibilityResult:
    covered: list[str] = []
    not_covered: list[str] = []
    reasons: list[str] = []
    for lever in scenario.levers:
        problem = _lever_problem(lever, roles, columns)
        is_label = roles.get((lever.table, lever.column)) is ColumnRole.LABEL
        label = (
            f"label {lever.table}.{lever.column} set by the overlay"
            if is_label
            else f"{lever.primitive} on {lever.table}.{lever.column}"
        )
        (not_covered if problem else covered).append(label)
        if problem:
            reasons.append(problem)
    unlevered = [
        p for p in scenario.primitives if p not in {lv.primitive for lv in scenario.levers}
    ]
    available = available_constructs()
    for primitive in unlevered:
        if RULES[primitive].construct not in available:
            not_covered.append(str(primitive))
            reasons.append(_unavailable(primitive))
    if not scenario.levers and not unlevered:
        reasons.append("the scenario changes no column")
    status = _status(covered, not_covered or reasons)
    return FeasibilityResult(
        scenario=scenario.name,
        status=status,
        covered=covered,
        not_covered=not_covered,
        reason="; ".join(reasons) or None,
        alternative=_alternative(scenario, status, roles),
    )


def _unavailable(primitive: Primitive) -> str:
    return (
        f"{primitive} needs a {RULES[primitive].construct}, which the generator cannot express yet"
    )


def _status(covered: list[str], problems: Sequence[str]) -> FeasibilityStatus:
    if not problems:
        return FeasibilityStatus.EXPRESSIBLE
    return FeasibilityStatus.PARTIAL if covered else FeasibilityStatus.NOT_EXPRESSIBLE


def _alternative(
    scenario: PlannedScenario,
    status: FeasibilityStatus,
    roles: Mapping[tuple[str, str], ColumnRole],
) -> str | None:
    """The nearest primitive that would work: one the engine has, that suits the lever's column."""
    if status is FeasibilityStatus.EXPRESSIBLE:
        return None
    available = available_constructs()
    for primitive in scenario.primitives:
        levered = [lv for lv in scenario.levers if lv.primitive == primitive]
        role = roles.get((levered[0].table, levered[0].column)) if levered else None
        step: Primitive | None = primitive
        while step is not None and not _works(step, role, available):
            step = RULES[step].alternative
        if step is not None and step != primitive:
            return f"use {step} instead of {primitive}"
    return None


def _works(primitive: Primitive, role: ColumnRole | None, available: frozenset[Construct]) -> bool:
    rule = RULES[primitive]
    return rule.construct in available and (role is None or role in rule.roles)
