"""Enforce the dependency rules from CLAUDE.md and the layering decided in PLAN.md (Q2)."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "synthgen"
MAIN = ROOT / "main.py"

# A package may import only packages of strictly lower rank, or itself.
RANKS: dict[str, int] = {
    "domain": 0,
    "config": 1,
    "metadata": 1,
    "connectors": 1,
    "storage": 1,
    "seed": 1,
    "profiling": 2,
    "scenarios": 3,
    "spec": 4,
    "engine": 5,
    "validation": 6,
    "agents": 7,
    "app": 8,
    "cli": 9,
}
LLM_LIBS = {"langgraph", "langchain", "langchain_core", "openai", "groq"}
FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__"}
FORBIDDEN_MODULES = {"importlib"}


@dataclass(frozen=True)
class Violation:
    """One broken rule in one file."""

    file: str
    message: str


def _package_of(path: Path) -> str | None:
    """Top-level synthgen package a source file belongs to, or None for the root init."""
    parts = path.relative_to(SRC).with_suffix("").parts
    return None if parts == ("__init__",) else parts[0]


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC.parent).with_suffix("")
    parts = rel.parts[:-1] if rel.name == "__init__" else rel.parts
    return ".".join(parts)


def _imports(tree: ast.AST, module: str, is_init: bool) -> list[str]:
    """Absolute module names imported by a parsed file."""
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                found.append(node.module or "")
            else:
                base = module.split(".")
                drop = node.level - 1 if is_init else node.level
                anchor = base[: len(base) - drop]
                found.append(".".join([*anchor, *([node.module] if node.module else [])]))
    return found


def check_source(source: str, package: str | None, module: str, is_init: bool) -> list[str]:
    """Return rule violations for one file's source code."""
    tree = ast.parse(source)
    problems: list[str] = []
    for name in _imports(tree, module, is_init):
        top = name.split(".")[0]
        if top in LLM_LIBS and package != "agents":
            problems.append(f"imports LLM library {name!r} outside agents")
        if top in FORBIDDEN_MODULES:
            problems.append(f"imports forbidden module {name!r}")
        if top == "synthgen" and package is not None:
            parts = name.split(".")
            target = parts[1] if len(parts) > 1 else None
            if target is None:
                problems.append("imports the synthgen root package")
            elif target == "agents" and package not in {"agents", "app"}:
                problems.append(f"only app may import agents, found {name!r}")
            elif target != package and RANKS.get(target, 99) >= RANKS[package]:
                problems.append(f"{package} must not import {name!r}")
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in FORBIDDEN_CALLS
        ):
            problems.append(f"calls forbidden builtin {node.func.id!r}")
    return problems


def _source_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def test_every_package_has_a_rank() -> None:
    packages = {_package_of(p) for p in _source_files()} - {None}
    assert packages <= set(RANKS), f"unranked packages: {packages - set(RANKS)}"


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_source_file_respects_rules(path: Path) -> None:
    problems = check_source(
        path.read_text(encoding="utf-8"),
        _package_of(path),
        _module_name(path),
        path.name == "__init__.py",
    )
    assert problems == []


def test_main_only_imports_cli() -> None:
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    project_imports = [
        name for name in _imports(tree, "main", False) if name.split(".")[0] == "synthgen"
    ]
    assert project_imports
    assert all(name.startswith("synthgen.cli") for name in project_imports)


@pytest.mark.parametrize(
    ("source", "package", "expected"),
    [
        ("import synthgen.spec", "domain", "domain must not import 'synthgen.spec'"),
        ("from synthgen.agents import graph", "cli", "only app may import agents"),
        ("from synthgen.engine import x", "spec", "spec must not import 'synthgen.engine'"),
        ("from synthgen.storage import x", "metadata", "metadata must not import"),
        ("import langgraph", "engine", "imports LLM library 'langgraph' outside agents"),
        ("import importlib", "agents", "imports forbidden module 'importlib'"),
        ("eval('1')", "agents", "calls forbidden builtin 'eval'"),
        ("exec('x = 1')", "spec", "calls forbidden builtin 'exec'"),
    ],
)
def test_checker_catches_violations(source: str, package: str, expected: str) -> None:
    problems = check_source(source, package, f"synthgen.{package}.mod", is_init=False)
    assert any(expected in p for p in problems), problems


@pytest.mark.parametrize(
    ("source", "package"),
    [
        ("from synthgen.domain import errors", "spec"),
        ("from synthgen.agents import graph", "app"),
        ("import langgraph", "agents"),
        ("from . import sibling", "engine"),
        ("import re\nre.compile('x')", "profiling"),
    ],
)
def test_checker_allows_valid_imports(source: str, package: str) -> None:
    assert check_source(source, package, f"synthgen.{package}.mod", is_init=False) == []


def test_relative_import_resolves_to_owner_package() -> None:
    problems = check_source("from ..engine import runner", "spec", "synthgen.spec.mod", False)
    assert problems == ["spec must not import 'synthgen.engine'"]
