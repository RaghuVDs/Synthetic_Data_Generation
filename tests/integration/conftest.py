from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from synthgen.seed.finance import build_finance
from synthgen.seed.orders import build_orders

from .helpers import FIXTURES

SEED = 3
SCALE = 0.02


@pytest.fixture(scope="session")
def seeded_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Small copies of both seed sources, built once per test session."""
    data_dir = tmp_path_factory.mktemp("data")
    build_finance(data_dir / "finance.db", SEED, SCALE)
    build_orders(data_dir / "orders.db", SEED, SCALE)
    return data_dir


@pytest.fixture
def job_env(seeded_dir: Path, tmp_path: Path) -> Callable[..., dict[str, str]]:
    """Environment for an isolated job run: its own data, output and state folders."""
    data = tmp_path / "data"
    data.mkdir()
    for name in ("finance.db", "orders.db"):
        shutil.copy(seeded_dir / name, data / name)

    def make(fixtures: Path = FIXTURES / "skeleton", **extra: str) -> dict[str, str]:
        return {
            "SYNTHGEN_PATHS__DATA_DIR": str(data),
            "SYNTHGEN_PATHS__OUTPUT_DIR": str(tmp_path / "output"),
            "LLM_PROVIDER": "stub",
            "LLM_STUB_FIXTURES": str(fixtures),
            **extra,
        }

    return make
