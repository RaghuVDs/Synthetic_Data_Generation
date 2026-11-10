from __future__ import annotations

from pathlib import Path

import pytest

from synthgen.seed.finance import build_finance
from synthgen.seed.orders import build_orders

SEED = 3
SCALE = 0.02


@pytest.fixture(scope="session")
def seeded_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Small copies of both seed sources, built once per test session."""
    data_dir = tmp_path_factory.mktemp("data")
    build_finance(data_dir / "finance.db", SEED, SCALE)
    build_orders(data_dir / "orders.db", SEED, SCALE)
    return data_dir
