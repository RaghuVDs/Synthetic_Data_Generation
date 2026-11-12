from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
from pydantic import TypeAdapter, ValidationError

from synthgen.engine.runner import RunOptions, generate_dataset
from synthgen.engine.writer import part_path
from synthgen.spec.dependencies import own_reads, parent_reads
from synthgen.spec.evaluator import to_internal
from synthgen.spec.expressions import ExprType
from synthgen.spec.generators import Generator
from synthgen.spec.models import ColumnSpec, GenerationSpec
from synthgen.spec.validator import validate_spec

from .test_generators import draw
from .test_runner import GENERATION, make_spec

DAY = 86_400.0
LAGS = [float(q) for q in np.linspace(0, 30 * DAY, 11)]  # uniform on 0 to 30 days


def _gen(
    after: dict[str, Any] | None, granularity: str = "date", end: str = "2024-12-31"
) -> dict[str, Any]:
    start = "2024-01-01" if granularity == "date" else "2024-01-01 00:00:00"
    return {
        "type": "datetime",
        "start": start,
        "end": end if granularity == "date" else f"{end} 23:59:59",
        "granularity": granularity,
        "weekday_weights": None,
        "hour_weights": None,
        "after": after,
    }


def _base(texts: Sequence[str | None], kind: ExprType = ExprType.DATE) -> dict[str, Any]:
    return {"parent.opened": (kind, np.array(texts, dtype=object))}


def _seconds(values: Any, kind: ExprType = ExprType.DATE) -> np.ndarray[Any, Any]:
    return to_internal(np.asarray(values, dtype=object), kind)


def test_lagged_dates_follow_the_lag_quantiles() -> None:
    n = 20_000
    base = ["2024-03-01"] * n
    values = draw(_gen({"column": "parent.opened", "lag_quantiles": LAGS}), n=n, frame=_base(base))
    lag = (_seconds(values) - _seconds(base)) / DAY
    assert lag.min() >= 0 and lag.max() <= 30
    assert np.median(lag) == pytest.approx(15, abs=1)


def test_lags_are_truncated_at_the_end_not_piled_up() -> None:
    n = 20_000
    values = draw(
        _gen({"column": "parent.opened", "lag_quantiles": LAGS}),
        n=n,
        frame=_base(["2024-12-21"] * n),
    )
    days = _seconds(values) / DAY
    end = _seconds(["2024-12-31"])[0] / DAY
    assert np.max(days) <= end
    assert np.mean(days == end) < 0.2  # an end-clipped draw would put two thirds here


def test_datetimes_keep_hour_mix_and_stay_after_the_earlier_value() -> None:
    hours = [0.0] * 24
    hours[20] = 1.0
    gen = _gen({"column": "parent.opened", "lag_quantiles": LAGS}, "datetime") | {
        "hour_weights": hours
    }
    base = ["2024-03-01 10:30:00"] * 5_000
    values = draw(gen, n=5_000, frame=_base(base, ExprType.DATETIME))
    seconds = _seconds(values, ExprType.DATETIME)
    assert (seconds >= _seconds(base, ExprType.DATETIME)).all()
    assert np.mean(np.floor((seconds % DAY) / 3_600) == 20) > 0.95


def test_missing_earlier_dates_draw_from_the_range() -> None:
    values = draw(
        _gen({"column": "parent.opened", "lag_quantiles": LAGS}),
        n=4,
        frame=_base(["2024-03-01", None, None, "2024-03-02"]),
    )
    assert all(isinstance(v, str) and v.startswith("2024-") for v in values)


@pytest.mark.parametrize("lags", [[-1.0, 5.0], [5.0, 1.0], [1.0]])
def test_bad_lags_are_rejected(lags: list[float]) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(Generator).validate_python(
            _gen({"column": "parent.opened", "lag_quantiles": lags})
        )


def test_lagged_generator_reads_its_column() -> None:
    gen = TypeAdapter(Generator).validate_python(
        _gen({"column": "customers.joined", "lag_quantiles": LAGS})
    )
    assert parent_reads(gen) == {"customers.joined"}
    own = TypeAdapter(Generator).validate_python(_gen({"column": "joined", "lag_quantiles": LAGS}))
    assert own_reads(own) == {"joined"}


def _dated_spec(after: str) -> GenerationSpec:
    spec = make_spec(customers=300)
    lag = {"column": after, "lag_quantiles": LAGS}
    customers, orders = spec.tables
    joined, placed = (
        ColumnSpec.model_validate(
            {
                "name": name,
                "type": "date",
                "nullable": False,
                "null_rate": 0.0,
                "primary_key": False,
                "unique": False,
                "sensitive": False,
                "generator": _gen(after),
            }
        )
        for name, after in (("joined", None), ("placed", lag))
    )
    tables = [
        customers.model_copy(update={"columns": [*customers.columns, joined]}),
        orders.model_copy(update={"columns": [*orders.columns, placed]}),
    ]
    return spec.model_copy(update={"tables": tables})


@pytest.mark.parametrize(
    ("after", "fragment"),
    [("customers.code", "can only follow"), ("placed", "can only follow"), ("ghost", "follow")],
)
def test_lag_must_follow_another_date(after: str, fragment: str) -> None:
    messages = [e.message for e in validate_spec(_dated_spec(after), GENERATION)]
    assert any(fragment in m for m in messages), messages


def test_children_never_precede_their_parent(tmp_path: Path) -> None:
    spec = _dated_spec("customers.joined")
    assert validate_spec(spec, GENERATION) == []
    generate_dataset(spec, tmp_path, RunOptions(), GENERATION)
    customers = pq.read_table(part_path(tmp_path, "customers", 0)).to_pydict()
    orders = pq.read_table(part_path(tmp_path, "orders", 0)).to_pydict()
    joined = dict(zip(customers["id"], customers["joined"], strict=True))
    pairs = zip(orders["customer_id"], orders["placed"], strict=True)
    assert all(placed >= joined[c] for c, placed in pairs)
