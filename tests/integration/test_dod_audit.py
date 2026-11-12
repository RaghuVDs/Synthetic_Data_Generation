"""Definition-of-done audit (T7.5): sensitive source values and a plan with no templates.

1. The recorded live finance job is scanned. No value of any sensitive source column appears
   in any LLM prompt or response, except words that also occur in a public column (fake
   merchant names are built from surnames, e.g. "Bailey-Welch"). No identifier-like sensitive
   value (email, phone, card number) appears anywhere in the output. Names and birth dates
   come from small fake vocabularies and match by chance (PLAN Q7), so output is not scanned
   for them.
2. An orders job runs with an empty template folder and completes. The answers come from the
   stub here; the plan's sensibility was judged live at M3 (PLAN progress log).
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

from synthgen.app import Application
from synthgen.domain.job import Approval, ApprovalDecision, ArtifactKind, JobStatus, UserRequest
from synthgen.domain.plan import ScenarioPlan

from .helpers import FIXTURES, fixture_set, skeleton
from .test_recorded_live import REQUEST as FRAUD

EnvFactory = Callable[..., dict[str, str]]
AppFactory = Callable[[dict[str, str]], Application]
ROOT = Path(__file__).resolve().parents[2]
SENSITIVE = {
    "customers": ["first_name", "last_name", "email", "phone", "date_of_birth"],
    "cards": ["card_number"],
}
IDENTIFIERS = {("customers", "email"), ("customers", "phone"), ("cards", "card_number")}
PUBLIC_TEXT = {"merchants": ["name", "city"], "customers": ["country", "segment"]}
WORD = r"[\w@.+\-]+"


def _source_values(db: Path) -> dict[tuple[str, str], set[str]]:
    with closing(sqlite3.connect(db)) as conn:
        return {
            (table, column): {
                str(v) for (v,) in conn.execute(f"SELECT DISTINCT {column} FROM {table}") if v
            }
            for table, columns in SENSITIVE.items()
            for column in columns
        }


def _public_words(db: Path) -> set[str]:
    with closing(sqlite3.connect(db)) as conn:
        return {
            word
            for table, columns in PUBLIC_TEXT.items()
            for column in columns
            for (v,) in conn.execute(f"SELECT DISTINCT {column} FROM {table}")
            for word in re.findall(WORD, str(v))
        }


def test_sensitive_source_values_never_reach_prompts_or_output(
    make_app: AppFactory, job_env: EnvFactory
) -> None:
    env = job_env(FIXTURES / "live_finance", SYNTHGEN_AGENT__CAPS__PLANNER_TOOL_CALLS="1")
    app = make_app(env)
    pending = app.start_job(FRAUD.model_copy(update={"auto_approve": False}))
    approval = Approval(
        decision=ApprovalDecision.EDITED,
        removed_scenarios=("volume_skew_transactions_merchant",),
        decided_at=datetime.now(UTC),
    )
    job_id = app.decide_plan(pending.job.job_id, approval).job.job_id
    assert app.get_status(job_id).status is JobStatus.COMPLETED
    db = Path(env["SYNTHGEN_PATHS__DATA_DIR"]) / "finance.db"
    values = _source_values(db)

    words = set().union(*values.values()) - _public_words(db)
    texts = [t for c in app._llm_log.for_job(job_id) for t in (c.prompt, c.response or "")]
    tokens = {w for text in texts for w in re.findall(WORD, text)}
    assert not words & tokens, sorted(words & tokens)[:5]

    data = Path(env["SYNTHGEN_PATHS__OUTPUT_DIR"]) / "jobs" / job_id / "data"
    identifiers = set().union(*(values[k] for k in IDENTIFIERS))
    cells = {
        str(v)
        for path in data.rglob("*.parquet")
        for column in pq.read_table(path).columns
        for v in column.to_pylist()
        if isinstance(v, str)
    }
    assert not identifiers & cells, sorted(identifiers & cells)[:5]


def _orders_answers(folder: Path) -> Path:
    roles = yaml.safe_load((ROOT / "tests" / "golden" / "roles" / "orders.yaml").read_text())
    shape = {
        "customers": ("entity", None, None, None),
        "orders": ("event", "customer_id", "order_ts", None),
        "shipments": ("event", "order_id", "shipped_date", "is_late"),
    }
    semantic = {
        "tables": [
            {
                "table": table,
                "grain": grain,
                "entity_key": key,
                "event_time": time,
                "label_column": label,
                "columns": [
                    {"name": c, "role": r[0], "note": None} for c, r in roles[table].items()
                ],
            }
            for table, (grain, key, time, label) in shape.items()
        ],
        "business_rules": [],
    }
    intent = skeleton("TestIntent") | {
        "domain": "online retail",
        "objective": "Train a late-delivery model",
        "test_type": "model_validation",
        "entities_of_interest": ["orders", "shipments"],
        "must_include": ["shipments delivered late"],
    }
    late: dict[str, Any] = {
        "name": "late_deliveries",
        "category": "target_scenario",
        "description": "Orders whose shipment arrives after the promised date",
        "rationale": "The model must learn what late deliveries look like",
        "share": 0.1,
        "unit": "entity",
        "tables": ["orders", "shipments"],
        "primitives": ["timing_shift"],
        "levers": [
            {
                "table": "shipments",
                "column": "delivered_date",
                "primitive": "timing_shift",
                "change": "delivered three days after the promised date",
            }
        ],
        "signature": [
            {
                "table": "shipments",
                "column": "is_late",
                "metric": "value",
                "operator": "eq",
                "value": 1.0,
                "window_hours": None,
            }
        ],
        "template_ref": None,
    }
    overlay = {
        "name": "late_deliveries",
        "category": "target_scenario",
        "share": 0.1,
        "unit": "entity",
        "entity": "orders",
        "labels": [{"table": "shipments", "column": "is_late", "value": 1}],
        "overrides": [
            {
                "table": "shipments",
                "column": "delivered_date",
                "generator": {
                    "type": "derived",
                    "expression": "date_add(max(shipped_date, promised_date), 3, 'days')",
                },
            }
        ],
        "correlations": [],
        "cardinality": [],
        "sequences": [],
        "defects": [],
    }
    summary = {
        "text": "Generated {total_rows} rows across {table_count} tables.",
        "caveats": [],
    }
    return fixture_set(
        folder,
        TestIntent=[(None, intent)],
        SemanticMap=[(None, semantic)],
        ScenarioPlan=[(None, {"scenarios": [late], "assumptions": []})],
        OverlaySet=[(None, {"scenarios": [overlay]})],
        Summary=[(None, summary)],
    )


def test_orders_plan_with_no_templates_installed(
    make_app: AppFactory, job_env: EnvFactory, tmp_path: Path
) -> None:
    empty = tmp_path / "no_templates"
    empty.mkdir()
    env = job_env(
        _orders_answers(tmp_path / "fx"),
        SYNTHGEN_PATHS__TEMPLATES_DIR=str(empty),
        SYNTHGEN_AGENT__CAPS__PLANNER_TOOL_CALLS="0",
    )
    app = make_app(env)
    request = UserRequest(
        source="orders",
        tables=("customers", "orders", "shipments"),
        text="Late deliveries for our delivery-time model",
        auto_approve=True,
    )
    view = app.start_job(request)
    assert view.job.status is JobStatus.COMPLETED, view.job.error
    plan = app._artifacts.get(view.job.job_id, ArtifactKind.PLAN, ScenarioPlan)
    assert [s.template_ref for s in plan.scenarios] == [None]
    report = app.get_report(view.job.job_id)
    assert report.final_report is not None and report.final_report.passed
