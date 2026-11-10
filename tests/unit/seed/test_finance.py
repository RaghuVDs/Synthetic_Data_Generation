from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from synthgen.seed.finance import build_finance

SEED = 11
SCALE = 0.05


def luhn_valid(number: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(number)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, int]]:
    path = tmp_path_factory.mktemp("seed") / "finance.db"
    return path, build_finance(path, SEED, SCALE)


@pytest.fixture(scope="module")
def conn(built: tuple[Path, dict[str, int]]) -> sqlite3.Connection:
    return sqlite3.connect(built[0])


def scalar(conn: sqlite3.Connection, sql: str) -> float:
    return float(conn.execute(sql).fetchone()[0])


def test_row_counts_follow_scale(built: tuple[Path, dict[str, int]]) -> None:
    counts = built[1]
    assert counts["customers"] == 1_000
    assert counts["merchants"] == 200
    assert counts["transactions"] == 10_000
    assert 1_300 < counts["accounts"] < 1_700
    assert counts["cards"] > 1_000


def test_every_foreign_key_resolves(conn: sqlite3.Connection) -> None:
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_fraud_rate_is_half_a_percent(conn: sqlite3.Connection) -> None:
    assert scalar(conn, "SELECT AVG(is_fraud) FROM transactions") == pytest.approx(0.005)


def test_fraud_rows_look_different(conn: sqlite3.Connection) -> None:
    fraud = scalar(conn, "SELECT AVG(amount) FROM transactions WHERE is_fraud = 1")
    normal = scalar(conn, "SELECT AVG(amount) FROM transactions WHERE is_fraud = 0")
    assert fraud > 2 * normal


@pytest.mark.parametrize(
    "violations",
    [
        "SELECT COUNT(*) FROM accounts a JOIN customers c USING (customer_id)"
        " WHERE a.opened_date < c.signup_date",
        "SELECT COUNT(*) FROM cards c JOIN accounts a USING (account_id)"
        " WHERE c.issued_date < a.opened_date",
        "SELECT COUNT(*) FROM cards WHERE expiry_date <= issued_date",
        "SELECT COUNT(*) FROM transactions t JOIN cards c USING (card_id)"
        " WHERE t.txn_timestamp < c.issued_date",
        "SELECT COUNT(*) FROM customers WHERE date_of_birth >= signup_date",
    ],
)
def test_date_rules_hold(conn: sqlite3.Connection, violations: str) -> None:
    assert scalar(conn, violations) == 0


def test_card_numbers_are_unique_luhn_valid(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT card_number, network FROM cards").fetchall()
    assert len({n for n, _ in rows}) == len(rows)
    assert all(luhn_valid(n) for n, _ in rows)
    assert all(len(n) == (15 if net == "amex" else 16) for n, net in rows)


def test_nulls_where_expected(conn: sqlite3.Connection) -> None:
    assert 0.03 < scalar(conn, "SELECT AVG(phone IS NULL) FROM customers") < 0.15
    assert (
        scalar(
            conn,
            "SELECT COUNT(*) FROM accounts WHERE account_type = 'credit' AND credit_limit IS NULL",
        )
        == 0
    )
    assert (
        scalar(
            conn,
            "SELECT COUNT(*) FROM accounts WHERE account_type <> 'credit'"
            " AND credit_limit IS NOT NULL",
        )
        == 0
    )


def test_transactions_are_in_time_order(conn: sqlite3.Connection) -> None:
    stamps = [r[0] for r in conn.execute("SELECT txn_timestamp FROM transactions ORDER BY 1")]
    in_id_order = [
        r[0] for r in conn.execute("SELECT txn_timestamp FROM transactions ORDER BY transaction_id")
    ]
    assert stamps == in_id_order


def _dump(path: Path) -> list[str]:
    conn = sqlite3.connect(path)
    try:
        return list(conn.iterdump())
    finally:
        conn.close()


def test_same_seed_same_data_other_seed_differs(
    built: tuple[Path, dict[str, int]], tmp_path: Path
) -> None:
    again = tmp_path / "again.db"
    other = tmp_path / "other.db"
    build_finance(again, SEED, SCALE)
    build_finance(other, SEED + 1, SCALE)
    assert _dump(again) == _dump(built[0])
    assert _dump(other) != _dump(built[0])
