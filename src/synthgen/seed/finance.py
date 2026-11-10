"""Fake `finance.db`: customers, accounts, cards, merchants and transactions (SPEC 11).

All values are invented. The shape is realistic on purpose so the profiler has something to
find: log-normal incomes and amounts, Zipf-skewed keys, correlated columns, nulls, ordered dates
(signup <= opened <= issued <= transaction) and an `is_fraud` label near 0.5%.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from faker import Faker

from synthgen.domain.errors import SeedError
from synthgen.seed import common as c
from synthgen.seed.finance_activity import build_merchants, build_transactions

SCHEMA = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    phone TEXT,
    date_of_birth TEXT NOT NULL,
    city TEXT NOT NULL,
    country TEXT NOT NULL,
    segment TEXT NOT NULL,
    signup_date TEXT NOT NULL,
    annual_income REAL,
    credit_score INTEGER NOT NULL
);
CREATE TABLE accounts (
    account_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    account_type TEXT NOT NULL,
    opened_date TEXT NOT NULL,
    status TEXT NOT NULL,
    balance REAL NOT NULL,
    credit_limit REAL
);
CREATE TABLE cards (
    card_id INTEGER PRIMARY KEY,
    account_id INTEGER NOT NULL REFERENCES accounts(account_id),
    card_number TEXT NOT NULL UNIQUE,
    network TEXT NOT NULL,
    card_type TEXT NOT NULL,
    issued_date TEXT NOT NULL,
    expiry_date TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE merchants (
    merchant_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    mcc TEXT NOT NULL,
    city TEXT NOT NULL,
    country TEXT NOT NULL,
    risk_level TEXT NOT NULL
);
CREATE TABLE transactions (
    transaction_id INTEGER PRIMARY KEY,
    card_id INTEGER NOT NULL REFERENCES cards(card_id),
    merchant_id INTEGER NOT NULL REFERENCES merchants(merchant_id),
    amount REAL NOT NULL,
    currency TEXT NOT NULL,
    txn_timestamp TEXT NOT NULL,
    channel TEXT NOT NULL,
    status TEXT NOT NULL,
    is_fraud INTEGER NOT NULL
);
"""

BASE_CUSTOMERS = 20_000
BASE_MERCHANTS = 4_000
BASE_TRANSACTIONS = 200_000
HISTORY_START = "2018-01-01"
SIGNUP_END = "2024-12-31"
LAST_ISSUE = "2025-06-30"
EXTRA_ACCOUNTS_MEAN = 0.5
EXTRA_CARDS_MEAN = 0.15
SAVINGS_CARD_RATE = 0.4
PHONE_NULL_RATE = 0.08
INCOME_NULL_RATE = 0.12
CARD_NETWORKS: dict[str, tuple[list[list[int]], int]] = {
    "visa": ([[4]], 16),
    "mastercard": ([[5, 1], [5, 2], [5, 3], [5, 4], [5, 5]], 16),
    "amex": ([[3, 4], [3, 7]], 15),
}
"""Issuer prefixes (as digits) and total length per network."""
CITIES = [
    ("Avalon", "US"), ("Brookfield", "US"), ("Cedar Falls", "US"), ("Dunmore", "US"),
    ("Eastport", "US"), ("Fairview", "US"), ("Glenwood", "US"), ("Harborview", "US"),
    ("Ironwood", "US"), ("Juniper", "US"), ("Kingsbridge", "GB"), ("Lindholm", "GB"),
    ("Marlow", "GB"), ("Northam", "GB"), ("Oberwald", "DE"), ("Petersau", "DE"),
    ("Quellheim", "DE"), ("Rivoli", "FR"), ("Saint-Aubin", "FR"), ("Tresca", "IT"),
]  # fmt: skip


@dataclass(frozen=True)
class CustomerFacts:
    """What later tables need to know about customers."""

    ids: c.IntArray
    signup: c.FloatArray
    income: c.FloatArray


@dataclass(frozen=True)
class AccountFacts:
    """What cards need to know about accounts."""

    ids: c.IntArray
    opened: c.FloatArray
    kind: c.StrArray


def build_finance(path: Path, seed: int, scale: float) -> dict[str, int]:
    """Write finance.db and return the row count of each table."""

    def fill(conn: sqlite3.Connection) -> dict[str, int]:
        rng = np.random.default_rng(seed)
        faker = Faker("en_US")
        faker.seed_instance(seed)
        counts: dict[str, int] = {}
        customers = _customers(conn, rng, faker, c.scaled(BASE_CUSTOMERS, scale), counts)
        accounts = _accounts(conn, rng, customers, counts)
        cards = _cards(conn, rng, accounts, counts)
        merchants = build_merchants(conn, rng, faker, c.scaled(BASE_MERCHANTS, scale), CITIES)
        counts["merchants"] = merchants.ids.shape[0]
        counts["transactions"] = build_transactions(
            conn, rng, cards, merchants, c.scaled(BASE_TRANSACTIONS, scale)
        )
        return counts

    return c.write_database(path, SCHEMA, fill)


def _customers(
    conn: sqlite3.Connection,
    rng: np.random.Generator,
    faker: Faker,
    n: int,
    counts: dict[str, int],
) -> CustomerFacts:
    ids = np.arange(1, n + 1, dtype=np.int64)
    first = np.asarray(c.name_pool(faker.first_name, 400))[rng.integers(0, 400, n)]
    last = np.asarray(c.name_pool(faker.last_name, 800))[rng.integers(0, 800, n)]
    domains = c.choice(rng, ["example.com", "example.org", "example.net"], [6, 3, 1], n)
    emails = [
        f"{f}.{s}.{i}@{d}".lower() for f, s, i, d in zip(first, last, ids, domains, strict=True)
    ]
    phones = [
        f"+1-555-{a:03d}-{b:04d}"
        for a, b in zip(rng.integers(0, 1000, n), rng.integers(0, 10_000, n), strict=True)
    ]
    city_index = c.zipf_indices(rng, len(CITIES), n, exponent=0.9)
    segment = c.choice(rng, ["retail", "premium", "business"], [75, 18, 7], n)
    start, end = c.epoch(HISTORY_START), c.epoch(SIGNUP_END)
    signup = start + (end - start) * rng.beta(2.0, 1.2, n)  # growth: more recent signups
    age_days = (18 + rng.gamma(4.0, 6.0, n).clip(0, 67)) * 365.25
    log_income = rng.normal(10.8, 0.5, n) + np.select(
        [segment == "premium", segment == "business"], [0.7, 1.1], 0.0
    )
    income = np.round(np.exp(log_income), -2)
    score = np.clip(np.round(650 + 55 * (log_income - 10.8) / 0.5 + rng.normal(0, 45, n)), 300, 850)
    counts["customers"] = c.insert(
        conn,
        "customers",
        {
            "customer_id": ids,
            "first_name": first,
            "last_name": last,
            "email": emails,
            "phone": c.with_nulls(rng, phones, PHONE_NULL_RATE),
            "date_of_birth": c.iso_dates(signup - age_days * c.SECONDS_PER_DAY),
            "city": [CITIES[i][0] for i in city_index],
            "country": [CITIES[i][1] for i in city_index],
            "segment": segment,
            "signup_date": c.iso_dates(signup),
            "annual_income": c.with_nulls(rng, income, INCOME_NULL_RATE),
            "credit_score": score.astype(np.int64),
        },
    )
    return CustomerFacts(ids=ids, signup=signup, income=income)


def _accounts(
    conn: sqlite3.Connection,
    rng: np.random.Generator,
    customers: CustomerFacts,
    counts: dict[str, int],
) -> AccountFacts:
    per_customer = 1 + rng.poisson(EXTRA_ACCOUNTS_MEAN, customers.ids.shape[0])
    owner = np.repeat(np.arange(customers.ids.shape[0]), per_customer)
    n = owner.shape[0]
    ids = np.arange(1, n + 1, dtype=np.int64)
    kind = c.choice(rng, ["checking", "savings", "credit"], [55, 30, 15], n)
    end = c.epoch(LAST_ISSUE)
    opened = np.minimum(customers.signup[owner] + rng.exponential(200, n) * c.SECONDS_PER_DAY, end)
    balance_mu = np.select([kind == "savings", kind == "credit"], [8.5, 6.5], 7.5)
    limit = np.round(customers.income[owner] * rng.uniform(0.1, 0.3, n), -2)
    counts["accounts"] = c.insert(
        conn,
        "accounts",
        {
            "account_id": ids,
            "customer_id": customers.ids[owner],
            "account_type": kind,
            "opened_date": c.iso_dates(opened),
            "status": c.choice(rng, ["active", "closed", "frozen"], [90, 8, 2], n),
            "balance": np.round(rng.lognormal(balance_mu, 1.0), 2),
            "credit_limit": [
                float(v) if k == "credit" else None for v, k in zip(limit, kind, strict=True)
            ],
        },
    )
    return AccountFacts(ids=ids, opened=opened, kind=kind)


def _cards(
    conn: sqlite3.Connection,
    rng: np.random.Generator,
    accounts: AccountFacts,
    counts: dict[str, int],
) -> tuple[c.IntArray, c.FloatArray]:
    kinds = accounts.kind
    has_card = (kinds != "savings") | (rng.random(kinds.shape[0]) < SAVINGS_CARD_RATE)
    per_account = has_card * (1 + rng.poisson(EXTRA_CARDS_MEAN, kinds.shape[0]))
    owner = np.repeat(np.arange(accounts.ids.shape[0]), per_account)
    n = owner.shape[0]
    network = c.choice(rng, ["visa", "mastercard", "amex"], [60, 30, 10], n)
    issued = np.minimum(
        accounts.opened[owner] + rng.exponential(30, n) * c.SECONDS_PER_DAY, c.epoch(LAST_ISSUE)
    )
    years = rng.choice([3, 4], n)
    counts["cards"] = c.insert(
        conn,
        "cards",
        {
            "card_id": np.arange(1, n + 1),
            "account_id": accounts.ids[owner],
            "card_number": _card_numbers(rng, network),
            "network": network,
            "card_type": np.where(kinds[owner] == "credit", "credit", "debit"),
            "issued_date": c.iso_dates(issued),
            "expiry_date": c.iso_dates(issued + years * 365.25 * c.SECONDS_PER_DAY),
            "status": c.choice(rng, ["active", "blocked", "expired"], [92, 3, 5], n),
        },
    )
    return np.arange(1, n + 1, dtype=np.int64), issued


def _card_numbers(rng: np.random.Generator, network: c.StrArray) -> list[str]:
    """Unique Luhn-valid numbers: visa 4..., mastercard 51-55..., amex 34/37 (15 digits)."""
    numbers = np.empty(network.shape[0], dtype=object)
    todo = np.arange(network.shape[0])
    for _ in range(c.MAX_REDRAWS):
        for net, (prefixes, length) in CARD_NETWORKS.items():
            rows = todo[network[todo] == net]
            if rows.size:
                prefix = np.asarray(prefixes)[rng.integers(0, len(prefixes), rows.size)]
                body = np.concatenate(
                    [prefix, rng.integers(0, 10, (rows.size, length - 1 - prefix.shape[1]))], axis=1
                )
                numbers[rows] = c.digits_to_text(c.luhn_complete(body.astype(np.int64)))
        _, first = np.unique(numbers.astype(str), return_index=True)
        todo = np.setdiff1d(np.arange(network.shape[0]), first)
        if not todo.size:
            return [str(n) for n in numbers]
    raise SeedError("could not draw unique card numbers")
