"""Merchants and transactions for the fake finance source, including the `is_fraud` label."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
from faker import Faker

from synthgen.seed import common as c

TXN_START = "2024-01-01"
TXN_END = "2025-12-31"
FRAUD_RATE = 0.005
FRAUD_DECLINE_RATE = 0.3
NIGHT_END_HOUR = 6
ATM_NOTE = 20.0
WEEKDAY_WEIGHTS = [0.9, 0.95, 1.0, 1.05, 1.3, 1.35, 0.8]  # Monday .. Sunday
HOUR_WEIGHTS = [
    0.2, 0.1, 0.1, 0.1, 0.1, 0.3, 0.8, 1.5, 2.2, 2.5, 2.8, 3.2,
    4.0, 3.8, 3.0, 2.8, 3.0, 3.6, 4.2, 4.0, 3.2, 2.2, 1.2, 0.6,
]  # fmt: skip
CURRENCY = {"US": "USD", "GB": "GBP", "DE": "EUR", "FR": "EUR", "IT": "EUR"}


@dataclass(frozen=True)
class Category:
    """A merchant category: MCC, popularity, log-normal amount and share of high-risk merchants."""

    mcc: str
    weight: float
    log_mean: float
    log_sd: float
    high_risk: float


CATEGORIES = {
    "grocery": Category("5411", 25, 3.8, 0.6, 0.02),
    "restaurants": Category("5812", 20, 3.4, 0.5, 0.02),
    "fuel": Category("5541", 12, 3.9, 0.4, 0.03),
    "clothing": Category("5651", 10, 4.2, 0.7, 0.04),
    "online_services": Category("5817", 10, 2.8, 0.9, 0.15),
    "pharmacy": Category("5912", 8, 3.2, 0.6, 0.02),
    "electronics": Category("5732", 6, 5.3, 0.9, 0.15),
    "entertainment": Category("7832", 5, 3.3, 0.5, 0.03),
    "travel": Category("4511", 4, 5.8, 0.8, 0.15),
}


@dataclass(frozen=True)
class MerchantFacts:
    """What transactions need to know about merchants."""

    ids: c.IntArray
    category: c.StrArray
    country: list[str]
    high_risk: c.BoolArray


def build_merchants(
    conn: sqlite3.Connection,
    rng: np.random.Generator,
    faker: Faker,
    n: int,
    cities: list[tuple[str, str]],
) -> MerchantFacts:
    """Insert merchants; categories decide MCC, amount scale and risk."""
    names = list(CATEGORIES)
    category = c.choice(rng, names, [CATEGORIES[k].weight for k in names], n)
    high_share = np.array([CATEGORIES[k].high_risk for k in category.tolist()])
    roll = rng.random(n)
    risk = np.where(roll < high_share, "high", np.where(roll < high_share + 0.15, "medium", "low"))
    city_index = c.zipf_indices(rng, len(cities), n, exponent=0.9)
    countries = [cities[i][1] for i in city_index]
    c.insert(
        conn,
        "merchants",
        {
            "merchant_id": np.arange(1, n + 1),
            "name": [faker.company() for _ in range(n)],
            "category": category,
            "mcc": [CATEGORIES[k].mcc for k in category.tolist()],
            "city": [cities[i][0] for i in city_index],
            "country": countries,
            "risk_level": risk,
        },
    )
    return MerchantFacts(
        ids=np.arange(1, n + 1, dtype=np.int64),
        category=category,
        country=countries,
        high_risk=risk == "high",
    )


def build_transactions(
    conn: sqlite3.Connection,
    rng: np.random.Generator,
    cards: tuple[c.IntArray, c.FloatArray],
    merchants: MerchantFacts,
    n: int,
) -> int:
    """Insert transactions ordered by time; label about 0.5% of them as fraud."""
    card_ids, issued = cards
    card = c.zipf_indices(rng, card_ids.shape[0], n, exponent=0.7)
    merchant = c.zipf_indices(rng, merchants.ids.shape[0], n, exponent=0.9)
    when = _times(rng, issued[card])
    category = merchants.category[merchant]
    mean = np.array([CATEGORIES[k].log_mean for k in category.tolist()])
    sd = np.array([CATEGORIES[k].log_sd for k in category.tolist()])
    log_amount = rng.normal(mean, sd)
    channel = _channels(rng, category)
    night = (when % c.SECONDS_PER_DAY) // 3600 < NIGHT_END_HOUR
    fraud = _pick_fraud(
        rng,
        (channel == "online") * 1.3
        + night * 1.1
        + (log_amount - mean) / sd * 0.9
        + merchants.high_risk[merchant] * 1.2,
    )
    amount = np.exp(log_amount) * np.where(fraud, rng.uniform(1.5, 3.5, n), 1.0)
    amount = np.where(
        channel == "atm", np.maximum(ATM_NOTE, np.round(amount / ATM_NOTE) * ATM_NOTE), amount
    )
    status = c.choice(rng, ["approved", "declined", "reversed"], [96, 3.5, 0.5], n)
    status = np.where(fraud & (rng.random(n) < FRAUD_DECLINE_RATE), "declined", status)
    order = np.argsort(when, kind="stable")
    return c.insert(
        conn,
        "transactions",
        {
            "transaction_id": np.arange(1, n + 1),
            "card_id": card_ids[card][order],
            "merchant_id": merchants.ids[merchant][order],
            "amount": np.round(amount, 2)[order],
            "currency": [CURRENCY[merchants.country[m]] for m in merchant[order].tolist()],
            "txn_timestamp": c.iso_timestamps(when[order]),
            "channel": channel[order],
            "status": status[order],
            "is_fraud": fraud[order].astype(np.int64),
        },
    )


def _times(rng: np.random.Generator, issued: c.FloatArray) -> c.FloatArray:
    """Transaction times inside the history window, never before the card was issued."""
    low = np.maximum(issued, c.epoch(TXN_START))
    high = np.full(issued.shape[0], c.epoch(TXN_END) + c.SECONDS_PER_DAY - 1)
    return c.weighted_times(rng, low, high, WEEKDAY_WEIGHTS, HOUR_WEIGHTS)


def _channels(rng: np.random.Generator, category: c.StrArray) -> c.StrArray:
    """Online services are always online; other merchants mix point-of-sale, online and ATM."""
    mixed = c.choice(rng, ["pos", "online", "atm"], [72, 22, 6], category.shape[0])
    return np.where(category == "online_services", "online", mixed)


def _pick_fraud(rng: np.random.Generator, logit: c.FloatArray) -> c.BoolArray:
    """Exactly round(FRAUD_RATE * n) fraud rows, chosen with weight exp(logit) (Gumbel top-k)."""
    k = max(1, round(FRAUD_RATE * logit.shape[0]))
    keys = logit + rng.gumbel(size=logit.shape[0])
    chosen = np.argpartition(-keys, k - 1)[:k]
    fraud = np.zeros(logit.shape[0], dtype=bool)
    fraud[chosen] = True
    return fraud
