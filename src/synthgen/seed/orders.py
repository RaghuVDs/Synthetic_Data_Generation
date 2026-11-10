"""Fake `orders.db`: customers, products, orders, order_items and shipments (SPEC 11).

All values are invented. Shape the profiler should find: Zipf-skewed customers and products,
log-normal prices by category, weekday and hour patterns, `line_total = quantity * unit_price`,
`total_amount` = sum of an order's lines, and ordered dates (order <= shipped <= delivered).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from faker import Faker

from synthgen.seed import common as c
from synthgen.seed.orders_fulfilment import ProductFacts, build_items, build_shipments

SCHEMA = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    city TEXT NOT NULL,
    country TEXT NOT NULL,
    signup_date TEXT NOT NULL,
    loyalty_tier TEXT NOT NULL,
    birth_year INTEGER
);
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    sku TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    unit_price REAL NOT NULL,
    weight_kg REAL NOT NULL,
    is_active INTEGER NOT NULL
);
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    order_ts TEXT NOT NULL,
    status TEXT NOT NULL,
    channel TEXT NOT NULL,
    discount_code TEXT,
    total_amount REAL NOT NULL
);
CREATE TABLE order_items (
    order_item_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(order_id),
    product_id INTEGER NOT NULL REFERENCES products(product_id),
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    line_total REAL NOT NULL
);
CREATE TABLE shipments (
    shipment_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL UNIQUE REFERENCES orders(order_id),
    carrier TEXT NOT NULL,
    shipped_date TEXT NOT NULL,
    promised_date TEXT NOT NULL,
    delivered_date TEXT,
    is_late INTEGER
);
"""

BASE_CUSTOMERS = 15_000
BASE_PRODUCTS = 2_000
BASE_ORDERS = 60_000
SIGNUP_START = "2019-01-01"
SIGNUP_END = "2024-12-31"
ORDER_START = "2023-01-01"
ORDER_END = "2025-12-31"
BIRTH_YEAR_NULL_RATE = 0.15
ACTIVE_PRODUCT_RATE = 0.9
STATUS_WEIGHTS = {"delivered": 78, "shipped": 6, "placed": 4, "cancelled": 7, "returned": 5}
DISCOUNT_CODES = {"SAVE10": 4, "WELCOME5": 3, "FREESHIP": 2, "VIP20": 1}
DISCOUNT_RATE = {"bronze": 0.12, "silver": 0.2, "gold": 0.35, "platinum": 0.5}
WEEKDAY_WEIGHTS = [1.0, 0.95, 0.95, 1.0, 1.1, 1.25, 1.35]  # Monday .. Sunday
HOUR_WEIGHTS = [
    0.3, 0.2, 0.1, 0.1, 0.1, 0.2, 0.5, 1.0, 1.5, 1.8, 2.0, 2.2,
    2.6, 2.4, 2.0, 2.0, 2.2, 2.6, 3.0, 3.6, 4.0, 3.6, 2.4, 1.0,
]  # fmt: skip
CITIES = [
    ("Ashby", "US"), ("Bramford", "US"), ("Coldwater", "US"), ("Dalton", "US"),
    ("Elmhurst", "US"), ("Foxley", "US"), ("Greyhaven", "US"), ("Hollins", "US"),
    ("Kelso", "GB"), ("Lowther", "GB"), ("Mirefield", "GB"), ("Nordheim", "DE"),
    ("Ostrava", "CZ"), ("Valmont", "FR"), ("Zelden", "NL"),
]  # fmt: skip


@dataclass(frozen=True)
class Category:
    """A product category: SKU code, popularity, log-normal price and gamma weight."""

    code: str
    weight: float
    log_price: float
    weight_shape: float
    weight_scale: float


CATEGORIES = {
    "electronics": Category("ELE", 15, 4.8, 2.0, 0.8),
    "home": Category("HOM", 18, 3.6, 2.0, 1.5),
    "toys": Category("TOY", 10, 3.0, 2.0, 0.4),
    "books": Category("BOK", 14, 2.7, 3.0, 0.15),
    "apparel": Category("APP", 16, 3.3, 2.0, 0.3),
    "beauty": Category("BEA", 9, 2.9, 2.0, 0.2),
    "sports": Category("SPO", 8, 3.7, 2.0, 1.2),
    "grocery": Category("GRO", 10, 2.2, 2.0, 0.6),
}


def build_orders(path: Path, seed: int, scale: float) -> dict[str, int]:
    """Write orders.db and return the row count of each table."""

    def fill(conn: sqlite3.Connection) -> dict[str, int]:
        rng = np.random.default_rng(seed)
        faker = Faker("en_US")
        faker.seed_instance(seed)
        counts: dict[str, int] = {}
        signup, tier = _customers(conn, rng, faker, c.scaled(BASE_CUSTOMERS, scale), counts)
        products = _products(conn, rng, faker, c.scaled(BASE_PRODUCTS, scale), counts)
        n_orders = c.scaled(BASE_ORDERS, scale)
        customer, when = _order_times(rng, signup, n_orders)
        discounted = rng.random(n_orders) < np.array([DISCOUNT_RATE[t] for t in tier[customer]])
        status = c.choice(rng, list(STATUS_WEIGHTS), list(STATUS_WEIGHTS.values()), n_orders)
        counts["order_items"], totals = build_items(conn, rng, products, discounted)
        counts["orders"] = c.insert(
            conn,
            "orders",
            {
                "order_id": np.arange(1, n_orders + 1),
                "customer_id": customer + 1,
                "order_ts": c.iso_timestamps(when),
                "status": status,
                "channel": c.choice(rng, ["web", "app", "store"], [55, 35, 10], n_orders),
                "discount_code": _discount_codes(rng, discounted),
                "total_amount": np.round(totals, 2),
            },
        )
        counts["shipments"] = build_shipments(conn, rng, when, status)
        return counts

    return c.write_database(path, SCHEMA, fill)


def _order_times(
    rng: np.random.Generator, signup: c.FloatArray, n: int
) -> tuple[c.IntArray, c.FloatArray]:
    """Zipf-skewed customers and order times after signup, sorted so IDs follow time."""
    customer = c.zipf_indices(rng, signup.shape[0], n, exponent=0.8)
    low = np.maximum(signup[customer], c.epoch(ORDER_START))
    high = np.full(n, c.epoch(ORDER_END) + c.SECONDS_PER_DAY - 1)
    when = c.weighted_times(rng, low, high, WEEKDAY_WEIGHTS, HOUR_WEIGHTS)
    order = np.argsort(when, kind="stable")
    return customer[order], when[order]


def _discount_codes(rng: np.random.Generator, discounted: c.BoolArray) -> list[str | None]:
    """A code for each discounted order, None otherwise."""
    codes = c.choice(rng, list(DISCOUNT_CODES), list(DISCOUNT_CODES.values()), discounted.shape[0])
    return [str(code) if d else None for code, d in zip(codes, discounted.tolist(), strict=True)]


def _customers(
    conn: sqlite3.Connection,
    rng: np.random.Generator,
    faker: Faker,
    n: int,
    counts: dict[str, int],
) -> tuple[c.FloatArray, c.StrArray]:
    first = np.asarray(c.name_pool(faker.first_name, 400))[rng.integers(0, 400, n)]
    last = np.asarray(c.name_pool(faker.last_name, 800))[rng.integers(0, 800, n)]
    ids = np.arange(1, n + 1)
    start, end = c.epoch(SIGNUP_START), c.epoch(SIGNUP_END)
    signup = start + (end - start) * rng.beta(2.0, 1.3, n)
    tier = c.choice(rng, ["bronze", "silver", "gold", "platinum"], [60, 25, 12, 3], n)
    city = c.zipf_indices(rng, len(CITIES), n, exponent=1.0)
    birth_year = np.clip(np.round(rng.normal(1985, 13, n)), 1940, 2006).astype(np.int64)
    counts["customers"] = c.insert(
        conn,
        "customers",
        {
            "customer_id": ids,
            "first_name": first,
            "last_name": last,
            "email": [
                f"{f}{s}{i}@shop.example".lower() for f, s, i in zip(first, last, ids, strict=True)
            ],
            "city": [CITIES[i][0] for i in city],
            "country": [CITIES[i][1] for i in city],
            "signup_date": c.iso_dates(signup),
            "loyalty_tier": tier,
            "birth_year": c.with_nulls(rng, birth_year, BIRTH_YEAR_NULL_RATE),
        },
    )
    return signup, tier


def _products(
    conn: sqlite3.Connection,
    rng: np.random.Generator,
    faker: Faker,
    n: int,
    counts: dict[str, int],
) -> ProductFacts:
    names = list(CATEGORIES)
    category = c.choice(rng, names, [CATEGORIES[k].weight for k in names], n)
    spec = [CATEGORIES[k] for k in category.tolist()]
    price = np.round(np.exp(rng.normal([s.log_price for s in spec], 0.5)), 2)
    weight = np.round(rng.gamma([s.weight_shape for s in spec], [s.weight_scale for s in spec]), 2)
    serial = rng.permutation(90_000)[:n] + 10_000
    counts["products"] = c.insert(
        conn,
        "products",
        {
            "product_id": np.arange(1, n + 1),
            "sku": [f"{s.code}-{num}" for s, num in zip(spec, serial.tolist(), strict=True)],
            "name": [f"{faker.word().title()} {faker.word().title()}" for _ in range(n)],
            "category": category,
            "unit_price": price,
            "weight_kg": np.maximum(weight, 0.01),
            "is_active": (rng.random(n) < ACTIVE_PRODUCT_RATE).astype(np.int64),
        },
    )
    return ProductFacts(price=price, weight=weight)
