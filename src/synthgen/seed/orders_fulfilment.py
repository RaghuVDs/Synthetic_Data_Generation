"""Order lines and shipments for the fake orders source, including the `is_late` label."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np

from synthgen.seed import common as c

EXTRA_ITEMS_MEAN = 2.33
EXTRA_QUANTITY_MEAN = 0.4
HEAVY_KG = 10.0
DISCOUNT_FACTOR = 0.9
HANDLING_DAYS = 1
DISPATCH_SHAPE = 2.0
DISPATCH_SCALE_DAYS = 0.6
SHIPPED_STATUSES = ["shipped", "delivered", "returned"]


@dataclass(frozen=True)
class ProductFacts:
    """What order lines need to know about products."""

    price: c.FloatArray
    weight: c.FloatArray


@dataclass(frozen=True)
class Carrier:
    """A carrier: popularity, promised days, and log-normal transit time."""

    name: str
    weight: float
    promised_days: int
    median_transit_days: float
    transit_sd: float


CARRIERS = [
    Carrier("SwiftParcel", 40, 3, 1.8, 0.40),
    Carrier("NorthStar Freight", 30, 5, 3.0, 0.40),
    Carrier("BlueLine Post", 20, 7, 4.2, 0.45),
    Carrier("Kestrel Express", 10, 2, 1.1, 0.60),
]


def build_items(
    conn: sqlite3.Connection,
    rng: np.random.Generator,
    products: ProductFacts,
    discounted: c.BoolArray,
) -> tuple[int, c.FloatArray]:
    """Insert order lines; return the line count and each order's total."""
    n_orders = discounted.shape[0]
    order = np.repeat(np.arange(n_orders), 1 + rng.poisson(EXTRA_ITEMS_MEAN, n_orders))
    m = order.shape[0]
    product = c.zipf_indices(rng, products.price.shape[0], m, exponent=1.0)
    heavy = products.weight[product] > HEAVY_KG
    quantity = np.where(heavy, 1, 1 + rng.poisson(EXTRA_QUANTITY_MEAN, m))
    unit_price = np.round(
        products.price[product] * np.where(discounted[order], DISCOUNT_FACTOR, 1.0), 2
    )
    line_total = np.round(quantity * unit_price, 2)
    count = c.insert(
        conn,
        "order_items",
        {
            "order_item_id": np.arange(1, m + 1),
            "order_id": order + 1,
            "product_id": product + 1,
            "quantity": quantity,
            "unit_price": unit_price,
            "line_total": line_total,
        },
    )
    return count, np.bincount(order, weights=line_total, minlength=n_orders)


def build_shipments(
    conn: sqlite3.Connection,
    rng: np.random.Generator,
    order_times: c.FloatArray,
    status: c.StrArray,
) -> int:
    """One shipment per shipped, delivered or returned order; label late deliveries."""
    rows = np.flatnonzero(np.isin(status, SHIPPED_STATUSES))
    k = rows.shape[0]
    carrier = rng.choice(len(CARRIERS), size=k, p=_weights())
    day = c.SECONDS_PER_DAY
    ordered = order_times[rows]
    shipped = ordered + rng.gamma(DISPATCH_SHAPE, DISPATCH_SCALE_DAYS, k) * day
    promised_days = np.array([CARRIERS[i].promised_days for i in carrier])
    promised = np.floor(ordered / day) * day + (promised_days + HANDLING_DAYS) * day
    median = np.array([CARRIERS[i].median_transit_days for i in carrier])
    spread = np.array([CARRIERS[i].transit_sd for i in carrier])
    delivered = shipped + rng.lognormal(np.log(median), spread) * day
    arrived = status[rows] != "shipped"
    late = np.floor(delivered / day) > np.floor(promised / day)
    return c.insert(
        conn,
        "shipments",
        {
            "shipment_id": np.arange(1, k + 1),
            "order_id": rows + 1,
            "carrier": [CARRIERS[i].name for i in carrier],
            "shipped_date": c.iso_dates(shipped),
            "promised_date": c.iso_dates(promised),
            "delivered_date": [
                d if a else None for d, a in zip(c.iso_dates(delivered), arrived, strict=True)
            ],
            "is_late": [
                int(lt) if a else None for lt, a in zip(late.tolist(), arrived, strict=True)
            ],
        },
    )


def _weights() -> c.FloatArray:
    """Carrier popularity as probabilities."""
    raw = np.array([carrier.weight for carrier in CARRIERS], dtype=np.float64)
    return np.asarray(raw / raw.sum(), dtype=np.float64)
