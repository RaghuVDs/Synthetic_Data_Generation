"""Text generators: masks (with an optional Luhn digit) and fake personal values from pools."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import numpy as np
from faker import Faker
from numpy.typing import NDArray

from synthgen.engine.generators.context import GenContext, Values
from synthgen.engine.hashing import derived_seed
from synthgen.spec.generators import PatternGen, SemanticGen, SemanticKind

DIGIT_ZERO = ord("0")
LUHN_BASE = 10


def luhn_digits(payload: NDArray[np.int64]) -> NDArray[np.int64]:
    """The Luhn check digit for each row of digits (rows x digits, check digit not included)."""
    weights = np.where(np.arange(payload.shape[1])[::-1] % 2 == 0, 2, 1)
    doubled = payload * weights
    total = (doubled - 9 * (doubled > 9)).sum(axis=1)  # noqa: PLR2004 - Luhn folds 10..18
    return np.asarray((LUHN_BASE - total % LUHN_BASE) % LUHN_BASE, dtype=np.int64)


def _mask_values(
    mask: str, rows: int, rng: np.random.Generator, check_digit: bool
) -> NDArray[np.str_]:
    codes = np.tile(np.array([ord(c) for c in mask], dtype=np.uint32), (rows, 1))
    for symbol, low, high in (("A", ord("a"), ord("z")), ("9", ord("0"), ord("9"))):
        cols = [i for i, c in enumerate(mask) if c == symbol]
        if cols:
            codes[:, cols] = rng.integers(low, high + 1, (rows, len(cols)), dtype=np.uint32)
    if check_digit:
        digits = [i for i, c in enumerate(mask) if c == "9"]
        payload = codes[:, digits[:-1]].astype(np.int64) - DIGIT_ZERO
        codes[:, digits[-1]] = luhn_digits(payload) + DIGIT_ZERO
    return np.ascontiguousarray(codes).view(f"<U{len(mask)}").ravel()


def patterns(gen: PatternGen, n: int, rng: np.random.Generator, _ctx: GenContext) -> Values:
    """`A` becomes a lowercase letter, `9` a digit; other characters are copied."""
    picks = rng.choice(len(gen.masks), size=n, p=np.asarray(gen.weights) / sum(gen.weights))
    out = np.empty(n, dtype=object)
    for index, mask in enumerate(gen.masks):
        rows = np.flatnonzero(picks == index)
        if rows.size == 0:
            continue
        if not mask:
            out[rows] = ""
            continue
        out[rows] = _mask_values(mask, rows.size, rng, gen.check_digit == "luhn").astype(object)
    return out


def semantic(gen: SemanticGen, n: int, rng: np.random.Generator, ctx: GenContext) -> Values:
    """Values drawn uniformly from the kind's seeded pool."""
    pool = ctx.pools.pool(gen.kind)
    return pool[rng.integers(0, pool.size, n)]


Maker = Callable[[Faker], str]


class SemanticPools:
    """Pools of fake values built once per kind with Faker seeded from the spec seed and the kind.

    Faker never reads the clock here: birth dates fall between two fixed dates from settings.
    """

    def __init__(self, seed: int, size: int, locale: str, birth_dates: tuple[date, date]) -> None:
        """Pools are built lazily; `birth_dates` bounds the date_of_birth kind."""
        self._seed = seed
        self._size = size
        self._locale = locale
        first, last = birth_dates
        self._makers: dict[SemanticKind, Maker] = {
            SemanticKind.FIRST_NAME: lambda f: f.first_name(),
            SemanticKind.LAST_NAME: lambda f: f.last_name(),
            SemanticKind.FULL_NAME: lambda f: f.name(),
            SemanticKind.EMAIL: lambda f: f.email(),
            SemanticKind.PHONE: lambda f: f.phone_number(),
            SemanticKind.STREET_ADDRESS: lambda f: f.street_address(),
            SemanticKind.CITY: lambda f: f.city(),
            SemanticKind.POSTCODE: lambda f: f.postcode(),
            SemanticKind.COMPANY: lambda f: f.company(),
            SemanticKind.CARD_NUMBER: lambda f: f.credit_card_number(),
            SemanticKind.DATE_OF_BIRTH: lambda f: f.date_between_dates(first, last).isoformat(),
        }
        self._pools: dict[SemanticKind, Values] = {}

    def pool(self, kind: SemanticKind) -> Values:
        """The pool for one kind, built on first use."""
        if kind not in self._pools:
            faker = Faker(self._locale)
            faker.seed_instance(derived_seed(self._seed, "semantic", kind.value))
            make = self._makers[kind]
            self._pools[kind] = np.array([make(faker) for _ in range(self._size)], dtype=object)
        return self._pools[kind]
