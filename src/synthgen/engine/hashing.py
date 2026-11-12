"""Seeded 64-bit hashes of keys: stable across runs, chunks and worker counts."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from numpy.typing import NDArray

_MIX_1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX_2 = np.uint64(0x94D049BB133111EB)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


def _splitmix(values: NDArray[np.uint64]) -> NDArray[np.uint64]:
    with np.errstate(over="ignore"):
        z = values + _GOLDEN
        z = (z ^ (z >> np.uint64(30))) * _MIX_1
        z = (z ^ (z >> np.uint64(27))) * _MIX_2
        return z ^ (z >> np.uint64(31))


def entity_hash(keys: NDArray[Any], seed: int) -> NDArray[np.uint64]:  # Any: int or text keys
    """A seeded 64-bit hash per key, stable across runs, chunks and worker counts."""
    salt = _splitmix(np.array([seed % 2**64], dtype=np.uint64))[0]
    if np.issubdtype(keys.dtype, np.integer):
        return _splitmix(keys.astype(np.int64).view(np.uint64) ^ salt)
    seed_bytes = int(seed % 2**64).to_bytes(8, "little")
    return np.array(
        [
            int.from_bytes(
                hashlib.blake2b(str(k).encode(), digest_size=8, key=seed_bytes).digest(), "little"
            )
            for k in keys
        ],
        dtype=np.uint64,
    )


def ranked(keys: NDArray[Any], seed: int) -> NDArray[np.int64]:  # Any: int or text keys
    """Positions of `keys` ordered by their seeded hash; the first ones are the 'hot' keys."""
    return np.argsort(entity_hash(keys, seed), kind="stable").astype(np.int64)


def derived_seed(seed: int, *parts: str) -> int:
    """A 63-bit seed for a named purpose (a pool, a hot-key ranking) derived from the spec seed."""
    text = ":".join([str(seed), *parts]).encode()
    return int.from_bytes(hashlib.blake2b(text, digest_size=8).digest(), "little") >> 1
