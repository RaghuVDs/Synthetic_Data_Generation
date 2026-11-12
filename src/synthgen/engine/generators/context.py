"""What a generator may read besides its own spec and random stream."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from synthgen.spec.evaluator import Column
from synthgen.spec.generators import SemanticKind

Values = NDArray[Any]  # Any: int64, float64 or object (text, mixed categorical values, None)
KeyLookup = Callable[[str, str], Values]


class PoolSource(Protocol):
    """Seeded pools of fake personal values, one per kind."""

    def pool(self, kind: SemanticKind) -> Values:
        """The pool for one kind; the same seed always gives the same pool."""
        ...


@dataclass(frozen=True)
class GenContext:
    """Row offset, spec seed, parent keys, pools and the other columns of the rows being drawn.

    `frame` maps a column name (or `parent_table.column`) to its type and values, one value per
    row being drawn; `derived` and `conditional` generators read it.
    """

    row_offset: int
    seed: int
    parent_keys: KeyLookup
    pools: PoolSource
    frame: Mapping[str, Column]

    def subset(self, rows: NDArray[np.int64]) -> GenContext:
        """The same context restricted to some rows of the frame."""
        return replace(self, frame={k: (t, v[rows]) for k, (t, v) in self.frame.items()})
