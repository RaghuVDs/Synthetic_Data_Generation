"""Read tables already written in this run: whole columns, or a range of rows."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from synthgen.domain.errors import EngineError
from synthgen.engine.generators import Values
from synthgen.engine.writer import TableManifest, read_table


class WrittenTables:
    """Columns of written tables, read from their Parquet parts and cached per column."""

    def __init__(self, out_dir: Path, written: Mapping[str, TableManifest]) -> None:
        """Read from `out_dir`; only tables listed in `written` exist."""
        self._out_dir = out_dir
        self._written = written
        self._cache: dict[tuple[str, str], Values] = {}

    def _entry(self, table: str) -> TableManifest:
        entry = self._written.get(table)
        if entry is None:
            raise EngineError(f"{table} must be generated before it is read")
        return entry

    def column(self, table: str, column: str) -> Values:
        """A whole column (every part, in order)."""
        key = (table, column)
        if key not in self._cache:
            data = read_table(self._out_dir, self._entry(table), [column]).column(column)
            self._cache[key] = np.asarray(data.to_numpy(zero_copy_only=False))
        return self._cache[key]

    def rows(self, table: str, column: str, start: int, stop: int) -> Values:
        """Rows [start, stop) of a column, reading only the parts that hold them."""
        if (table, column) in self._cache:
            return self._cache[(table, column)][start:stop]
        pieces: list[Values] = []
        first = 0
        for part in self._entry(table).files:
            last = first + part.rows
            if last > start and first < stop:
                data = pq.read_table(self._out_dir / part.path, columns=[column]).column(column)
                values = np.asarray(data.to_numpy(zero_copy_only=False))
                pieces.append(values[max(start - first, 0) : min(stop, last) - first])
            first = last
        return np.concatenate(pieces) if pieces else np.array([], dtype=object)
