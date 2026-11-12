"""What validation reads: the engine's output, one column at a time, and the run's inputs.

Columns are read one at a time from the Parquet parts, so a table of thousands of columns never
has to fit in memory at once; the label columns are cached per table.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from synthgen.config import ValidationSettings
from synthgen.domain.plan import ScenarioPlan
from synthgen.domain.ports import Connector
from synthgen.engine.runner import RunOptions
from synthgen.engine.schema import DEFECT_COLUMN, SCENARIO_COLUMN
from synthgen.engine.writer import Manifest, TableManifest, read_table
from synthgen.profiling.models import TableProfile
from synthgen.spec.models import TYPICAL, GenerationSpec

Cells = NDArray[Any]  # Any: int64, float64 or object columns as Parquet returns them


@dataclass(frozen=True)
class ValidationInputs:
    """Everything the checks compare the output with.

    `plan` carries the scenario signatures; `source` answers privacy lookups and may be None
    when the source is not open (those checks then report `info`).
    """

    spec: GenerationSpec
    options: RunOptions
    settings: ValidationSettings
    profiles: Mapping[str, TableProfile]
    plan: ScenarioPlan | None
    source: Connector | None


class OutputData:
    """Columns of the written tables, with the per-row scenario and defect labels cached."""

    def __init__(self, out_dir: Path, manifest: Manifest) -> None:
        """Read from `out_dir` as the manifest describes it."""
        self._out_dir = out_dir
        self._manifest = manifest
        self._labels: dict[tuple[str, str], Cells] = {}

    def entry(self, table: str) -> TableManifest | None:
        """The manifest entry of a table, or None when it was not written."""
        return self._manifest.table(table)

    def column(self, table: str, name: str) -> Cells:
        """One whole column, in row order."""
        entry = self.entry(table)
        if entry is None:
            return np.array([], dtype=object)
        data = read_table(self._out_dir, entry, [name]).column(name)
        values = np.asarray(data.to_numpy(zero_copy_only=False))
        if not data.null_count:
            return values
        cells = values.astype(object)  # Arrow gives NaN for null floats; nulls must stay None
        cells[np.asarray(data.is_null().to_numpy(zero_copy_only=False), dtype=bool)] = None
        return cells

    def label(self, table: str, name: str) -> Cells:
        """A label column (`_scenario`, `_category`, `_defect`), cached."""
        key = (table, name)
        if key not in self._labels:
            self._labels[key] = self.column(table, name).astype(object)
        return self._labels[key]

    def typical(self, table: str) -> NDArray[np.bool_]:
        """Rows no scenario touched and no defect broke; fidelity looks only at these."""
        scenarios = self.label(table, SCENARIO_COLUMN)
        defects = self.label(table, DEFECT_COLUMN)
        return np.asarray((scenarios == TYPICAL) & np.equal(defects, None), dtype=bool)
