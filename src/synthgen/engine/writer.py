"""Parquet output and the manifest (SPEC 8.5, 10): one zstd file per chunk, plus manifest.json.

Parts are written to a temporary name and renamed, so a crash never leaves half a part behind.
The manifest is rewritten after every chunk; until `complete` it records progress for a restart.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq

from synthgen.domain.base import StrictModel

DATA_DIR = "data"
MANIFEST_FILE = "manifest.json"
TEMPORARY_SUFFIX = ".tmp"


class ScenarioCount(StrictModel):
    """Rows carrying one scenario label."""

    scenario: str
    rows: int


class ManifestFile(StrictModel):
    """One written Parquet file (a chunk), relative to the output folder."""

    path: str
    rows: int
    scenarios: list[ScenarioCount]


class ColumnSchema(StrictModel):
    """A column name and its Arrow type."""

    name: str
    type: str


class TableManifest(StrictModel):
    """What was written for one table."""

    name: str
    rows: int
    files: list[ManifestFile]
    columns: list[ColumnSchema]
    scenario_counts: list[ScenarioCount]


class Manifest(StrictModel):
    """Everything a reader needs to trust and reproduce the output.

    `run_hash` covers the spec, the run options and the generation settings that shape output;
    a restart reuses finished chunks only when it matches.
    """

    seed: int
    spec_hash: str
    run_hash: str
    complete: bool
    tables: list[TableManifest]

    def table(self, name: str) -> TableManifest | None:
        """One table's entry, or None when the table was not written."""
        return next((t for t in self.tables if t.name == name), None)


def part_path(out_dir: Path, table: str, part: int) -> Path:
    """Where chunk `part` of a table is written."""
    return out_dir / DATA_DIR / table / f"part-{part:05d}.parquet"


def write_part(
    out_dir: Path, table: str, part: int, data: pa.Table, compression: Literal["zstd"]
) -> Path:
    """Write one chunk with fixed writer options so equal data gives equal bytes."""
    path = part_path(out_dir, table, part)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + TEMPORARY_SUFFIX)
    pq.write_table(data, temporary, compression=compression)
    os.replace(temporary, path)
    return path


def remove_part(out_dir: Path, table: str, part: int) -> None:
    """Delete a chunk's file and any half-written temporary file."""
    path = part_path(out_dir, table, part)
    path.with_name(path.name + TEMPORARY_SUFFIX).unlink(missing_ok=True)
    path.unlink(missing_ok=True)


def read_table(out_dir: Path, entry: TableManifest, columns: list[str] | None = None) -> pa.Table:
    """Read a written table back, in part order."""
    parts = [pq.read_table(out_dir / f.path, columns=columns) for f in entry.files]
    return pa.concat_tables(parts) if parts else pa.table({})


def write_manifest(out_dir: Path, manifest: Manifest) -> Path:
    """Write manifest.json next to the data folder (atomically)."""
    path = out_dir / MANIFEST_FILE
    out_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + TEMPORARY_SUFFIX)
    temporary.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return path


def read_manifest(out_dir: Path) -> Manifest:
    """Load manifest.json from an output folder."""
    return Manifest.model_validate_json((out_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
