"""Parquet output and the manifest (SPEC 8.5, 10): one zstd file per chunk, plus manifest.json."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq

from synthgen.domain.base import StrictModel

DATA_DIR = "data"
MANIFEST_FILE = "manifest.json"


class ManifestFile(StrictModel):
    """One written Parquet file, relative to the output folder."""

    path: str
    rows: int


class ColumnSchema(StrictModel):
    """A column name and its Arrow type."""

    name: str
    type: str


class ScenarioCount(StrictModel):
    """Rows carrying one scenario label."""

    scenario: str
    rows: int


class TableManifest(StrictModel):
    """What was written for one table."""

    name: str
    rows: int
    files: list[ManifestFile]
    columns: list[ColumnSchema]
    scenario_counts: list[ScenarioCount]


class Manifest(StrictModel):
    """Everything a reader needs to trust and reproduce the output."""

    seed: int
    spec_hash: str
    tables: list[TableManifest]

    def table(self, name: str) -> TableManifest | None:
        """One table's entry, or None when the table was not written."""
        return next((t for t in self.tables if t.name == name), None)


def part_path(out_dir: Path, table: str, part: int) -> Path:
    """Where chunk `part` of a table is written."""
    return out_dir / DATA_DIR / table / f"part-{part:05d}.parquet"


def write_part(
    out_dir: Path, table: str, part: int, data: pa.Table, compression: Literal["zstd"]
) -> ManifestFile:
    """Write one chunk with fixed writer options so equal data gives equal bytes."""
    path = part_path(out_dir, table, part)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(data, path, compression=compression)
    return ManifestFile(path=path.relative_to(out_dir).as_posix(), rows=data.num_rows)


def read_table(out_dir: Path, entry: TableManifest, columns: list[str] | None = None) -> pa.Table:
    """Read a written table back, in part order."""
    parts = [pq.read_table(out_dir / f.path, columns=columns) for f in entry.files]
    return pa.concat_tables(parts) if parts else pa.table({})


def write_manifest(out_dir: Path, manifest: Manifest) -> Path:
    """Write manifest.json next to the data folder."""
    path = out_dir / MANIFEST_FILE
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path


def read_manifest(out_dir: Path) -> Manifest:
    """Load manifest.json from an output folder."""
    return Manifest.model_validate_json((out_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
