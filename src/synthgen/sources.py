"""Source catalog: find, open, seed and describe the SQLite sources named in requests."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from synthgen.config import PathSettings, SeedSettings
from synthgen.connectors.sqlite import SqliteConnector
from synthgen.domain.errors import MetadataError, NotFoundError
from synthgen.domain.job import IDENTIFIER_PATTERN
from synthgen.domain.metadata import SourceOverview, TableSummary
from synthgen.domain.ports import Source
from synthgen.metadata.provider import SourceDoc, YamlMetadataProvider, load_source_doc
from synthgen.seed.finance import build_finance
from synthgen.seed.orders import build_orders

SeedBuilder = Callable[[Path, int, float], dict[str, int]]
SEED_BUILDERS: dict[str, SeedBuilder] = {"finance": build_finance, "orders": build_orders}


class SourceCatalog:
    """Maps source names to `<data_dir>/<name>.db` and `<metadata_dir>/<name>.yaml`."""

    def __init__(self, paths: PathSettings) -> None:
        """Use the configured folders."""
        self._paths = paths

    def path(self, source: str) -> Path:
        """Where a source's database lives; rejects names that are not plain identifiers."""
        if not re.fullmatch(IDENTIFIER_PATTERN, source):
            raise NotFoundError(f"invalid source name {source!r}")
        return self._paths.data_dir / f"{source}.db"

    def require(self, source: str) -> Path:
        """The database path, failing with a hint when the source has not been seeded."""
        path = self.path(source)
        if not path.is_file():
            raise NotFoundError(
                f"source {source!r} not found at {path}; run `python main.py seed` first"
            )
        return path

    @contextmanager
    def open(self, source: str) -> Iterator[Source]:
        """A read-only connector and merged metadata, closed on exit."""
        connector = SqliteConnector(self.require(source))
        try:
            yield Source(
                connector=connector, metadata=YamlMetadataProvider(self.doc(source), connector)
            )
        finally:
            connector.close()

    def doc(self, source: str) -> SourceDoc:
        """The source's metadata YAML; without one, relationships come from the database."""
        path = self._paths.metadata_dir / f"{source}.yaml"
        if not path.is_file():
            return SourceDoc(source=source)
        doc = load_source_doc(path)
        if doc.source != source:
            raise MetadataError(f"{path} describes source {doc.source!r}, expected {source!r}")
        return doc

    def seed(self, settings: SeedSettings) -> dict[str, dict[str, int]]:
        """Build every fake source database; row counts per source and table."""
        return {
            name: build(self.path(name), settings.random_seed, settings.scale)
            for name, build in SEED_BUILDERS.items()
        }

    def overview(self, source: str) -> SourceOverview:
        """Tables with row counts, plus every relationship."""
        with self.open(source) as src:
            return SourceOverview(
                source=source,
                description=self.doc(source).description,
                tables=tuple(
                    TableSummary(
                        name=t.name,
                        description=t.description,
                        row_count=src.connector.row_count(t.name),
                        column_count=len(t.columns),
                    )
                    for t in src.metadata.tables()
                ),
                relationships=tuple(src.metadata.relationships()),
            )
