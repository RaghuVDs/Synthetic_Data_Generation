"""Composition root and use-case functions. Each public method mirrors a future REST endpoint."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from pathlib import Path

from synthgen.config import Settings, load_settings, read_environment
from synthgen.connectors.sqlite import SqliteConnector
from synthgen.domain.errors import MetadataError, NotFoundError
from synthgen.domain.job import IDENTIFIER_PATTERN
from synthgen.domain.metadata import SourceOverview, TableSummary
from synthgen.metadata.provider import SourceDoc, YamlMetadataProvider, load_source_doc
from synthgen.seed.finance import build_finance
from synthgen.seed.orders import build_orders

SeedBuilder = Callable[[Path, int, float], dict[str, int]]
SEED_BUILDERS: dict[str, SeedBuilder] = {"finance": build_finance, "orders": build_orders}


class Application:
    """Wires concrete classes from settings and exposes the use cases."""

    def __init__(self, settings: Settings) -> None:
        """Keep the settings; every collaborator is built from them on demand."""
        self._settings = settings

    @property
    def settings(self) -> Settings:
        """The settings this application was built with."""
        return self._settings

    def seed(self) -> dict[str, dict[str, int]]:
        """Build every fake source database; return row counts per source and table."""
        cfg = self._settings.seed
        return {
            name: build(self._source_path(name), cfg.random_seed, cfg.scale)
            for name, build in SEED_BUILDERS.items()
        }

    def list_tables(self, source: str) -> SourceOverview:
        """Tables of a source with row counts, plus every relationship."""
        connector = self._connector(source)
        doc = self._source_doc(source)
        metadata = YamlMetadataProvider(doc, connector)
        return SourceOverview(
            source=source,
            description=doc.description,
            tables=tuple(
                TableSummary(
                    name=t.name,
                    description=t.description,
                    row_count=connector.row_count(t.name),
                    column_count=len(t.columns),
                )
                for t in metadata.tables()
            ),
            relationships=tuple(metadata.relationships()),
        )

    def _source_path(self, source: str) -> Path:
        if not re.fullmatch(IDENTIFIER_PATTERN, source):
            raise NotFoundError(f"invalid source name {source!r}")
        return self._settings.paths.data_dir / f"{source}.db"

    def _connector(self, source: str) -> SqliteConnector:
        path = self._source_path(source)
        if not path.is_file():
            raise NotFoundError(
                f"source {source!r} not found at {path}; run `python main.py seed` first"
            )
        return SqliteConnector(path)

    def _source_doc(self, source: str) -> SourceDoc:
        """The source's metadata YAML; without one, relationships come from the database."""
        path = self._settings.paths.metadata_dir / f"{source}.yaml"
        if not path.is_file():
            return SourceDoc(source=source)
        doc = load_source_doc(path)
        if doc.source != source:
            raise MetadataError(f"{path} describes source {doc.source!r}, expected {source!r}")
        return doc


def build_application(
    environ: Mapping[str, str], settings_file: Path, env_file: Path | None
) -> Application:
    """Load settings from YAML, an optional .env file and the environment, then wire the app."""
    return Application(load_settings(settings_file, read_environment(environ, env_file)))
