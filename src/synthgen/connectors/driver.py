"""SQLite driver selection: a wide build when installed, otherwise the standard library.

Python's bundled `sqlite3` cannot open tables wider than 2,000 columns. `make sqlite-wide`
installs `pysqlite3` built with SQLITE_MAX_COLUMN=32767; when present it is used for sources.
"""

from __future__ import annotations

import re
import sqlite3
from types import ModuleType

DEFAULT_MAX_COLUMN = 2000
_MAX_COLUMN = re.compile(r"^MAX_COLUMN=(\d+)$")


def sqlite_driver() -> ModuleType:
    """The widest SQLite driver available."""
    try:
        import pysqlite3.dbapi2 as wide  # noqa: PLC0415 - optional dependency, chosen at run time
    except ImportError:
        return sqlite3
    module: ModuleType = wide
    return module


def max_columns(driver: ModuleType) -> int:
    """The driver's compile-time column limit."""
    conn = driver.connect(":memory:")
    try:
        options = [str(row[0]) for row in conn.execute("PRAGMA compile_options")]
    finally:
        conn.close()
    found = [int(m.group(1)) for o in options if (m := _MAX_COLUMN.match(o))]
    return found[0] if found else DEFAULT_MAX_COLUMN
