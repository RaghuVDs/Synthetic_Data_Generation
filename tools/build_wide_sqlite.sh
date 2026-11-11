#!/usr/bin/env sh
# Build and install a SQLite driver for tables wider than 2,000 columns.
#
# Python's bundled sqlite3 is compiled with SQLITE_MAX_COLUMN=2000. This script builds pysqlite3
# statically on the official SQLite amalgamation with SQLITE_MAX_COLUMN=32767 (SQLite's maximum)
# and installs the wheel into the project environment. Sources are pinned by SHA-256. The C
# compiler comes from the `ziglang` pip package, so no system compiler or sudo is needed.
set -eu

PYSQLITE3_VERSION=0.5.4
PYSQLITE3_URL="https://files.pythonhosted.org/packages/source/p/pysqlite3/pysqlite3-${PYSQLITE3_VERSION}.tar.gz"
PYSQLITE3_SHA256=fbc69bfdc0cb43a5badd5403b126d5151371b5037e0397ba9802bb440c5b0021
SQLITE_NAME=sqlite-amalgamation-3500400
SQLITE_URL="https://www.sqlite.org/2025/${SQLITE_NAME}.zip"
SQLITE_SHA256=1d3049dd0f830a025a53105fc79fd2ab9431aea99e137809d064d8ee8356b032
MAX_COLUMN=32767

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK="${WIDE_SQLITE_BUILD_DIR:-$ROOT/.cache/wide-sqlite}"
PROJECT_PYTHON=$(cd "$ROOT" && uv run python -c 'import sys; print(sys.executable)')

mkdir -p "$WORK"
cd "$WORK"
[ -f pysqlite3.tar.gz ] || curl -sSfL "$PYSQLITE3_URL" -o pysqlite3.tar.gz
[ -f amalgamation.zip ] || curl -sSfL "$SQLITE_URL" -o amalgamation.zip
printf '%s  pysqlite3.tar.gz\n%s  amalgamation.zip\n' "$PYSQLITE3_SHA256" "$SQLITE_SHA256" | sha256sum -c -

rm -rf "pysqlite3-${PYSQLITE3_VERSION}" "$SQLITE_NAME" build-env
tar xzf pysqlite3.tar.gz
"$PROJECT_PYTHON" -c "import zipfile; zipfile.ZipFile('amalgamation.zip').extractall('.')"
cp "$SQLITE_NAME/sqlite3.c" "$SQLITE_NAME/sqlite3.h" "pysqlite3-${PYSQLITE3_VERSION}/"

uv venv -q --python "$PROJECT_PYTHON" build-env
VIRTUAL_ENV="$WORK/build-env" uv pip install -q ziglang setuptools wheel
printf '#!/bin/sh\nexec "%s" -m ziglang cc "$@"\n' "$WORK/build-env/bin/python" > zcc
chmod +x zcc

cd "pysqlite3-${PYSQLITE3_VERSION}"
CC="$WORK/zcc" LDSHARED="$WORK/zcc -shared" CFLAGS="-DSQLITE_MAX_COLUMN=$MAX_COLUMN" \
    "$WORK/build-env/bin/python" setup.py -q build_static bdist_wheel
uv pip install -q --reinstall --python "$PROJECT_PYTHON" dist/pysqlite3-*.whl
cd "$ROOT"  # import the installed wheel, not the source tree
"$PROJECT_PYTHON" -c "import pysqlite3.dbapi2 as d; print('installed wide SQLite', d.sqlite_version)"
