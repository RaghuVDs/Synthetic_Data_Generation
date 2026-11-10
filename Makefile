.PHONY: setup check test test-slow format coverage

COVERED = */synthgen/domain/*,*/synthgen/spec/*,*/synthgen/engine/*,*/synthgen/validation/*

setup:
	uv sync

check:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy
	uv run pytest --cov=synthgen --cov-report=
	$(MAKE) coverage

coverage:
	uv run coverage report --include="$(COVERED)" --skip-covered --fail-under=85

test:
	uv run pytest

test-slow:
	uv run pytest -m slow

format:
	uv run ruff format .
	uv run ruff check --fix .
