.PHONY: setup check test test-slow format

setup:
	uv sync

check:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy
	uv run pytest

test:
	uv run pytest

test-slow:
	uv run pytest -m slow

format:
	uv run ruff format .
	uv run ruff check --fix .
