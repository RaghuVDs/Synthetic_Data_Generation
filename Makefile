.PHONY: setup sqlite-wide check test test-slow format coverage demo

COVERED = */synthgen/domain/*,*/synthgen/spec/*,*/synthgen/engine/*,*/synthgen/validation/*

setup:
	uv sync

# Optional: SQLite driver for tables wider than 2,000 columns (run after setup).
sqlite-wide:
	sh tools/build_wide_sqlite.sh

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

# Seed both sources, run one finance job on the stub LLM and show its report.
DEMO_LOG = output/demo.log
demo:
	uv run python main.py seed
	mkdir -p output
	LLM_PROVIDER=stub LLM_STUB_FIXTURES=tests/fixtures/llm/skeleton uv run python main.py run \
		--source finance --tables customers,accounts --rows customers=2000 --auto-approve \
		--request "Customers with unusually high balances for our balance alerts" | tee $(DEMO_LOG)
	LLM_PROVIDER=stub uv run python main.py report \
		$$(sed -n 's/^job \([0-9a-f]*\) started.*/\1/p' $(DEMO_LOG) | tail -1)
