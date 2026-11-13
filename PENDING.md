# Pending work

Last updated 2026-11-12. All code tasks in `PLAN.md` are done; what remains is
below. Run every command from WSL (Ubuntu) in `/mnt/d/Synthetic_Data_Generation`,
not from Windows PowerShell: `uv` and `.venv` live in WSL.

## 1. Full check and commits (can be done now)

Uncommitted work:

- **Row-target change:** a row count on a child table also sizes its parent
  tables (3,000 orders gives about 2,000 customers).
  Files: `src/synthgen/spec/row_targets.py`,
  `src/synthgen/agents/nodes/baseline_compiler.py`, and the updated tests.
- **Local web page:** `python main.py ui`.
  Files: `src/synthgen/web/`, `src/synthgen/cli/ui.py`,
  `tests/integration/test_web.py`, plus `CLAUDE.md`, `README.md`, `PLAN.md`
  and `config/settings.yaml`.

Steps:

- [ ] Try the page: `uv run python main.py ui`, open http://localhost:8000.
  The first start can take about 2 minutes.
- [ ] Run `make check` (20–30 minutes on `/mnt/d`). Lint, typing and the
  targeted tests pass; the full suite has not run since these changes.
- [ ] Commit in two parts:
  - `feat: child row targets size their root tables`
  - `feat: local web page to pick tables and generate`

## 2. T3.10 Live check (needs the real model)

- [ ] Set the LLM variables in `.env` for the company Llama model.
- [ ] Run one real request per source (finance and orders).
- [ ] Record the responses into `tests/fixtures/llm/` with no secrets in them.

## 3. T7.3 Scale runs (needs the real model)

- [ ] Run `uv run python main.py eval`: 18 golden requests at 100, 100,000 and
  1,000,000 rows, 54 jobs in all. This is a long run.
- [ ] Fix any findings.
- Earlier stub runs completed on both sources at all three volumes. Expect
  more rows and memory now, because child targets enlarge parent tables.

When 2 and 3 pass, milestone M7 and the POC are complete.

## Optional

- Move the repo from `/mnt/d` to the Linux filesystem (for example
  `~/Synthetic_Data_Generation`). Startup and `make check` would take a
  fraction of the time.
