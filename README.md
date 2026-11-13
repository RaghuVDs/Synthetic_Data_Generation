# Synthetic Data Agent (POC)

Turns a plain-language test request into labelled synthetic data for chosen
SQLite tables. An LLM reads the request and plans scenarios; deterministic code
profiles the source, generates the rows and validates them. The LLM never sees
source rows and never writes data, SQL or code. Design: `SPEC.md`. Progress and
decisions: `PLAN.md`.

## Setup

Needs Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
make setup                  # uv sync
cp .env.example .env        # then edit; .env is never committed
uv run python main.py seed  # builds data/finance.db and data/orders.db (fake data)
```

With `LLM_PROVIDER=stub` (the default) every LLM answer comes from recorded
fixtures and nothing touches the network. For a real model set
`LLM_PROVIDER=openai_compatible`, `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY`
(Groq by default). `.env.example` lists every variable, including settings for
self-hosted servers and small token budgets. Any value in
`config/settings.yaml` can be overridden as `SYNTHGEN_<GROUP>__<KEY>`.

## Commands

All commands run as `uv run python main.py <command>`; `--help` on any of them
lists its options.

| Command | What it does |
| --- | --- |
| `seed` | Build the fake source databases |
| `tables --source finance` | List tables, row counts and relationships |
| `profile --source finance --tables cards,transactions` | Profile tables and print their LLM-safe cards |
| `run --source S --tables a,b --request "..."` | Start a job: `--rows t=N`, `--seed N`, `--include-invalid`, `--auto-approve`, `--workers N` |
| `resume JOB` | Continue a paused or interrupted job |
| `status JOB`, `report JOB`, `preview JOB --table t` | Inspect a job |
| `eval` | Run the golden set and print metrics: `--volumes 100,100000`, `--only name`, `--source orders` |
| `ui` | Serve a local web page (http://127.0.0.1:8000) to pick tables, describe the test and generate: `--port N` |

A job asks clarifying questions when the request is ambiguous, then shows its
plan (scenario, category, share, feasibility, signature, source) for you to
approve, edit or reject. Output goes to `output/jobs/<job_id>/`: Parquet per
table under `data/`, `manifest.json`, `spec.json` and `validation_report.json`.
Every row carries a `_scenario` label; deliberately broken rows also carry `_defect`.

## Web page

`uv run python main.py ui` opens a page on your machine only. Pick a source,
tick tables (rows per table are optional), describe what to test and press
Generate. The page shows the agent's questions and its plan (untick scenarios
to drop, or reject with feedback), then the summary, rows per table, any failed
checks and a preview of each table. It runs the same job as `run`.

## Demo

```sh
make demo
```

This seeds both sources and runs a finance job on the stub LLM, with
auto-approval and 2,000 customers. It prints the job ID and status, then the
summary, row counts and validation results. Then:

```sh
uv run python main.py preview <job_id> --table accounts
uv run python main.py report <job_id>
```

For a live run, set the LLM variables in `.env` and drop `--auto-approve` to
review the plan yourself:

```sh
uv run python main.py run --source orders --tables customers,orders,shipments \
  --request "Late deliveries for our delivery-time model, plus orders that ship fast" \
  --rows orders=100000
```

`eval` with a real model runs the 18 golden requests (`tests/golden/`) at 100,
100,000 and 1,000,000 rows on each request's main table, and saves a JSON
report under `output/eval/`.

## Development

`make check` runs formatting, lint, strict typing, all tests and the coverage
gate (85% on `domain`, `spec`, `engine`, `validation`). `make test-slow`
generates 1,000,000 rows twice and checks the bytes are identical. Tests never
call a real LLM. Architecture rules and conventions are in `CLAUDE.md`.
