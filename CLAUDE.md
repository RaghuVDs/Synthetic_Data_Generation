# Synthetic Data Agent (POC)

An agentic system that turns a plain-language test request into labelled synthetic
data for selected SQLite tables. An LLM plans; deterministic code profiles,
generates and validates. The full design is in `SPEC.md`.

## Start here (every session)

1. If `PLAN.md` does not exist in the repo root: read `SPEC.md` in full, then
   create `PLAN.md` exactly as `SPEC.md` section 12 defines. Write no code and
   create no other file first. Then stop and ask the user to approve the plan.
2. If `PLAN.md` exists: continue from its first unchecked task. Before starting
   a milestone, re-read the `SPEC.md` sections that milestone lists.
3. Work one milestone at a time. Tick a task in `PLAN.md` only when its tests
   pass. At the end of a milestone run `make check`, update the progress log in
   `PLAN.md`, and stop for review.
4. If `SPEC.md` is ambiguous or conflicts with what you find, ask. Record the
   answer under "Decisions" in `PLAN.md`. Do not guess.

## POC scope

In scope:
- The agent layer end to end: 15-node LangGraph graph, six LLM roles, planner
  tools, approval gate, two repair loops, checkpoint and resume.
- The code steps the agents depend on: metadata, SQLite connector, profiler,
  spec compiler, generation engine, validator.
- A command-line interface with one entry point, `main.py`.
- SQLite only: seed source databases and a separate state database.
- Parquet output in a local `output/` folder.

Out of scope (do not build): REST API, web UI, Docker, authentication, object
storage, any database other than SQLite, masking of real data, learned
generative models.

## Hard rules

- The LLM never produces data rows. It produces validated JSON only.
- The LLM never sees source rows. The only profile type an agent may receive is
  `LlmSafeProfile`. Sensitive columns expose masks and lengths, never values.
- The LLM never writes SQL or code. No `eval`, no `exec`, no dynamic imports.
- Code decides routing between graph nodes. The LLM decides content inside a node.
- Every loop has a cap. A reached cap ends the job as `needs_review`.
- Only the approval gate can start generation.
- All randomness comes from an injected `numpy.random.Generator`. Same seed,
  spec and profile version must give byte-identical Parquet.
- SQL values are always parameterised. Identifiers are validated against
  metadata, then quoted. The source database is opened read-only.
- Secrets come from environment variables. Never log or commit them.
- Tests never call a real LLM or the network. Use the stub client.

## Tech stack

- Python 3.11+, managed with `uv`, configured in `pyproject.toml`
- LangGraph 1.x with its SQLite checkpointer
- Pydantic v2 for every model that crosses a boundary
- OpenAI-compatible HTTP client for the LLM (Groq in the POC)
- Standard-library `sqlite3` for source and state databases
- NumPy, SciPy, pandas for statistics; NumPy, SciPy, PyArrow for generation
- Faker for seeded pools of personal fields
- Typer and Rich for the CLI
- pytest, ruff (lint and format), mypy (strict)

Check the installed version's documentation before using any LangGraph, Groq or
PyArrow API. Do not rely on memory for library signatures.

## Repository structure

```
.
├── CLAUDE.md  SPEC.md  PLAN.md  README.md
├── pyproject.toml  Makefile  .env.example  .gitignore
├── main.py                     # the ONLY entry point; no business logic
├── config/
│   ├── settings.yaml           # caps, thresholds, sample sizes, chunk size
│   └── metadata/               # one YAML per source: tables, relationships, tags
├── data/                       # gitignored: source *.db, state.db, checkpoints.db
├── output/                     # gitignored: jobs/<job_id>/...
├── src/synthgen/
│   ├── app.py                  # composition root and use-case functions
│   ├── config.py               # typed settings
│   ├── domain/                 # pure Pydantic models and errors; no I/O
│   ├── cli/                    # Typer commands, Rich rendering, prompts
│   ├── agents/                 # ONLY package that imports LangGraph or an LLM
│   │   ├── graph.py  state.py  routing.py
│   │   ├── llm/                # client protocol, openai_compatible, stub
│   │   ├── nodes/              # one module per node
│   │   ├── prompts/            # one versioned template per LLM role
│   │   └── tools/              # planner tools and registry
│   ├── metadata/               # provider, dependency graph
│   ├── connectors/             # Connector protocol, SQLite implementation
│   ├── profiling/              # statistics, privacy classes, cache
│   ├── scenarios/              # primitives, feasibility, templates/
│   ├── spec/                   # spec models, baseline compiler, merge, validator
│   ├── engine/                 # generators, copula, sequences, writer, runner
│   ├── validation/             # fidelity, integrity, scenario, privacy, report
│   ├── storage/                # job store, artifact store, profile cache
│   └── seed/                   # builds the fake source databases
└── tests/
    ├── unit/  integration/  golden/  fixtures/
```

## Architecture rules

- Dependencies point inward only:
  `main.py` → `cli` → `app` → `agents` → (`metadata`, `connectors`, `profiling`,
  `scenarios`, `spec`, `engine`, `validation`, `storage`) → `domain`.
- `domain` imports nothing from this project. Only `main.py` imports `cli`. Only
  `app` imports `agents`.
- `profiling`, `spec`, `engine` and `validation` must run with no LLM installed.
- Cross-package calls go through protocols defined in the consuming package or
  in `domain`. Concrete classes are wired only in `app.py`.
- `main.py` parses arguments and calls `synthgen.cli`. Nothing else.

## Engineering principles

- Single responsibility: one reason to change per module. Keep modules under
  300 lines and functions under 50.
- Depend on protocols, inject dependencies through constructors. No globals, no
  singletons, no module-level state.
- Pure functions for statistics, generators and validators: inputs in, outputs
  out, no I/O.
- Type hints everywhere; `mypy --strict` passes. No `Any` without a comment.
- Validate at every boundary with Pydantic: config, metadata, LLM output, specs.
- Fail loudly with typed errors from `domain/errors.py`. Never swallow an
  exception. Never return `None` to signal failure.
- Configuration lives in `config/settings.yaml` and environment variables. No
  magic numbers in code.
- Logging through the `logging` module with job ID and node name. `print` only
  in `cli`.
- Build the simplest thing that satisfies `SPEC.md`. No speculative features,
  no unused abstractions, no dead code, no commented-out code.
- Docstrings on every public class and function, saying what and why.
- Write the test with the code. Unit tests for every pure function; integration
  tests for the graph with the stub LLM, including failure paths.
- Coverage of at least 85% on `domain`, `spec`, `engine`, `validation`.

## Commands

- `make setup` installs dependencies with `uv sync`
- `make check` runs `ruff format --check`, `ruff check`, `mypy`, `pytest`
- `make test` runs `pytest`
- `python main.py seed` builds the fake source databases
- `python main.py run --source <name> --tables <a,b> --request "<text>"`
- `python main.py resume <job_id>` continues a paused or interrupted job
- `python main.py --help` lists every command

## Git workflow

- One commit per finished task, using Conventional Commits (`feat:`, `fix:`,
  `test:`, `refactor:`, `docs:`, `chore:`).
- `make check` must pass before every commit.
- Never commit `data/`, `output/`, `.env` or any key.
