# PLAN: Synthetic Data Agent (POC)

Single source of truth for progress. Rules: milestone order is fixed (ask before
changing it), no scope beyond `SPEC.md`, tick a task only when its tests pass.

---

## 1. Understanding

- A CLI tool (`main.py`) for testers: pick tables in a SQLite source, describe a
  test in plain language, get labelled synthetic Parquet data that covers it.
- Volume from 100 to 1,000,000 rows (10,000,000 is a stretch goal).
- An LLM plans through a 15-node LangGraph graph (six LLM roles, eight code
  steps, one human approval gate) with checkpoints, resume and capped loops.
- Deterministic code profiles the source, compiles a baseline spec, applies
  LLM-written scenario overlays, generates data and validates it.
- The LLM sees only `LlmSafeProfile` cards, emits validated JSON only, and never
  writes SQL, code or data rows.
- Domain-neutral: card fraud and order/delivery tests run through the same code.
- Same seed + spec + profile version gives byte-identical Parquet.

## 2. Assumptions and open questions

Questions marked **BLOCKING (Mx)** need an answer before milestone Mx starts.
Every other item is an assumption I will follow unless you object; once
confirmed or corrected it moves to section 3.

### Blocking questions

- **Q1 BLOCKING (M0): toolchain.** `uv` and `make` are not installed in this
  WSL environment (Python 3.13.11 and git 2.34.1 are). Proposal: I install `uv`
  user-level with its official installer (`~/.local/bin`, no sudo); you run
  `! sudo apt-get install -y make`. Git has no global `user.name`/`user.email`;
  proposal: set them repo-local (`git config user.name/user.email`) to your name
  and `raghunaras97@gmail.com`. Please confirm both, and the name to use.
- **Q2 BLOCKING (M1): where protocols live and which lateral imports are
  allowed.** `CLAUDE.md` says protocols live "in the consuming package or in
  `domain`", but the repo layout puts the `Connector` protocol in `connectors/`.
  Also `engine` and `validation` must read `spec` models, and `spec` must read
  profile models, which are lateral imports between inner packages.
  Proposal: (a) put `Connector`, `MetadataProvider` and store protocols in
  `domain/ports.py`; `connectors/` holds only the SQLite implementation.
  (b) Allow this acyclic order between inner packages, enforced by an
  architecture test: `domain` ← `metadata`, `connectors`, `storage` ←
  `profiling` ← `scenarios` ← `spec` ← `engine` ← `validation`.
- **Q3 BLOCKING (M2): row targets.** Spec says root tables are sized "from the
  request" and child tables from parent count × cardinality. Proposal:
  precedence for a root table is `--rows` > intent `volume_hint` >
  `generation.default_root_rows` in settings (1,000). `--rows` on a child table
  rescales its cardinality distribution so the expected total hits the target.
- **Q4 BLOCKING (M2): child tables with several parents** (e.g. `transactions`
  → `cards` and `merchants`). Proposal: the metadata YAML marks one relationship
  per child with `drives_cardinality: true` (default: the first non-nullable
  relationship listed). That parent sizes the child; other foreign keys are
  drawn from their parent's key range.
- **Q5 BLOCKING (M3): planner tool-call cap.** The hard rule says "a reached cap
  ends the job as `needs_review`". For the 8-call planner cap, ending the job
  is harsh. Proposal: at 8 tool calls the planner stops offering tools and makes
  its final structured call; only a failed final call (after 3 re-prompts) ends
  the job as `needs_review`.
- **Q6 BLOCKING (M6): fidelity at small volumes.** At 100 rows a KS statistic of
  ≤ 0.10 cannot be met by sampling noise alone (the 95% critical value at n=100
  is ≈ 0.136). Proposal: fidelity checks whose typical-row sample is below
  `validation.min_fidelity_rows` (default 1,000) report `skipped` with the
  reason; they do not fail. Integrity, privacy and scenario presence always run.
- **Q7 BLOCKING (M6): privacy checks on low-entropy values.** Synthetic names
  and cities drawn from Faker will match source values that also came from
  Faker (and real names in any real source). Proposal: the exact-match check
  ("10,000 synthetic sensitive values found in source = 0") applies to
  sensitive columns whose source cardinality ratio is ≥ 0.5 (identifier-like:
  email, phone, card number). Other sensitive columns are reported as `info`.
  The row-hash check hashes all non-key, non-label columns and skips tables
  whose combination space is smaller than the sample (reference tables).

### Assumptions (non-blocking)

- **A1 Feasibility check** runs as code at the end of the `scenario_planner`
  node, so the graph keeps exactly 15 nodes.
- **A2 Scenario allocation** is exact, not probabilistic: per-scenario entity
  counts are `max(1, round(share × N))` allocated by largest remainder across
  chunks; which entities get a scenario is chosen by seeded hash rank of the
  entity key. The share check then uses tolerance max(10% relative, 1 unit).
- **A3 Repair → approval.** A `SpecPatch` with `plan_change_required` routes back
  to `approval_gate` and counts against `plan_rounds`. An approval edit that
  fails validation is shown again with errors and also counts against
  `plan_rounds`.
- **A4 Baseline errors.** `spec_validator` errors that come from the baseline
  (no overlay involved) are code bugs: the job ends `failed`.
- **A5 Failure traced to an overlay** means every failing row has a non-typical
  `_scenario`. Otherwise an integrity or privacy failure ends the job `failed`.
- **A6 Reporter failure.** If the reporter fails after 3 re-prompts the job ends
  `needs_review` (hard rule), even though data and report are on disk.
- **A7 Support tables** (unselected parents added by `context_builder`) are
  generated and written, flagged `support` in the manifest, validated for
  integrity and fidelity, and get no scenarios.
- **A8 Rare-value folding.** For `internal` columns, values with fewer than 10
  rows fold into "other"; the generator redistributes that mass over retained
  values, so no literal "other" or suppressed value reaches the output.
- **A9 Stub LLM fixtures** are JSON files in `tests/fixtures/llm/<set>/`, keyed
  by role and call index; the set is chosen by `LLM_STUB_FIXTURES`. The CLI can
  run on the stub (`LLM_PROVIDER=stub`) for the M2 acceptance.
- **A10 Slow tests** (100,000 and 1,000,000 rows, full seed size) carry the
  pytest marker `slow`, are excluded from `make check`, and run with
  `make test-slow`.
- **A11 Coverage gate** (≥ 85% on `domain`, `spec`, `engine`, `validation`) is
  added to `make check` in M2, once those packages contain code.
- **A12 Settings** load with `pydantic-settings` + PyYAML: YAML defaults, then
  environment variables (prefix `SYNTHGEN_`, plus the `LLM_*` names from the
  spec), with an optional `.env` file.
- **A13 LLM HTTP client** uses the official `openai` Python SDK pointed at
  `LLM_BASE_URL` (OpenAI-compatible; works for Groq). Docs checked at M3.
- **A14 Text pools** (`text_content` primitive) are the only model-written field
  values. They are capped at 500, length-checked, and screened against source
  values of sensitive columns before use.
- **A15 Seed metadata.** Both seed databases declare foreign keys in their
  schema. One relationship in `orders.yaml` is deliberately left out to
  exercise the `PRAGMA foreign_key_list` fallback.
- **A16 Demo script** is a `make demo` target plus a "Demo" section in
  `README.md`; no extra `scripts/` folder.
- **A17 Eval at scale.** M7 golden runs use the real LLM (network, `LLM_API_KEY`)
  through the `eval` command; they are not part of `pytest`.

## 3. Decisions

- 2026-11-10: User accepted all proposed defaults for Q1–Q7 and A1–A17 as
  written in section 2. They are binding decisions from this date.
- 2026-11-10: A12 revised during T0.4. Settings use a small explicit loader in
  `config.py` (PyYAML + Pydantic + `python-dotenv`) instead of
  `pydantic-settings`. The environment is passed in as a mapping, so tests
  never touch real variables, and the unprefixed `LLM_*` names map cleanly.
  Override syntax is unchanged: `SYNTHGEN_<GROUP>__<KEY>` and `LLM_*`.
- 2026-11-10: Ruff excludes `*.md`; ruff 0.17 otherwise reformats Python code
  blocks inside `SPEC.md`.
- 2026-11-10: `uv` 0.13.7 installed user-level in `~/.local/bin`.
- 2026-11-10: Q1 resolved. Commits use the repo-local identity
  `raghunaras97 <raghunaras97@gmail.com>`; no remote is configured yet.
- 2026-11-10: `make` 4.3 installed without sudo (user choice): Ubuntu package
  unpacked into `~/.local/make`, binary linked as `~/.local/bin/make`.
- 2026-11-10 (M1): The `Connector` protocol in `domain/ports.py` adds
  `list_tables()` and `foreign_keys()` to the four methods of SPEC 8.2; the
  metadata provider needs them for identifier checks and the PRAGMA fallback.
  Composite foreign keys raise `MetadataError` (none exist in the seed data).
- 2026-11-10 (M1): Seeded sampling uses a golden-ratio multiplicative hash on
  `rowid` modulo 2^31-1, offset by a SHA-256 of the seed; a plain small
  multiplier selected contiguous blocks, and a raw seed offset made nearby
  seeds select almost the same rows.
- 2026-11-10 (M1): A source without a metadata YAML is allowed; its
  relationships come only from `PRAGMA foreign_key_list`. A YAML whose
  `source:` names another source raises `MetadataError`.
- 2026-11-10 (M1): `config/settings.yaml` gains a `seed` group (`random_seed`,
  `scale`) so tests can build small copies of the seed sources.
- 2026-11-10 (M1): Tooling: pydantic mypy plugin on; `pyarrow-stubs` for strict
  typing; ruff D401 off (docstrings state what and why); S608 allowed only in
  `connectors/sqlite.py`, where identifiers are validated and quoted.
- 2026-11-10 (M2): Profiling modules are `inference`, `numeric`, `categorical`,
  `temporal`, `profiler`, `card` and `cache` instead of the planned
  `table_stats`/`column_stats` split. The connector keeps one read-only
  connection with a 256 MiB page cache per instance; per-query connections
  made profiling 4x slower on the Windows drive.
- 2026-11-10 (M2): Profile versions are content hashes; `profile_cache` is keyed
  by (source, table, version) so old versions stay loadable. Profiles sample
  with `profiling.sample_seed`, independent of the generation seed.
- 2026-11-10 (M2): The graph has the 15 work nodes plus a `finish` bookkeeping
  node that records the terminal reason. `JobState` adds `approval_errors`,
  `patch` and `route`. A wrapper around each node evaluates the pure router
  from `routing.py` and stores the route; edges follow `state.route`.
- 2026-11-10 (M2): Node errors: `LlmOutputError` ends the job `needs_review`,
  any other `SynthgenError` ends it `failed`; any other exception propagates
  as a crash and the job stays resumable from its last checkpoint.
- 2026-11-10 (M2): Loop counters count per job and are never reset by a new
  plan round. The node running the loop body increments; the router checks
  `counter < cap`. Blocking questions left after the clarification cap end the
  job `needs_review` (hard rule in `CLAUDE.md`).
- 2026-11-10 (M2): Refines A9. Stub fixtures are `<SchemaName>.json` lists of
  `{match, response}`; the stub returns the last entry whose `match` text is in
  the prompt. Selection by content, not call count, keeps resumed jobs stable.
- 2026-11-10 (M2): Approved shares and categories are binding: code copies
  them from the approved plan onto the overlays. The overlay writer and repair
  analyst see column names and types, cards, the plan and overlays, never the
  baseline spec (it holds profile-derived values).
- 2026-11-10 (M2): Child tables get stratified children-per-parent draws nudged
  to an exact total (the target, else parents x mean, rounded). Dry runs size
  children by the mean. Rows inherit their parent's scenario through the
  driving foreign key.
- 2026-11-10 (M2): Known limitation until M5: sensitive columns are generated
  from character masks, so names are random letters and a sensitive date such
  as `date_of_birth` is text that is not a valid date.
- 2026-11-10 (M2): Use cases add `create_job` and `run_job` so the CLI can
  print the job ID before work starts; `resume_job` runs a job from its start
  or its last checkpoint. `Application` is a context manager that closes the
  checkpoint connection.
- 2026-11-10 (M2): T2.11 and T2.12 share one commit (their tests import both).
  Test folders are packages so module names stay unique for mypy.
- 2026-11-10 (M3): Revises A13. The OpenAI-compatible client uses `httpx`
  directly, not the `openai` SDK, so retries, backoff with jitter and
  Retry-After stay in one place and tests use `httpx.MockTransport`. Groq docs
  checked: strict `json_schema` output is supported on `openai/gpt-oss-120b`;
  `anyOf`, `enum`, `$defs`/`$ref` are documented, other validation keywords
  are not, so the strict schema drops them (Pydantic still enforces them);
  structured output cannot be combined with tools; no parallel tool calls.
- 2026-11-10 (M3): `LlmClient.complete` returns the raw reply text with usage
  and latency instead of a parsed model (SPEC 4.1 sketch). `StructuredLlm`
  runs gates 2 and 3, re-prompts with the previous output and a numbered error
  list, logs every call to `llm_calls`, and raises `LlmOutputError` after
  `llm_reprompts` (3) attempts in total. T3.1 and T3.2 share one commit.
- 2026-11-10 (M3): Repair patches may only change generator parameters
  (`/scenarios/i/overrides/j/generator/<param>` and
  `/tables/i/columns/j/generator/<param>`, never `type`); other paths are a
  gate-3 error asking for `plan_change_required`. Gate 3 also requires the
  summary to contain no digits outside `{placeholders}`.
- 2026-11-10 (M3): Revises Q2 layering: `spec` now ranks below `scenarios`
  (`profiling <- spec <- scenarios <- engine <- validation`). The feasibility
  check reads which constructs the spec models offer; `spec` never needs
  `scenarios`. The architecture test enforces the new order.
- 2026-11-10 (M3): Not-expressible scenarios stay in the plan and are shown at
  approval with a reason and the nearest alternative; the overlay writer only
  receives expressible or partial scenarios, and the reporter adds a caveat for
  each skipped one. Feasibility reads available constructs from the spec
  models, so M5 additions become expressible without code changes here.
- 2026-11-10 (M3): Plan expiry is enforced when a job is viewed, resumed or
  decided: a plan older than `plan_expiry_days` cancels the job with the reason
  "plan expired after 7 days without a decision".
- 2026-11-10 (M3): Source handling moved from `app.py` to `sources.py`
  (`SourceCatalog`, ranked with `agents`) to keep `app.py` under 300 lines;
  `Source` moved to `domain/ports.py`.
- 2026-11-11 (M3): Live check against Groq's free tier (8,000 tokens per minute)
  completed on `finance` end to end and planned sensibly on `orders` with no
  templates, but re-prompts do not fit that budget reliably. The user will run
  the live check on the company's Llama model (OpenAI-compatible, no token
  limit). T3.10 is deferred to that run; Groq findings kept as fixes (schema
  rejections re-prompted, unambiguous unions, compact cards, bounded retries).
- 2026-11-11 (M3): New requirement from the user: tables of up to 3,000
  columns. Python's bundled SQLite is compiled with SQLITE_MAX_COLUMN=2000 and
  cannot open wider tables; prebuilt pysqlite3-binary and apsw wheels have the
  same cap. Decision (user): an optional SQLite build with
  SQLITE_MAX_COLUMN=32767, built by `make sqlite-wide` (pysqlite3 0.5.4 static
  build on the SQLite 3.50.4 amalgamation, sources pinned by SHA-256, compiled
  with the `ziglang` pip package so no system compiler or sudo is needed). The
  connector uses it when installed and falls back to `sqlite3` otherwise. The
  rest of the system works in column batches so width is never bounded by one
  SQL statement or one prompt.
- 2026-11-11: Pacing (user): stop for review after each milestone.
- 2026-11-11 (M3): Uniqueness is taken from declared constraints only (primary
  keys, single-column UNIQUE constraints and unique indexes, read from
  `PRAGMA index_list`), never from "all sampled values distinct"; the latter
  made timestamps "unique" and broke valid overlays. Declared-unique text
  columns are generated from masks, not from observed values.
- 2026-11-11 (M3): The Groq live finance run is kept as a recorded fixture set
  (`tests/fixtures/llm/live_finance/`) and replayed offline through every gate;
  the replay found and fixed two bugs (unique names, unique timestamps).
- 2026-11-11 (M3): Signature metrics gained `hour_of_day`, `min` and `max`;
  signature values are a number, text, boolean or one list of numbers or text
  (Groq rejects unions it cannot tell apart, e.g. two list types or integer
  with number), and the strict-schema builder now rejects such unions offline.
- 2026-11-11 (M3): Plans may not contain `typical` scenarios or scenarios that
  only change label columns; label levers next to data levers are accepted and
  shown as "label set by the overlay". The overlay writer is told which levers
  are not expressible and must leave them out.
- 2026-11-11 (M4): Wide tables keep profiling linear in width. With more than
  `profiling.max_fit_columns` numeric columns, only the closed-form fits
  (norm, lognorm, expon, uniform) run and only the best is KS-tested; KDE
  modes and normality tests are skipped. Pairwise statistics (group F) use at
  most `max_pair_columns` columns (dates and booleans first, then the most
  variable) on `pair_sample_size` rows; only strong pairs
  (`strong_association`) are stored. Null co-occurrence looks only at null
  masks, so it also covers constant and sensitive columns.
- 2026-11-11 (M4): Within-table ordering rules (a <= b) are recorded only
  when the two ranges overlap; an order the ranges already imply (birth date
  before signup) adds nothing the marginals do not enforce. Cross-table date
  order (child on or after parent) is recorded per relationship.
- 2026-11-11 (M4): Behaviour (group H) reads the full histories of a seeded
  sample of `behaviour_entities` entities through grouped SQL with an `IN`
  filter; a row sample would stretch the gaps between events. The main
  timestamp is the first date-time column (else the first date), the measure
  is the first float column (else integer), and state columns are categorical
  with 2 to `max_conditional_levels` levels.
- 2026-11-11 (M4): The label column is named by the semantic mapper, which
  runs after profiling. The profiler therefore profiles label candidates
  (group I, and H per class): binary, non-key, non-sensitive columns, boolean
  flags first and then the rarest minority, at most `max_label_columns`. Any
  other label goes through `get_class_separation` on demand. A date-time
  column is also scored on its hour of day and keeps the stronger score.
- 2026-11-11 (M4): Sensitivity lives in `profiling/privacy.py`; internal
  values under `internal_min_count` rows fold in every stored statistic, and
  every profile passes a privacy guard (`PrivacyError` otherwise). The card
  still filters internal values because it is the LLM boundary. The cache is
  the M2 `storage/profile_store.py` plus `profiling/cache.py` (not a new
  `storage/profile_cache.py`); its schema hash now also covers sensitivity
  tags and a `PROFILE_FORMAT` number, so retagged tables and profiles of an
  older shape miss the cache. Storing unchanged content refreshes its age (a
  bug made week-old profiles recompute on every request).
- 2026-11-11 (M4): `profile_table` takes the reference time `now` (future-date
  rates) and the parent tables' metadata (`parents`, for parent date and
  category columns). Sentinel dates are counted but left out of trend,
  seasonality, gaps and growth; ranges over `max_series_days` skip trend and
  seasonality. Growth counts rows per month, or per year past
  `max_series_months`.
- 2026-11-11 (M5): Expressions are split into `spec/expressions.py` (tokenizer,
  recursive-descent parser, type check) and `spec/evaluator.py` (vectorised
  evaluation) to keep modules under 300 lines. The conditional is Python-style
  (`a if cond else b`); `parent.column` reads the driving parent row; units are
  seconds to weeks (no months). Nulls propagate through arithmetic and
  functions; a comparison with a null is false; division by zero gives null.
  Length (1,000 characters) and nesting (40) are capped.
- 2026-11-11 (M5): Generator shapes. `conditional` holds leaf generators only
  (no recursion in the strict schema) and its fallback field is `otherwise`
  (a property named `default` collides with the dropped JSON-schema keyword).
  `foreign_key` keeps one model with nullable `exponent`, `hot_keys`,
  `hot_share`, `group_size`, each required by exactly its distribution; hot
  keys and shared groups are chosen by a seeded hash of the parent keys, so
  every chunk agrees. `distribution` covers the families the profiler fits
  (plus poisson and nbinom) with SciPy parameter order. `boundary` draws
  uniformly from listed points. `text_pool` values are at most 1,000
  characters and `generation.text_pool_max_values` (500) in number.
- 2026-11-11 (M5): `semantic` pools are built per kind with Faker
  (`generation.faker_locale`, `semantic_pool_size` values) seeded from the spec
  seed and the kind; Faker never reads the clock (birth dates fall in
  `generation.birth_dates`). The baseline picks a kind for a sensitive column
  from its profiled format (email, phone, card number) or the words in its
  name (first/last/full name, birth date, city, postcode, street, company);
  otherwise masks. Declared-unique sensitive text stays on masks (with a Luhn
  digit for card numbers) until T5.5 adds counters. This ends the M2
  limitation: names are names and `date_of_birth` is a valid date.
- 2026-11-11 (M5): Columns are generated in dependency order; an overlay may
  read its own column's baseline value (`amount * 3`), and baseline columns
  computed from an overridden column are recomputed on the overlay's rows.
  Override cycles are an overlay error. `generate_dataset` now takes the
  `GenerationSettings`. The spec format version is 2.
- 2026-11-11 (M5): A14 screening of text-pool values against source values of
  sensitive columns needs a lookup the connector does not offer yet; it moves
  to T6.4 with the privacy family.
- 2026-11-11 (M5): The recorded Groq replay (`tests/fixtures/llm/live_finance`)
  gained two hand-written overlays, marked as such in its README: M5 made
  `boundary_transactions_amount` and `volume_skew_transactions_merchant`
  expressible, and the recording never asked the model for them.
- 2026-11-11 (M5): Copula. `TableSpec.correlation` holds the baseline Spearman
  matrix; `ScenarioSpec.correlations` carries correlation_change overrides
  (the feasibility check now keys that construct on the scenario field). Only
  columns with `empirical` or `distribution` generators join a copula, at most
  `generation.max_copula_columns`; the baseline keeps columns whose strongest
  |rho| reaches `generation.copula_min_rho`. `repair_correlations` (in
  `spec/validator.py`, run by the spec_validator node) replaces a matrix that
  is not positive semi-definite with the nearest valid one (Higham's
  alternating projections, `spec/correlation.py`); the engine repairs the
  normal-scale matrix again after `2 sin(pi rho / 6)`.
- 2026-11-11 (M5): Sequences. `ScenarioSpec` gains `cardinality` (children per
  parent row of a child table for the scenario's parents; `children: 0` is
  absence) and `sequences` (`SequenceSpec` in `spec/shapes.py`: child table,
  its driving key as `entity_key`, a date or datetime `time_column`, steps of
  `count` events with a gap range in seconds and optional step columns for a
  state path). The shaped table must hang below the scenario entity through
  driving keys. A sequence fixes its parents' children to the steps' event
  count unless `cardinality` says otherwise; the other parents absorb the
  difference so the table total still matches its target. Events are ordered
  by entity and drawn time; the first keeps its time and each later one
  follows by a gap from its step. `generation.max_sequence_events` caps events
  per entity (1,000).
- 2026-11-11 (M5): Once M5 offers every construct, "not expressible" comes only
  from missing columns or role mismatches. The nearest alternative (SPEC 7.4)
  now also covers role mismatches: the primitive's alternatives are walked to
  the first one the engine offers that suits the lever column's role
  (timing_shift on a measure suggests value_shift). Tests that used
  `sequence` as the unavailable example use a role mismatch, or patch the
  construct list.
- 2026-11-11 (M5): Constraints (`spec/constraints.py`, `TableSpec.constraints`):
  order, range, unique (several columns), not_null and expression. The engine
  draws all columns, then enforces constraints, then adds nulls (SPEC 8.5
  order), so not_null columns are skipped by the null step. A broken order
  swaps the two values (a parent-side order reflects the child above the
  parent value); range clips; unique and expression redraw the free columns
  (and recompute what is derived from them) up to 20 rounds, then fail. The
  baseline adds profiled `a <= b` rules, child-on-or-after-parent dates for
  the driving relationship (`generation.order_min_share`), and makes
  `c = a + b` / `c = a * b` columns derived (one level deep). Overlays and
  sequences run after constraints, so scenario rows may break them on purpose.
- 2026-11-11 (M5): Unique semantic columns get a permuted counter starting at
  the chunk's row offset (inserted before the `@` of an email, else appended),
  so unique emails use the semantic pool; other unique sensitive text keeps
  masks.
- 2026-11-11 (M5): Defects (`ScenarioSpec.defects`: null, out_of_range,
  bad_format, orphan_key, duplicate_key at a `rate` of the scenario's rows in
  a table on or below its entity) run last and are labelled `kind:column` in a
  new `_defect` column on every table (null when clean). They are allowed only
  when the intent has `include_invalid` (`validate_spec(..., allow_defects=)`).
  Integrity checks skip labelled defect rows, and uniqueness is judged among
  clean rows; M6 checks that defect rows really are broken.
- 2026-11-12 (M5): Chunking. `engine/planning.py` plans a table once its parents
  are written: root tables get consecutive chunks; a child table draws
  children per parent for every parent row (stream slot 0 of the table), then
  cuts chunks at parent boundaries so one family always shares a chunk (a
  family larger than the limit gets a chunk of its own). The row limit is
  `chunk_rows`, lowered to `max_cells_per_chunk / columns` for wide tables.
  Chunk i draws from `SeedSequence(seed, spawn_key=(table, i + 1))`, built
  directly (identical to `spawn`). Row offsets give disjoint sequence keys and
  unique counters. Each scenario's exact table total (PLAN A2) is split over
  chunks by largest remainder in proportion to eligible rows; inside a chunk
  rows are picked by seeded hash rank. `engine/chunk.py` runs one chunk end to
  end (parent keys, scenarios and attributes are read for its parent row range
  from the parent's parts) and writes `part-<i>`; `engine/schema.py` holds the
  output columns and `engine/reader.py` reads written tables by row range.
- 2026-11-12 (M5): Workers and recovery. Chunks of one table run in this
  process (`--workers 1`) or a spawn-based `ProcessPoolExecutor` (spawn keeps
  workers free of the parent's threads and connections); tables still run in
  order. A failed chunk is retried with the same stream up to
  `generation.chunk_retries` attempts; each failure removes the chunk's files,
  and parts are written to a `.tmp` name and renamed, so no half-written part
  survives. The manifest is rewritten after every chunk with `complete: false`
  and per-part scenario counts; a rerun whose `run_hash` (spec, run options and
  the generation settings except `workers`) matches skips the parts it lists,
  otherwise the data folder is cleared. `generate_dataset` takes `workers`
  (from the job request) and an injectable chunk runner for tests.
- 2026-11-12 (M5): The scale fixture `tests/fixtures/specs/finance_1m.json` is
  written by hand (not compiled from a profile) and uses most constructs:
  semantic pools with a unique email counter, Luhn card numbers, a copula,
  lognormal and empirical marginals, Zipf merchant keys, derived columns,
  parent-date orders, a range constraint, a card-testing sequence and two
  override scenarios; transactions have a target of exactly 1,000,000 rows.
  The slow test runs it with 2 workers and again with 1, and compares bytes.
- 2026-11-12 (M6): Fidelity reads typical rows (no scenario, no defect), at most
  `validation.fidelity_sample_size` per table with a fixed seed, one column at a
  time (wide tables never load whole). Wasserstein/IQR compares quantile
  functions with the stored quantiles interpolated linearly (what the generator
  samples); as point masses the grid misplaces tail mass, and the finance
  source failed against its own profile (0.195). Checks per column: null-rate
  difference (all non-key columns), KS and W/IQR (numbers with quantile-type
  generators), category JS (categorical generators, unseen values as one extra
  bucket), month/weekday/hour JS (datetime generators). Per table: largest
  Spearman gap over the copula columns, and children-per-parent KS over
  typical parents (skipped when a row target rescales them). Validation needs
  the profiles; the nodes load them.
- 2026-11-12 (M6): Arrow returns NaN for null floats; validation maps nulls to
  None. A parent-side order repair now redraws the child from its own generator
  (up to 20 rounds), then lifts rows that still break to the first value at or
  after the parent, keeping the time of day. The M5 reflection mirrored the hour
  of day (hour-mix JS 0.33 on transactions) and pushed dates years past the
  source range (cards issued in 2034). Constraint passes now repeat until a
  pass changes nothing, since a parent-date repair can undo an earlier order
  between two child dates; contradictory constraints fail with EngineError.
- 2026-11-12 (M6): Refines Q6 to both sides. A distribution check is also
  skipped when the profile behind it rests on fewer than `min_fidelity_rows`
  source rows (`sample_rows`, the pairwise sample, or the profiled parents):
  at 2% seed scale a 12-bin month mix of ~600 source rows differs from a fresh
  sample by about 0.12 through noise alone.
- 2026-11-12 (M6): Integrity reads one column at a time (`OutputData`) and adds:
  formats (ISO dates; pattern masks with Luhn digits; semantic emails and card
  numbers) checked only on rows the baseline generator made (rows of scenarios
  that rewrite the column follow their overlay); every table constraint
  (`constraint_<kind>`, orders against a parent join the driving key, which
  covers parent date before child date); the written Arrow type from the part
  schema. A text that is not a date counts as missing in comparisons (the
  format check reports it).
- 2026-11-12 (M6): The recorded Groq replay gained a hand-written
  `SpecPatch.json` (marked in its README): the recorded overlays put
  transactions before their card's issue date, which M6 integrity now catches,
  and the patch moves the overlay time windows after the last issue date. The
  replay therefore includes one repair round.
- 2026-11-12 (M6): Scenario family. `present` (at least one entity row) and
  `share` (exact count from the approved share over eligible rows of the
  entity table, tolerance max(10% relative, 1 row), PLAN A2); `signature` per
  plan assertion (pass rate over rows for `value`/`hour_of_day`, over scenario
  entities for aggregates, which group rows by walking driving keys up to the
  scenario entity; `window_hours` takes the best window for count, sum and
  distinct count; event times come from the scenario's sequence time column,
  else the first date-time, else the first date); `separation` (KS against
  typical rows on numeric and time signature columns, skipping columns the
  scenario sets as labels); `near_miss` (near-miss rows matching every
  row-level assertion of a target's signature on the same table); `labels`
  (scenario rows carry their label values) and `label_counts` (rows per
  scenario equal the manifest). Text compares as text.
- 2026-11-12 (M6): The recorded replay's `volume_skew_transactions_merchant`
  signature (at least 160,000 rows per transaction) cannot be met; the replay
  test removes it at approval and its hand-written overlay is gone.
- 2026-11-12 (M6): Privacy family (`validation/privacy.py`). Lookups use the
  connector's parameterised `aggregate` with `IN` batches of
  `validation.privacy_lookup_batch`; the validation nodes open the source.
  `exact_match` samples `privacy_sample_size` values per sensitive column. Refines
  Q7: a column is identifier-like when its cardinality ratio reaches
  `privacy_identifier_min_ratio`, it is not a date, and its semantic kind is not
  a small vocabulary (names, city, postcode, company, birth date). At 2% seed
  scale first names reach ratio 0.6 and matched 607 times by chance. Others
  report matches as info. `row_hash` hashes non-key, non-driving, non-foreign-key
  columns against the profiling sample (same seed and size) and is info when the
  profiled combinations are fewer than the sample. `suppressed_values` flags
  internal-column values outside the profile's kept list that the source holds
  in fewer than `internal_min_count` rows. `text_pool_values` (A14) looks every
  model-written pool value up in every sensitive text column, traced to the
  scenario. Without an open source privacy reports one info row.
- 2026-11-12 (M6): `validation_report.json` holds `passed`, counts per family
  and status, and every row (check, table, column, metric, value, threshold,
  status, scenario, detail). Failure tracing (A5) covers integrity and every
  privacy check: a failure names the scenarios of its failing rows only when
  none of them is typical, so routing sends it to repair; otherwise the job
  fails. Scenario-family rows always name their scenario; fidelity rows never.
- 2026-11-12 (M6): The patch allow-list now covers everything SPEC 5 lists:
  generator parameters of overrides, baseline columns and sequence step
  columns (clip bounds and conditional cases are parameters), and sequence
  step counts and gaps. The repair analyst (prompt v2) also sees the approved
  scenarios with their signatures.

## 4. Milestones

### M0 Scaffold

- **Goal:** an empty, fully tooled project where `make check` passes.
- **Spec sections:** `CLAUDE.md` (stack, structure, rules), SPEC 2, 9, 10, 13.
- **Tasks:**
  - [x] T0.1 `git init`; `.gitignore` (`data/`, `output/`, `.env`, caches,
    `.venv/`); `.env.example` (`LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_MODEL`,
    `LLM_API_KEY`). Proof: `git check-ignore data/x output/x .env` lists all three.
  - [x] T0.2 `pyproject.toml` (Python ≥ 3.11, dependencies, ruff, mypy strict,
    pytest), `uv.lock`, `Makefile` (`setup`, `check`, `test`, `format`).
    Proof: `make setup` and `make check` succeed.
  - [x] T0.3 Package skeleton `src/synthgen/**/__init__.py` per the repo layout,
    `tests/{unit,integration,golden,fixtures}/`; architecture test
    `tests/unit/test_architecture.py` (AST scan: `domain` imports nothing from
    the project, only `agents` imports `langgraph`/LLM libs, only `app` imports
    `agents`, only `main.py` imports `cli`, no `eval`/`exec`/`importlib`).
  - [x] T0.4 `config/settings.yaml` (groups llm, profiling, generation, agent,
    validation, paths), `src/synthgen/config.py`, minimal
    `src/synthgen/domain/errors.py` (`SynthgenError`, `ConfigError`).
    Tests: `tests/unit/test_config.py` (defaults load, env override, invalid
    value raises `ConfigError`).
  - [x] T0.5 `main.py` and `src/synthgen/cli/__init__.py` (Typer root app).
    Tests: `tests/unit/cli/test_help.py` (`--help` exits 0).
- **Acceptance:** `make check` passes; `uv run python main.py --help` exits 0.

### M1 Foundations

- **Goal:** typed domain, state store, read-only SQLite access, metadata and
  seed sources, so `seed` and `tables` work.
- **Spec sections:** 2, 3.2, 3.6, 5, 8.1, 8.2, 10, 11.
- **Tasks:**
  - [x] T1.1 Job and metadata models: `domain/job.py` (`UserRequest`,
    `JobStatus`, `LoopCounters`, `QuestionAnswer`, `Approval`),
    `domain/metadata.py` (`TableMeta`, `ColumnMeta`, `Relationship`,
    `TableSet`), `domain/errors.py` (full typed error set), `domain/ports.py`
    (per Q2). Tests: `tests/unit/domain/test_job.py`, `test_metadata.py`.
  - [x] T1.2 LLM output schemas and enums: `domain/intent.py` (`TestIntent`),
    `domain/semantic.py` (`SemanticMap`), `domain/plan.py` (`ScenarioPlan`,
    signature assertions), `domain/patch.py` (`SpecPatch`),
    `domain/summary.py`, `domain/report.py` (`ValidationReport`). Tests:
    `tests/unit/domain/test_llm_schemas.py` (round trip; generated JSON schema
    is strict-mode compliant: all fields required, nullable unions,
    `additionalProperties: false`, enums).
  - [x] T1.3 State store: `storage/db.py` (schema for `jobs`, `job_artifacts`,
    `approvals`, `llm_calls`, `profile_cache`), `storage/job_store.py`,
    `storage/artifact_store.py` (versioned JSON). Tests:
    `tests/unit/storage/test_job_store.py`, `test_artifact_store.py`.
  - [x] T1.4 SQLite connector: `connectors/sqlite.py` (read-only URI,
    `get_columns`, `row_count`, `aggregate` with whitelisted functions and
    parameterised filters, identifier validation and quoting, seeded
    hash-modulo `sample` returning `pa.Table`). Tests:
    `tests/unit/connectors/test_sqlite.py` (write attempt fails, unknown
    identifier rejected, sample deterministic, full read when small).
  - [x] T1.5 Metadata provider and dependency graph: `metadata/provider.py`
    (YAML + `PRAGMA foreign_key_list` fallback), `metadata/graph.py` (add
    parents, topological sort, self-reference second pass, mutual reference
    broken at nullable key). Tests: `tests/unit/metadata/test_provider.py`,
    `test_graph.py`.
  - [x] T1.6 Seed `finance.db`: `seed/common.py`, `seed/finance.py`
    (customers, accounts, cards, merchants, transactions; log-normal amounts,
    Zipf keys, hour/weekday patterns, correlations, nulls, date-order rules,
    `is_fraud` ≈ 0.5%; ~200,000 transactions). Tests:
    `tests/unit/seed/test_finance.py` (small scale: counts, FK integrity, fraud
    rate band, determinism).
  - [x] T1.7 Seed `orders.db`: `seed/orders.py` (customers, products, orders,
    order_items, shipments; ~200,000 order_items). Tests:
    `tests/unit/seed/test_orders.py`.
  - [x] T1.8 Hand-written metadata: `config/metadata/finance.yaml`,
    `config/metadata/orders.yaml` (descriptions, sensitivity tags,
    relationships). Tests: `tests/integration/test_metadata_matches_seed.py`.
  - [x] T1.9 Use cases and CLI: `app.py` (composition root, `list_tables`,
    seed use case), `cli/commands.py` (`seed`, `tables`), `cli/render.py`.
    Tests: `tests/integration/test_cli_seed_tables.py`.
- **Acceptance:** `python main.py seed` creates `data/finance.db` and
  `data/orders.db`; `python main.py tables --source finance` (and `orders`)
  lists tables and relationships; `make check` passes.

### M2 Walking skeleton

- **Goal:** all 15 nodes wired end to end on the stub LLM, with basic
  profiling, generation and validation, interrupts, checkpoints and resume.
- **Spec sections:** 3 (all), 4.1, 5, 8.3 (groups A–E core, `LlmSafeProfile`),
  8.4, 8.5, 8.6 (integrity, row count), 9, 10.
- **Tasks:**
  - [x] T2.1 Profile models and groups A, B: `profiling/models.py` (stats carry
    source `sql`/`sample`/`derived`), `profiling/table_stats.py`,
    `profiling/column_stats.py`. Tests: `tests/unit/profiling/test_table_stats.py`,
    `test_column_stats.py`.
  - [x] T2.2 Core of C, D, E: `profiling/numeric.py` (SQL moments,
    percentiles, 50-bin histogram, 1,001-point quantiles),
    `profiling/categorical.py` (frequencies, entropy, lengths, masks),
    `profiling/temporal.py` (min/max/span, hour/weekday/month counts). Tests per
    module.
  - [x] T2.3 Profiler orchestration and cards: `profiling/profiler.py`,
    `profiling/card.py` (`LlmSafeProfile`), `storage/profile_store.py`
    (versioned store). Tests: `tests/unit/profiling/test_card.py` (no sensitive
    source value appears in any card).
  - [x] T2.4 Spec models: `spec/models.py` (`GenerationSpec`, `TableSpec`,
    `ColumnSpec`, `ScenarioSpec`, `SequenceSpec`, `Constraint`, generator
    tagged union for the generators built in M2). Tests:
    `tests/unit/spec/test_models.py`.
  - [x] T2.5 Baseline, merge, validator (basic): `spec/baseline.py`,
    `spec/merge.py`, `spec/validator.py` (tables/columns exist, generator suits
    type, weights and shares, FK order, row limits). Tests per module.
  - [x] T2.6 Basic generators and keys: `engine/generators.py` (sequence,
    uuid, empirical, categorical, datetime, pattern, constant),
    `engine/keys.py` (PK ranges, uniform FK). Tests:
    `tests/unit/engine/test_generators.py`, `test_keys.py`.
  - [x] T2.7 Scenarios, writer, runner (single chunk): `engine/scenarios.py`
    (exact allocation, `_scenario`, `_category`, label columns, overlays),
    `engine/writer.py` (PyArrow zstd, manifest), `engine/runner.py`. Tests:
    `tests/unit/engine/test_runner.py` (byte-identical rerun, FKs resolve).
  - [x] T2.8 Basic validator: `validation/integrity.py` (PK unique, FK
    resolves, not-null, types), `validation/fidelity.py` (row count),
    `validation/report.py`. Tests with injected failures.
  - [x] T2.9 LLM protocol and stub: `agents/llm/client.py` (`LlmClient`,
    `Message`, `ToolDef`, `ToolTurn`), `agents/llm/stub.py`,
    `tests/fixtures/llm/skeleton/`. Tests: `tests/unit/agents/test_stub.py`.
  - [x] T2.10 State and routing: `agents/state.py` (`JobState`),
    `agents/routing.py` (every row of SPEC 3.3, caps from settings). Tests:
    `tests/unit/agents/test_routing.py` (one case per table row, every cap).
  - [x] T2.11 Code nodes: `agents/nodes/{context_builder,profiler,
    baseline_compiler,spec_validator,dry_run,validator,full_generation,
    final_validation}.py`. Tests: `tests/unit/agents/nodes/test_code_nodes.py`.
  - [x] T2.12 LLM nodes and human gates (minimal prompts):
    `agents/nodes/{intent_analyst,semantic_mapper,scenario_planner,
    approval_gate,overlay_writer,repair_analyst,reporter}.py`, clarification
    and approval interrupts. Tests: `tests/unit/agents/nodes/test_llm_nodes.py`.
  - [x] T2.13 Graph and use cases: `agents/graph.py` (SQLite checkpointer,
    thread ID = job ID), `app.py` (`start_job`, `get_status`, `get_pending`,
    `answer_clarification`, `decide_plan`, `get_report`, `get_preview`,
    `profile_tables`). Tests: `tests/integration/test_graph_happy.py`,
    `test_graph_failures.py` (cap → `needs_review`, reject → `cancelled`,
    integrity failure → `failed`, clarification round trip).
  - [x] T2.14 CLI job commands: `cli/commands.py` (`profile`, `run`, `resume`,
    `status`, `report`, `preview`), `cli/prompts.py` (approval and question
    prompts), coverage gate in `Makefile` (A11). Tests:
    `tests/integration/test_resume.py` (forced stop mid-graph, resume gives the
    same output hash), `tests/integration/test_cli_run.py`.
- **Acceptance:** with `LLM_PROVIDER=stub`, `python main.py run --source
  finance --tables customers,accounts --request "..." --auto-approve` ends
  `completed` with Parquet, manifest and report; a run stopped mid-graph
  completes with `python main.py resume <job_id>`; `make check` passes.

### M3 Real agents

- **Goal:** real LLM roles with structured output, three gates, planner tools,
  primitives and feasibility, producing approved plans and valid overlays.
- **Spec sections:** 4, 5, 6, 7, 3.3–3.5.
- **Tasks:**
  - [x] T3.1 OpenAI-compatible client: `agents/llm/openai_compatible.py`,
    `agents/llm/retry.py` (backoff with jitter, `Retry-After`, rate limiter).
    Tests: `tests/unit/agents/llm/test_openai_compatible.py` (mocked HTTP
    transport only).
  - [x] T3.2 Structured output and gates: `agents/llm/structured.py` (strict
    schema, JSON-mode fallback, numbered-error re-prompt, cap 3),
    `storage/llm_call_log.py`. Tests: `test_structured.py` (bad JSON, schema
    failure, meaning failure, cap → `needs_review`, every call logged, no
    secret in log).
  - [x] T3.3 Prompt templates: `agents/prompts/{intent_analyst,semantic_mapper,
    scenario_planner,overlay_writer,repair_analyst,reporter}.md`,
    `agents/prompts/loader.py` (version header, four parts, untrusted blocks).
    Tests: `tests/unit/agents/test_prompts.py`.
  - [x] T3.4 Intent analyst and semantic mapper (meaning gates, clarification,
    batching above 15 tables). Tests with stub fixtures, including invalid
    outputs that are re-prompted.
  - [x] T3.5 Planner tools: `agents/tools/registry.py`,
    `agents/tools/profile_tools.py` (seven tools, argument validation,
    ~2,000-token cap, per-job cache, sensitive masks). Tests:
    `tests/unit/agents/tools/test_tools.py`.
  - [x] T3.6 Scenarios package: `scenarios/primitives.py` (14 primitives →
    spec constructs), `scenarios/profile_derived.py`,
    `scenarios/feasibility.py`, `scenarios/templates/loader.py`,
    `scenarios/templates/card_fraud/*.yaml`. Tests: `tests/unit/scenarios/`.
  - [x] T3.7 Scenario planner node: tool turns then one structured call, tool
    cap (Q5), meaning gate, feasibility attached. Tests: stub tool-call flow.
  - [x] T3.8 Overlay writer, repair analyst, reporter: `spec/patch.py` (JSON
    pointer apply, allow-list, `plan_change_required` → approval), reporter
    with code-inserted numbers. Tests per node.
  - [x] T3.9 Approval UX: plan table (scenario, category, share, feasibility,
    signature, template or model-proposed), edit shares, remove, reject with
    feedback, plan expiry after 7 days. Tests: `tests/unit/cli/test_prompts.py`.
  - [ ] T3.10 Live check and fixture capture: one real request per source;
    record responses into `tests/fixtures/llm/` (no secrets). DEFERRED to the
    user's run on the company Llama model (see Decisions 2026-11-11).
  - [x] T3.11 Wide SQLite driver: `tools/build_wide_sqlite.sh`, `make
    sqlite-wide`, driver selection in `connectors/`, a clear error when a table
    exceeds the active driver's column limit. Tests:
    `tests/unit/connectors/test_wide.py` (3,000 columns, skipped without the
    wide build; stdlib limit error).
  - [x] T3.12 Wide-table profiling: aggregates and samples in column batches
    (`profiling.max_select_columns`). Tests: 1,500 columns on stdlib SQLite,
    3,000 columns on the wide build.
  - [x] T3.13 Wide-table prompts: mapper batches columns within a table
    (`profiling.mapper_batch_columns`); planner, overlay writer and repair
    analyst receive role-bearing and lever columns only once a table exceeds
    `profiling.card_max_columns`. Tests with a 3,000-column card.
  - [x] T3.14 Llama-compatible output: `llm.output_mode` (strict
    `json_schema`, `json_object`, or prompt-only JSON) and a documented
    no-tools mode (`planner_tool_calls = 0`). Tests on the mocked transport.
- **Acceptance:** with `LLM_API_KEY` set, a real request on `finance` and one on
  `orders` each yield an approved plan and overlays that pass `spec_validator`;
  `make check` passes on the stub only.

### M4 Full profiler

- **Goal:** complete groups C–I, sensitivity classes, cache and cards.
- **Spec sections:** 8.3, 6, 4.4 (card size).
- **Tasks:**
  - [x] T4.1 Numeric C complete: `profiling/numeric.py`, `profiling/fitting.py`
    (AIC + KS ranking, normality tests, KDE modes, outliers, constant step,
    Benford, Poisson/NB). Tests recover known parameters.
  - [x] T4.2 Categorical and text D complete: `profiling/categorical.py`,
    `profiling/text.py` (Zipf exponent, masks, prefixes/suffixes, Luhn/email
    validity, case, words). Tests.
  - [x] T4.3 Date and time E complete: `profiling/temporal.py` (granularity,
    `linregress` trend, periodogram, gaps, future/sentinel, weekend and
    business hours). Tests.
  - [x] T4.4 Cross-column F: `profiling/cross_column.py`. Tests (known
    correlations, FDs, a ≤ b, c = a + b, null co-occurrence). Wide tables:
    pairwise statistics limited to `profiling.max_pair_columns` columns chosen
    by role and variance, so 3,000 columns never mean 4.5 million pairs.
  - [x] T4.5 Cross-table G: `profiling/relationships.py`. Tests.
  - [x] T4.6 Behaviour H: `profiling/behaviour.py`. Tests.
  - [x] T4.7 Label-stratified I: `profiling/stratified.py`. Tests.
  - [x] T4.8 Sensitivity and cache: `profiling/privacy.py` (classes, folding),
    `storage/profile_cache.py` (schema hash, 7-day age, `--refresh`), full
    cards. Tests: `tests/unit/profiling/test_privacy.py`,
    `tests/unit/storage/test_profile_cache.py`.
  - [x] T4.9 Recovery test: `tests/integration/test_profile_recovers_seed.py`
    (log-normal amounts, Zipf keys, hour/weekday mix, fraud rate,
    correlations, date-order rules).
- **Acceptance:** `python main.py profile --source finance --tables
  transactions` prints cards; the recovery test passes; `make check` passes.

### M5 Full engine

- **Goal:** every generator and primitive, copula, sequences, chunking and
  workers, reproducible at 1,000,000 rows.
- **Spec sections:** 7.2, 8.4, 8.5, 10 (chunk errors).
- **Tasks:**
  - [x] T5.1 Expression grammar: `spec/expressions.py` (tokenizer, parser,
    vectorised evaluator; arithmetic, comparisons, if/else, min, max, round,
    date add/diff, concat, column refs). Tests incl. rejected input.
  - [x] T5.2 Remaining generators: `engine/generators/` (distribution,
    conditional, boundary, derived, pattern with check digit, semantic via
    seeded Faker pools, text_pool, FK Zipf/weighted/shared group). Tests.
  - [x] T5.3 Gaussian copula: `engine/copula.py`; nearest-PSD repair in
    `spec/validator.py`. Tests: Spearman matrix recovered. Wide tables: the
    copula covers at most `generation.max_copula_columns` correlated columns.
  - [x] T5.4 Sequences and state paths: `engine/sequences.py`. Tests.
  - [x] T5.5 Constraints, nulls, defects: `engine/constraints.py`,
    `engine/defects.py` (`_defect`). Tests.
  - [x] T5.6 Primitive coverage: `tests/unit/engine/test_primitives.py` (one
    test per primitive, spec construct → observable effect).
  - [x] T5.7 Chunk planning: `engine/planning.py` (sizes, chunks, child chunks
    by parent key range, `SeedSequence.spawn`, disjoint key ranges, parent
    attributes from Parquet). Chunk rows also bounded by
    `generation.max_cells_per_chunk` / columns, so 3,000-column tables stay in
    memory. Tests.
  - [x] T5.8 Workers and chunk recovery: `engine/runner.py` (process pool,
    retry 3 with same seed, skip chunks in manifest, remove partial files).
    Tests: `--workers 1` vs `2` byte-identical; resume skips done chunks.
  - [x] T5.9 Scale test: `tests/fixtures/specs/finance_1m.json`,
    `tests/integration/test_engine_scale.py` (`slow`).
- **Acceptance:** `make test-slow` generates 1,000,000 rows from the
  hand-written spec and a rerun is byte-identical; `make check` passes.

### M6 Full validator and repair

- **Goal:** all four validation families, the report, and a working repair loop.
- **Spec sections:** 8.6, 3.3, 5 (patch allow-list), 10.
- **Tasks:**
  - [x] T6.1 Fidelity complete: `validation/fidelity.py` (KS, Wasserstein/IQR,
    JS category, null diff, Spearman diff, temporal JS, children-per-parent
    KS, row count; typical rows only; small-sample rule per Q6). Validation
    reads Parquet in column batches for wide tables. Tests.
  - [x] T6.2 Integrity complete: formats, constraints, parent date before child
    date. Tests.
  - [x] T6.3 Scenario family: `validation/signature.py`,
    `validation/scenario.py` (share, presence, signature ≥ 99%, KS ≥ 0.30,
    near-miss = 0, labels vs manifest). Tests.
  - [x] T6.4 Privacy family: `validation/privacy.py` (per Q7). Tests.
  - [x] T6.5 Report and failure tracing: `validation/report.py`,
    `validation_report.json`, overlay tracing for routing (A5). Tests.
  - [x] T6.6 Repair loop: repair prompt fed failing report rows; integration
    tests `tests/integration/test_repair.py` (seeded failure repaired; injected
    defects caught; cap → `needs_review`).
- **Acceptance:** typical output passes fidelity; injected defects are caught;
  a seeded failure is repaired within the cap; coverage ≥ 85% on `domain`,
  `spec`, `engine`, `validation`; `make check` passes.

### M7 Evaluation and demo

- **Goal:** measurable quality on a golden set, plus docs and a demo.
- **Spec sections:** 11, 9, 14.
- **Tasks:**
  - [ ] T7.1 Golden set: `tests/golden/requests/*.yaml` (15–20 requests, both
    sources, all five test types, expected intent, must-have scenarios),
    `tests/golden/roles/{finance,orders}.yaml` (hand-labelled roles).
  - [ ] T7.2 Eval: `app.run_eval`, `cli` `eval` command, metrics (intent
    accuracy, role accuracy, plan coverage, first-pass spec validity, repair
    loops, dry-run pass rate, tokens, latency). Tests on the stub.
  - [ ] T7.3 Scale runs: golden set on both sources at 100, 100,000 and
    1,000,000 rows; fix findings.
  - [ ] T7.4 `README.md` (setup, commands, demo; under two pages), `make demo`.
  - [ ] T7.5 Definition-of-done audit: scan outputs and `llm_calls` for
    sensitive source values; `orders.db` plan with no templates installed;
    interrupt-at-every-node resume test `tests/integration/test_resume_any_node.py`.
- **Acceptance:** `python main.py eval` runs the golden set on both sources at
  all three volumes and prints metrics; every item in section 6 holds.

## 5. Risks

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| SQLite locking and slow I/O on the Windows drive (`/mnt/d`, WSL2 DrvFs); WAL may not work | Medium | High | Paths in settings with env override (`SYNTHGEN_DATA_DIR`, `SYNTHGEN_OUTPUT_DIR`) so data can live on the Linux filesystem; avoid WAL mode |
| Groq strict structured output or tool calling differs from docs for `gpt-oss-120b` | Medium | High | Check docs at M3; JSON-mode fallback with Pydantic parse and re-prompt (SPEC 4.2) |
| Groq rate limits and cost during golden runs | High | Medium | Client-side rate limiter, backoff, per-job tool-result cache, stub for all tests |
| LangGraph 1.x API (interrupts, SQLite checkpointer) differs from memory | Medium | Medium | Read installed docs before use; pin versions in `uv.lock` |
| `mypy --strict` with SciPy/pandas/Faker stubs | High | Low | `pandas-stubs`; narrow per-module overrides with a comment; typed wrappers at the boundary |
| Byte-identical Parquet broken by worker order, float summation order or library metadata | Medium | High | Per-chunk streams via `SeedSequence.spawn`, deterministic chunk naming, fixed writer options, determinism test in CI |
| 1,000,000-row runs exceed 7 GB RAM or take too long | Medium | Medium | Chunking (250,000 rows), streamed Parquet writes, sampled validation where the spec allows |
| Statistical checks are noisy at small volumes | High | Medium | Q6 rule; effect-size thresholds from config |
| LLM proposes infeasible or vague scenarios | Medium | Medium | Feasibility check, meaning gate, profile-quantified levers, approval gate |
| Sensitive values leak into prompts or output | Low | High | Only `LlmSafeProfile` reaches agents; property tests on cards; privacy family; DoD audit scan |
| Scope creep beyond POC | Medium | Medium | Out-of-scope list in `CLAUDE.md`; ask before adding anything |

## 6. Definition of done

- `make check` passes: format, lint, strict typing, all tests.
- Coverage is at least 85% on `domain`, `spec`, `engine`, `validation`.
- The dependency rule in `CLAUDE.md` holds; only `agents` imports LangGraph.
- A job can be interrupted at any node and resumed to the same result.
- Same seed, spec and profile version give byte-identical Parquet.
- No source value from a sensitive column appears in any output or LLM prompt.
- The system produces a sensible plan for a request on `orders.db` with no
  template installed.
- `README.md` explains setup, the commands and the demo in under two pages.

## 7. Progress log

- 2026-11-10 M0 Scaffold done: `make check` passes (ruff format, ruff, mypy
  strict, 49 tests); `main.py --help` exits 0. Not committed (user choice).
- 2026-11-10 M1 Foundations done: `make check` passes (228 tests); `seed`
  builds finance.db (200,000 transactions, 0.5% fraud) and orders.db (200,030
  order lines) in ~20 s; `tables` lists both with relationships, including
  the PRAGMA-derived `order_items.product_id`. Committed per task.
- 2026-11-10 M2 Walking skeleton done: `make check` passes (457 tests, no
  warnings, 98% coverage on domain/spec/engine/validation). A stub run on
  finance customers,accounts completes; a job killed with SIGKILL during final
  validation completed with `resume` (1,249,975 rows, 0 of 43 checks failed).
- 2026-11-11 M3 Real agents done except T3.10 (live check deferred to the
  user's Llama run): `make check` passes (585 tests, 98% coverage). Live on
  Groq gpt-oss-120b: a finance card-fraud job completed end to end (14,921
  rows, 0 of 104 checks failed) and an orders late-delivery job with no
  templates produced a sensible plan. Added wide tables (3,000 columns via
  `make sqlite-wide`) and Llama-compatible output modes (T3.11-T3.14).
- 2026-11-11 M4 Full profiler done: `make check` passes (668 tests, no
  warnings, 98% coverage). Groups C-I, privacy classes with folding and a
  guard, cache reuse rules tested. The recovery test on the finance seed finds
  log-normal amounts and incomes, Zipf keys (0.95 for 0.9, 0.84 for 0.7), the
  hour and weekday mix (total variation 0.013 and 0.009), the exact fraud
  count, income-score Spearman 0.83, and date orders. `profile --source
  finance --tables transactions` profiles 200,000 rows in about 15 s; a
  3,000-column table profiles in about 17 s.
- 2026-11-12 M5 Full engine done: `make check` passes (878 tests, 97% coverage
  on domain/spec/engine/validation). All 14 generators, the expression
  grammar, the Gaussian copula with nearest-valid repair, sequences and state
  paths, per-scenario cardinality, constraints, unique counters and opt-in
  labelled defects; one test per primitive. Chunks are planned per table
  (families never split), run on 1 or more spawn workers with same-seed
  retries, and a restart skips finished chunks. `make test-slow` generates
  1,000,000 transactions (plus parents) from `finance_1m.json` with 2 workers
  and again with 1: identical bytes, 41 s for both runs, about 0.6 GB peak in
  the main process. The recorded Groq replay gained two hand-written overlays
  for scenarios M5 made expressible (marked in its README).
- 2026-11-12 M6 Full validator and repair done, with one open issue: `make
  check` passes (961 tests, 97% coverage on domain/spec/engine/validation) and
  `make test-slow` still passes (41 s). All four families run on dry runs and
  final output; failures are traced to overlays for repair; the repair loop
  fixes a seeded signature failure and an overlay that broke a parent-date
  order, labelled defects pass integrity, and an unfixable scenario stops at
  the cap. On the 2% seed every stub job completes. Open: at full seed scale
  (20,000 customers with accounts, cards and transactions) 233 of 234 final
  checks pass; `cards.issued_date` month mix is 0.1034 against 0.10, because
  repairing "issued on or after the account opened" truncates independently
  drawn dates toward the end of the range. Fixing it needs dates drawn
  relative to the parent (a SPEC change); awaiting a decision.

