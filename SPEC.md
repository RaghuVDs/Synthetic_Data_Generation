# SPEC: Synthetic Data Agent (POC)

This is the build specification. `CLAUDE.md` holds the rules; this file holds the
design. Section 12 defines `PLAN.md`. Section 13 lists the milestones.

---

## 1. Goal and flow

A tester selects tables in a SQLite source and describes a test in plain
language. The system returns labelled synthetic Parquet data that covers the
test, at any volume from 100 to millions of rows.

Flow: request → understand intent → profile tables → plan scenarios → user
approves → compile spec → dry run on a sample → validate → repair if needed →
full generation → final validation → report.

The design must be generic. It may not assume a domain. Fraud on card tables is
one demo; churn, late delivery or pipeline negative-testing must work through
the same code.

## 2. POC decisions

| Topic | Decision |
| --- | --- |
| Source database | SQLite files in `data/`, opened read-only |
| State | `data/state.db` (jobs, artifacts, approvals, LLM calls, profile cache) |
| Checkpoints | LangGraph SQLite checkpointer in `data/checkpoints.db`, thread ID = job ID |
| Output | `output/jobs/<job_id>/` on local disk |
| Interface | CLI through `main.py`; approval and questions in the terminal |
| LLM | OpenAI-compatible endpoint; Groq `openai/gpt-oss-120b`; stub in tests |
| Volume | Must pass at 100, 100,000 and 1,000,000 rows; 10,000,000 is a stretch |
| Parallelism | In-process by default; optional process pool via `--workers` |

Use-case functions in `app.py` mirror a future REST API one to one:
`start_job`, `get_status`, `get_pending`, `answer_clarification`, `decide_plan`,
`get_report`, `get_preview`, `list_tables`, `profile_tables`.

---

## 3. Agent graph

### 3.1 Nodes

Fifteen nodes: six LLM roles, eight code steps, one human gate.

| # | Node | Run by | Does |
| --- | --- | --- | --- |
| 1 | intent_analyst | LLM | Request text → `TestIntent`. May ask the user a question. |
| 2 | context_builder | Code | Load metadata, add required parent tables, order tables. |
| 3 | profiler | Code | Compute or load cached profiles. |
| 4 | semantic_mapper | LLM | Profile cards → `SemanticMap` (column roles, rules). |
| 5 | scenario_planner | LLM + tools | Build `ScenarioPlan`. Then code runs the feasibility check. |
| 6 | approval_gate | Human | Approve, edit or reject the plan. |
| 7 | baseline_compiler | Code | Profile → baseline spec. No LLM. |
| 8 | overlay_writer | LLM | Approved scenarios → `ScenarioSpec` overlays. |
| 9 | spec_validator | Code | Schema and semantic checks on the merged spec. |
| 10 | dry_run | Code | Generate a sample (up to 5,000 root entities). |
| 11 | validator | Code | Run all validation families on the sample. |
| 12 | repair_analyst | LLM | Failed checks → `SpecPatch`. |
| 13 | full_generation | Code | Generate all rows, write Parquet. |
| 14 | final_validation | Code | Run all validation families on the output. |
| 15 | reporter | LLM | Short summary. Code inserts every number. |

### 3.2 State

One `TypedDict`, checkpointed after every node. It holds small structures and
references. Profiles and specs stay in the stores.

```python
class JobState(TypedDict):
    job_id: str
    request: UserRequest              # source, tables, text, row targets, options
    intent: TestIntent | None
    clarifications: list[QuestionAnswer]
    table_set: TableSet | None        # selected + added tables, generation order
    profile_refs: dict[str, str]      # table -> profile version
    semantic_map: SemanticMap | None
    plan: ScenarioPlan | None
    approval: Approval | None
    overlays: list[ScenarioSpec]
    spec_ref: str | None              # pointer to merged spec in artifact store
    spec_errors: list[SpecError]
    dry_run_report: ValidationReport | None
    final_report: ValidationReport | None
    counters: LoopCounters            # clarify, plan_rounds, spec_repair, validation_repair
    status: JobStatus
    error: str | None
```

### 3.3 Routing

All edges are evaluated by pure functions in `agents/routing.py`.

| After | Condition | Next |
| --- | --- | --- |
| intent_analyst | blocking ambiguity and `clarify < 2` | pause for answer → intent_analyst |
| intent_analyst | otherwise | context_builder |
| scenario_planner | always | approval_gate (pause) |
| approval_gate | approved, or edit passes validation | baseline_compiler |
| approval_gate | rejected with feedback and `plan_rounds < 3` | scenario_planner |
| approval_gate | rejected without feedback | end `cancelled` |
| spec_validator | no errors | dry_run |
| spec_validator | errors and `spec_repair < 3` | overlay_writer (with errors) |
| validator | all pass | full_generation |
| validator | fidelity or scenario failures and `validation_repair < 3` | repair_analyst → spec_validator |
| validator | integrity or privacy failure traced to an overlay and `validation_repair < 3` | repair_analyst → spec_validator |
| validator | any other integrity or privacy failure | end `failed` |
| any loop | cap reached | end `needs_review` |
| final_validation | all pass | reporter → end `completed` |
| final_validation | any failure | end `needs_review` |

### 3.4 Caps

Clarification 2 rounds. Plan revisions 3. Planner tool calls 8. Spec repair 3.
Validation repair 3. LLM re-prompts per call 3. All in `config/settings.yaml`.

### 3.5 Pauses and resume

- Clarification and approval use LangGraph interrupts. The CLI renders the
  question or plan, collects the answer, and resumes the graph.
- `python main.py resume <job_id>` continues any paused or crashed job from its
  last checkpoint.
- `--auto-approve` approves the plan without a prompt. Use it in tests and eval.

### 3.6 Job statuses

`queued`, `analysing`, `awaiting_clarification`, `profiling`, `planning`,
`awaiting_approval`, `compiling`, `dry_run`, `repairing`, `generating`,
`validating`, then `completed`, `needs_review`, `failed` or `cancelled`.

---

## 4. LLM layer

### 4.1 Client

```python
class LlmClient(Protocol):
    def complete(self, messages: list[Message], schema: type[BaseModel]) -> BaseModel: ...
    def complete_with_tools(self, messages: list[Message], tools: list[ToolDef]) -> ToolTurn: ...
```

Implementations: `OpenAiCompatibleClient` and `StubClient` (recorded responses
from `tests/fixtures/llm/`). Selected by `LLM_PROVIDER`.

Environment: `LLM_PROVIDER` (`openai_compatible` or `stub`), `LLM_BASE_URL`
(`https://api.groq.com/openai/v1`), `LLM_MODEL` (`openai/gpt-oss-120b`),
`LLM_API_KEY`.

### 4.2 Structured output

- Send a JSON schema with `strict: true` where the provider supports it.
- Strict-mode schema rules: every field `required`, optional values as a union
  with null, `additionalProperties: false` on every object, enums for closed
  sets, generators as a tagged union on `type`.
- Groq does not combine structured output with tool use. The planner therefore
  runs tool-calling turns first, then one final structured call.
- Without strict mode: request JSON, parse with Pydantic, retry on failure.

### 4.3 Three gates on every response

1. Schema: provider-enforced or JSON mode.
2. Parsing: Pydantic model for the node.
3. Meaning: code checks references, sums and limits against metadata and plan.

On failure, re-prompt with the previous output and a numbered error list. After
three failures the job ends as `needs_review`.

### 4.4 Prompts

- One template per role in `agents/prompts/`, with a version in its header.
- Four parts: role and task, hard rules, delimited context blocks, output rules.
- Request text and column descriptions are untrusted. Put them in marked
  blocks with an instruction to treat them as data.
- Temperature 0 for all roles except the planner (0.3).
- One worked example per role, from a subject unrelated to the seed data.
- A profile card is about 300 to 600 tokens. Above 15 tables, run the mapper in
  batches and give the planner role-bearing columns only.

### 4.5 Resilience and audit

- Exponential backoff with jitter on rate limits and server errors, up to five
  attempts, honouring `Retry-After`. A client-side rate limiter is configurable.
- Log every call to `llm_calls`: job, node, model, prompt version, prompt,
  response, tokens, latency, status.

---

## 5. LLM output schemas

| Role | Schema | Fields |
| --- | --- | --- |
| intent_analyst | `TestIntent` | domain (free text), objective, test_type, entities_of_interest, must_include, must_exclude, volume_hint, include_invalid, ambiguities (question, options, blocking), confidence |
| semantic_mapper | `SemanticMap` | per table: grain, entity_key, event_time, label_column, columns (name, role, note); business_rules (rule, columns, confidence) |
| scenario_planner | `ScenarioPlan` | scenarios (name, category, description, rationale, share, unit, tables, primitives, levers, signature, template_ref); assumptions |
| overlay_writer | `list[ScenarioSpec]` | see section 8.4 |
| repair_analyst | `SpecPatch` | patches (JSON pointer path, new value, reason); plan_change_required; explanation |
| reporter | `Summary` | text of at most 150 words; caveats |

Enumerations:
- `test_type`: model_validation, rule_testing, pipeline_testing,
  performance_testing, ui_testing.
- Column `role`: identifier, entity_key, foreign_key, event_time, measure,
  category, status, location, free_text, label, sensitive_identifier, other.
- Table `grain`: entity, event, reference.
- Scenario `category`: typical, target_scenario, near_miss, boundary, rare,
  temporal, volume_skew, invalid.

A `signature` is a list of assertions the validator can evaluate. Each has a
column or metric, an operator, a value and an optional time window.

Repair patches apply only to allow-listed paths: generator parameters, clip
bounds, sequence counts and gaps, conditional cases. Anything else sets
`plan_change_required` and returns the job to approval.

---

## 6. Planner tools

Read-only. Arguments are validated against metadata; an unknown table or column
returns an error string to the model. Results are capped at about 2,000 tokens
and cached per job. Sensitive columns return masks and lengths only.

| Tool | Arguments | Returns |
| --- | --- | --- |
| `get_column_profile` | table, column | Full LLM-safe statistics for one column |
| `get_conditional_stats` | table, target, by | Target summary per category, top 20 groups |
| `get_relationship_profile` | child, parent | Children per parent, key skew, orphan rate |
| `get_behaviour_profile` | table, entity_key, time_column, measure | Velocity, inter-arrival, burst sizes |
| `get_class_separation` | table, label | Columns ranked by class separation |
| `search_scenario_templates` | domain, keywords | Matching template summaries |
| `get_scenario_template` | template_id | The full template |

---

## 7. Scenario generation

### 7.1 Sources of scenarios

1. User-stated: parsed from the request.
2. Profile-derived: produced by code from the profile (boundaries, rare values,
   peaks, skew, rule breaks). Needs no domain knowledge.
3. Knowledge-derived: the planner's own knowledge, plus optional templates.

### 7.2 Primitives

Every scenario is a composition of these fourteen. They are domain-neutral.

| Primitive | Changes | Spec construct |
| --- | --- | --- |
| value_shift | A column's distribution for chosen rows | generator override |
| mix_shift | Category proportions | `categorical` override |
| conditional_rule | Values that depend on other columns | `conditional`, `derived` |
| sequence | Ordered events per entity with set gaps | `SequenceSpec` |
| timing_shift | When events happen | `datetime` override |
| gap_or_absence | An expected event that stops or never comes | sequence with long gap; zero children |
| relationship_shape | Children per parent and key skew | cardinality, `foreign_key` override |
| cross_entity_link | Entities sharing a link they normally would not | `foreign_key` from a shared group |
| state_path | Order of statuses | sequence over a status column |
| correlation_change | How columns move together | correlation override |
| boundary_value | Values at type, profile or rule limits | `boundary` |
| volume_skew | Hot keys and row multipliers | `foreign_key` weights; row targets |
| text_content | What free-text fields say | `text_pool` (at most 500 model-written values) |
| rule_break | Deliberately invalid data | defects |

### 7.3 Planner procedure

1. Split the request into scenario intents.
2. Ground each in real columns through semantic roles.
3. Compose it from primitives.
4. Quantify it from the profile ("large" = above the 97th percentile). Never
   invent a number.
5. Declare a signature.
6. Add profile-derived scenarios for categories the request did not mention.

Scenario shares sum to at most 1. The remainder is typical data.

### 7.4 Feasibility check (code, after the planner)

| Result | Meaning | Shown at approval |
| --- | --- | --- |
| expressible | Columns exist and primitives apply | The scenario |
| partial | Part needs data the job lacks | Covered, not covered, why |
| not_expressible | No primitive or column can carry it | Reason and nearest alternative |

Nothing is dropped silently.

### 7.5 Templates (optional)

YAML files in `scenarios/templates/`: id, domain, name, version, category,
required_roles, primitives, levers on roles, signature, parameters, default
share. The system must work with zero templates. Ship one small pack for card
fraud as an example. A scenario with no template is marked model-proposed.

---

## 8. Deterministic components

### 8.1 Metadata

`config/metadata/<source>.yaml` holds table descriptions, column descriptions,
sensitivity tags and relationships. A `MetadataProvider` protocol reads it;
fall back to `PRAGMA foreign_key_list` when a relationship is missing.

Models: `TableMeta`, `ColumnMeta`, `Relationship`, `TableSet`.

Dependency graph: edge parent → child per relationship; add unselected parents
as support tables; topological sort; fill a self-reference in a second pass;
break a mutual reference at its nullable key.

### 8.2 Connector

```python
class Connector(Protocol):
    def get_columns(self, table: str) -> list[ColumnInfo]: ...
    def row_count(self, table: str) -> int: ...
    def aggregate(self, table: str, exprs: list[AggExpr],
                  group_by: list[str] | None = None,
                  where: Filter | None = None) -> list[dict[str, Any]]: ...
    def sample(self, table: str, columns: list[str], n: int, seed: int) -> pa.Table: ...
```

SQLite notes:
- Open with `sqlite3.connect("file:<path>?mode=ro", uri=True)`.
- Declared types come from `PRAGMA table_info`. SQLite typing is loose, so
  infer the real type from the sample. Dates are usually ISO-8601 text.
- SQLite has no standard-deviation or percentile function by default. Compute
  variance from `COUNT`, `SUM(x)` and `SUM(x*x)`. Compute percentiles from the
  sample with NumPy.
- Seeded sampling: a deterministic hash-modulo filter on `rowid`. When the
  table has no more rows than the sample size, read it all.
- Default sample size 200,000 rows per table.

### 8.3 Profiler

Nine groups. Each stored statistic records its source: `sql` (exact), `sample`
or `derived`. All SciPy work runs on the seeded sample.

**A. Table:** row count; primary-key uniqueness; duplicate-row rate; freshness
(min and max of date columns); growth (rows per day and month); completeness.

**B. Every column:** declared and inferred type; null count and rate; distinct
count and cardinality ratio; unique and constant flags; most frequent value and
share; empty and whitespace-only rate; semantic type; sensitivity class.

**C. Numeric:**
- min, max, range, sum, mean, standard deviation, variance (SQL)
- percentiles 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9
- `scipy.stats.iqr`, `median_abs_deviation`, `variation`, `skew`, `kurtosis`
- zero rate, negative rate; integer flag, decimal places, round-number rates
- histogram with 50 bins; empirical quantile function with 1,001 points
- best-fit distribution: `.fit` on norm, lognorm, gamma, expon, weibull_min,
  beta, pareto, uniform, t; ranked by AIC and `scipy.stats.kstest`
- normality: `shapiro` (up to 5,000 rows), `normaltest`, `anderson`, `jarque_bera`
- modes: `scipy.stats.gaussian_kde` with `scipy.signal.find_peaks`
- outliers: beyond 1.5 IQR fences; z-score above 3; modified z above 3.5
- sequence detection (constant step); first-digit test against Benford's law
  with `scipy.stats.chisquare`; Poisson and negative-binomial fit for counts

**D. Categorical and text** (booleans are two-value categoricals):
- frequencies: full table up to 1,000 distinct values, else top 100 + "other"
- `scipy.stats.entropy`, normalised entropy; rare-value count (below 0.1%)
- Zipf exponent of the rank-frequency curve
- length min, max, mean, percentiles; character-class shares
- pattern masks (letters → `A`, digits → `9`, punctuation kept) with shares
- fixed prefixes, suffixes and positions; format validity per semantic type
  (for example Luhn pass rate, email syntax)
- case style; leading and trailing spaces; word-count distribution

**E. Date and time:** min, max, span; granularity; counts by month, weekday,
day of month, hour; trend with `scipy.stats.linregress`; seasonality with
`scipy.signal.periodogram`; gaps; future-date and sentinel-date rates; weekend
and business-hours share.

**F. Cross-column:**
- numeric–numeric: `pearsonr`, `spearmanr`, `kendalltau` (Spearman matrix feeds
  the copula)
- categorical–categorical: Cramér's V via `scipy.stats.contingency.association`,
  `chi2_contingency`, Theil's U
- numeric–categorical: correlation ratio, `f_oneway`, `kruskal`
- boolean–numeric: `pointbiserialr`
- conditional distributions for strongly associated pairs
- functional dependencies; ordering rules (a ≤ b); arithmetic rules (c = a + b,
  c = a × b); null co-occurrence

**G. Cross-table, per relationship:** orphan rate; parent coverage; children
per parent (min, max, mean, percentiles, histogram) with Poisson,
negative-binomial, log-normal and Zipf fits; key skew (top 1% share, Gini);
temporal order between parent and child dates; child summaries by parent
category.

**H. Behaviour per entity over time** (for each foreign key × main timestamp on
event tables; other pairs on demand through a tool): events per entity per day
and week; inter-arrival percentiles and fit (expon, gamma, lognorm); burst size
in 1 hour and 24 hours; value relative to the entity's median; distinct
counterparties per entity; state-transition matrix; lag-1 autocorrelation;
active lifetime.

**I. Label-stratified** (when a label column exists): class balance; groups C,
D, E and H per class; class separation per column with `scipy.stats.ks_2samp`,
`wasserstein_distance` and `scipy.spatial.distance.jensenshannon`.

Sensitivity classes (metadata tags override rules):

| Class | Stored | Generated from |
| --- | --- | --- |
| public | Full statistics and frequencies | Frequencies |
| internal | Full statistics; values with fewer than 10 rows fold into "other" | Distributions |
| sensitive | Masks, lengths, validity rates. No values. | Masks and format rules |

Profile cache: table `profile_cache` keyed by source, table and schema hash.
Reuse until the schema hash changes, the profile is older than 7 days, or the
user passes `--refresh`.

`LlmSafeProfile` (the profile card): row count, relationships, and per column
the type, semantic type, role, null rate, distinct count, a five-number summary
or top five values, best-fit distribution. Sensitive columns show the mask only.

### 8.4 Generation spec

Objects: `GenerationSpec` (version, seed, profile versions, tables, scenarios,
output), `TableSpec` (name, row rule, columns, correlation matrix, constraints),
`ColumnSpec` (name, type, null rate, generator), `ScenarioSpec` (name, category,
share, unit, entity, labels, overrides, sequence, defects), `SequenceSpec`
(table, entity key, ordered steps with count, gap and column settings),
`Constraint` (order, range, unique, not-null, expression).

Generators: `sequence`, `uuid`, `foreign_key` (uniform, Zipf, weighted, shared
group), `distribution`, `empirical`, `categorical`, `conditional`, `datetime`,
`pattern` (with optional check digit), `semantic`, `text_pool`, `boundary`,
`derived`, `constant`.

`derived` expressions use a whitelisted grammar parsed by our own parser:
arithmetic, comparisons, if/else, min, max, round, date add and diff, string
concatenation, column references.

Layers: code compiles the baseline from the profile. The LLM writes scenario
overlays only. An overlay replaces the generators it lists; every other column
inherits the baseline.

Spec validation (all must pass): schema; tables and columns exist; generator
suits the column type; weights sum to 1 and shares to at most 1; foreign keys
point to earlier tables; expressions parse and reference known columns;
sensitive columns use `pattern` or `semantic` only; correlation matrices are
symmetric positive semi-definite (else nearest valid); row counts within
limits; sequence gaps positive.

### 8.5 Generation engine

Plan: order tables; size root tables from the request and child tables from
parent count × cardinality; split into chunks (default 250,000 rows; child
tables chunked by parent key range so one entity's events share a chunk);
derive one random stream per chunk with `numpy.random.SeedSequence.spawn`;
pre-allocate disjoint key ranges.

Inside a chunk, in order: scenario assignment by seeded hash of the entity key;
foreign keys drawn from the known parent key range; parent attributes read from
the parent's Parquet; correlated numerics by Gaussian copula; categoricals;
timestamps; patterns; personal fields from seeded pools; unique columns from a
permuted counter; derived columns and constraints; nulls; scenario overlays and
sequences; defects (opt-in, labelled in `_defect`).

Gaussian copula: convert the Spearman matrix with `2 * sin(pi * rho / 6)`; draw
correlated normals; map to uniforms with `scipy.stats.norm.cdf`; map to each
column through the fitted `ppf` or `numpy.interp` on the stored quantiles.

Every row carries `_scenario` and `_category`. A chunk reruns with the same
seed and gives identical output. Each chunk is one Parquet file (PyArrow, zstd).

### 8.6 Validation

Thresholds are effect sizes, not p-values. Defaults live in config.

| Family | Check | Default |
| --- | --- | --- |
| Fidelity | KS statistic vs stored quantiles (`ks_2samp`) | ≤ 0.10 |
| Fidelity | Wasserstein distance / IQR | ≤ 0.10 |
| Fidelity | Jensen-Shannon distance on category mix | ≤ 0.10 |
| Fidelity | Null-rate difference | ≤ 0.02 |
| Fidelity | Largest Spearman difference | ≤ 0.10 |
| Fidelity | JS distance on hour, weekday, month mix | ≤ 0.10 |
| Fidelity | KS on children per parent | ≤ 0.10 |
| Fidelity | Row count vs target | within 1% |
| Integrity | PK unique; FK resolves; not-null; types and formats; constraints; parent date before child date | 100% |
| Scenario | Share vs approved share | within 10% relative |
| Scenario | Each scenario appears, even at 100 rows | ≥ 1 |
| Scenario | Signature assertions pass on labelled rows | ≥ 99% |
| Scenario | KS between scenario and typical rows on signature columns | ≥ 0.30 |
| Scenario | Near-miss rows matching a target signature | 0 |
| Scenario | Labels present and counts match manifest | 100% |
| Privacy | 10,000 synthetic sensitive values found in source | 0 |
| Privacy | Synthetic row hashes matching profiling-sample hashes | 0 |
| Privacy | Suppressed rare values in output | 0 |

Fidelity applies to typical rows only. Scenario rows are tested against their
own signature. Integrity and privacy are hard gates: output is never released
with a failure. Each report row: check, table, column, metric, value,
threshold, status.

---

## 9. CLI

All commands go through `main.py`.

| Command | Purpose |
| --- | --- |
| `seed` | Build the fake source databases |
| `tables --source S` | List tables and relationships |
| `profile --source S --tables a,b [--refresh]` | Profile and print profile cards |
| `run --source S --tables a,b --request "..." [--rows t=N] [--seed N] [--include-invalid] [--auto-approve] [--workers N]` | Start a job |
| `resume JOB` | Continue a paused or interrupted job |
| `status JOB` / `report JOB` / `preview JOB --table t` | Inspect a job |
| `eval` | Run the golden set and print metrics |

The plan is rendered as a table: scenario, category, share, feasibility,
signature, template or model-proposed. The user can approve, edit shares,
remove scenarios, or reject with feedback.

## 10. Persistence, output and config

`state.db` tables: `jobs`, `job_artifacts` (versioned intent, plan, spec,
report as JSON), `approvals`, `llm_calls`, `profile_cache`.

```
output/jobs/<job_id>/
  data/<table>/part-00000.parquet ...
  manifest.json            files, row counts, schema, seed, spec hash, scenario counts
  spec.json
  validation_report.json
```

`config/settings.yaml` groups: llm, profiling, generation, agent (caps, prompt
versions), validation (thresholds), paths. Environment variables override.
`.env.example` documents every variable.

Error handling: retry a failed chunk three times with the same seed, then fail
the job and remove partial files; on restart skip chunks already in the
manifest; expire unapproved plans after 7 days.

## 11. Seed data and golden set

`python main.py seed` builds two fake sources with a fixed seed:

- `finance.db`: customers, accounts, cards, merchants, transactions. Include a
  label column `is_fraud` at roughly 0.5%.
- `orders.db`: customers, products, orders, order_items, shipments.

Seed data must have realistic shape so the profiler has something to find:
log-normal amounts, Zipf-skewed foreign keys, hour-of-day and weekday patterns,
correlated columns, nulls, a few rule-following date pairs. All values fake.
The largest table in each source has about 200,000 rows. The metadata YAML for
both sources is written by hand and committed in `config/metadata/`.

Golden set in `tests/golden/`: 15 to 20 requests across both sources and all
five test types, each with an expected intent and must-have scenarios.

Metrics from `eval`: intent accuracy; role accuracy against hand labels; plan
coverage; first-pass spec validity; repair loops per job; dry-run pass rate;
tokens and latency per job.

---

## 12. PLAN.md format

Create `PLAN.md` in the repo root with exactly these sections.

1. **Understanding.** Ten lines at most: what is being built and for whom.
2. **Assumptions and open questions.** Anything `SPEC.md` leaves unclear.
   Questions the user must answer before a milestone are marked BLOCKING.
3. **Decisions.** Empty at first. Append dated entries as questions are answered.
4. **Milestones.** One subsection per milestone in section 13. Each has:
   - Goal: one sentence.
   - Spec sections to read.
   - Tasks: a checkbox list. Each task names the files it creates or changes
     and the tests that prove it. A task is at most half a day of work.
   - Acceptance: the commands to run and the result expected.
5. **Risks.** Each with likelihood, impact and mitigation.
6. **Definition of done.** Copied from section 14.
7. **Progress log.** Empty at first. One dated line per finished milestone.

Rules: do not change milestone order without asking. Do not add scope. Keep
`PLAN.md` current; it is the single source of truth for progress.

## 13. Milestones

Agent-first: a thin end-to-end slice comes early, then each part deepens.

| # | Milestone | Builds | Done when |
| --- | --- | --- | --- |
| M0 | Scaffold | `git init` if needed, repo structure, `pyproject.toml`, Makefile, tooling, config loader, `main.py --help` | `make check` passes on an empty project |
| M1 | Foundations | Domain models, errors, state store, SQLite connector, metadata provider, dependency graph, seed command | `seed` builds both sources; `tables` lists them with relationships |
| M2 | Walking skeleton | All 15 nodes wired with the stub LLM; basic profiler (groups A, B, core of C, D, E); basic engine (independent columns, keys, labels); basic validator (integrity, row count); interrupts, checkpoints, CLI approval | `run` on two related tables finishes with the stub; `resume` works after a forced stop |
| M3 | Real agents | LLM client, six prompts, structured output and the three gates, planner tools, primitives, feasibility check, patch allow-list, LLM call log | A real request on each source yields an approved plan and valid overlays |
| M4 | Full profiler | Groups C to I complete, sensitivity classes, cache, profile cards | Profiles of the seed data recover the distributions the seed used |
| M5 | Full engine | Copula, conditionals, sequences, every primitive, chunking, optional workers, defects | A hand-written spec yields 1,000,000 rows; a rerun with the same seed is identical |
| M6 | Full validator and repair | All four families, report, repair loop tuned | Typical output passes fidelity; injected defects are caught; a seeded failure is repaired |
| M7 | Evaluation and demo | Golden set, `eval` command, README, demo script | Golden set runs on both sources at 100, 100,000 and 1,000,000 rows |

## 14. Definition of done

- `make check` passes: format, lint, strict typing, all tests.
- Coverage is at least 85% on `domain`, `spec`, `engine`, `validation`.
- The dependency rule in `CLAUDE.md` holds; only `agents` imports LangGraph.
- A job can be interrupted at any node and resumed to the same result.
- Same seed, spec and profile version give byte-identical Parquet.
- No source value from a sensitive column appears in any output or LLM prompt.
- The system produces a sensible plan for a request on `orders.db` with no
  template installed.
- `README.md` explains setup, the commands and the demo in under two pages.
