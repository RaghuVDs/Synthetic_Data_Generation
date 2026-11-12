# Recorded live responses

Answers from `openai/gpt-oss-120b` on Groq for the card-fraud request on `finance`
(customers, accounts, cards, transactions), recorded 2026-11-11 from the `llm_calls` log of a
completed job. Every file holds the response that passed all three gates. No keys or source
values are included.

Replay with `LLM_PROVIDER=stub`, `LLM_STUB_FIXTURES=tests/fixtures/llm/live_finance` and
`SYNTHGEN_AGENT__CAPS__PLANNER_TOOL_CALLS=1` (the live run's tool-call cap).

Hand-added (not recorded): the last two overlays in `OverlaySet.json`,
`boundary_transactions_amount` and `volume_skew_transactions_merchant`. The live run's
feasibility check found those scenarios not expressible, so the model was never asked for them.
M5 added the `boundary` generator and weighted foreign keys, which makes them expressible; the
overlays were written by hand on 2026-11-11 from the recorded plan's levers so the replay still
passes every gate. Everything else is unchanged from the recording.

Fields added to `ScenarioSpec` after the recording (M5: `correlations`, `cardinality`,
`sequences`, `defects`) are filled with their empty value in the recorded overlays; no recorded value was
changed.

Hand-written (not recorded): `SpecPatch.json`. The live run needed no repair, but M6's integrity
checks find that the recorded overlays put transactions before their card's issue date. The
patch moves the three time windows after the last issue date in the seed (2025-06-30), the kind
of parameter-only repair the repair analyst may make, so the replay now includes a repair round.
