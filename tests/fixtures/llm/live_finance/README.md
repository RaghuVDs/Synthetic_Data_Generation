# Recorded live responses

Answers from `openai/gpt-oss-120b` on Groq for the card-fraud request on `finance`
(customers, accounts, cards, transactions), recorded 2026-11-11 from the `llm_calls` log of a
completed job. Every file holds the response that passed all three gates. `SpecPatch.json` is a
placeholder: the live run needed no repair. No keys or source values are included.

Replay with `LLM_PROVIDER=stub`, `LLM_STUB_FIXTURES=tests/fixtures/llm/live_finance` and
`SYNTHGEN_AGENT__CAPS__PLANNER_TOOL_CALLS=1` (the live run's tool-call cap).
