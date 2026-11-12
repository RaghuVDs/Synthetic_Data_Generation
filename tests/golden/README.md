# Golden set (SPEC 11, PLAN T7.1)

`requests/*.yaml`: one request each, run by `python main.py eval`.

- `source`, `tables`, `request`: what a tester would pass to `run`.
- `target_table`: the table each eval volume (100, 100,000, 1,000,000 rows) is set on.
- `answer`: reply to a clarification question, or null for the eval default.
- `expected.test_type` and `expected.include_invalid`: the intent the request states.
- `expected.must_have`: scenarios the plan must contain. One is covered when a planned
  scenario has one of `categories`, spans every table in `tables`, and (when `columns` is not
  empty) has a lever or signature assertion on one of `columns` in those tables. `what` is a
  note for people.

`roles/<source>.yaml`: hand labels, table -> column -> acceptable roles. Role accuracy is the
share of labelled columns of a job's tables whose mapped role is acceptable.
