# Pattern: mount = `session_end`

## When to choose this mount

Bookkeeping only in v1. Use for logging / telemetry that should fire after the answer is emitted but before the task returns. **No decision changes the outcome in v1** — `session_end` decisions are recorded in `fired.jsonl` but not applied.

If you find yourself reaching for `session_end` to *change* the answer, move to `pre_answer_emit` instead.

## Which classes admit this mount

None of the standard decisions are dispatched at SESSION_END in v1. This mount is reserved for future use (e.g., a cross-session aggregation component that updates a SESSION_END-scoped state file with the task's outcome for later durability audits).

## v2 plans

When CROSS_SESSION state_scope is implemented end-to-end:
- A SESSION_END mechanism_layer with `state_scope=CROSS_SESSION` could write per-task aggregated stats.
- That state would be readable by SESSION_START components on the next task (e.g., adaptive priors).

Not in scope for v1.
