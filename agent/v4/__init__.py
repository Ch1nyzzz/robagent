"""GAIA agent v4 — v3 plus retry-on-empty.

If `extract_final_answer` returns empty after the first LLM call (and the
finish_reason was 'length' so we already burned the budget on reasoning),
do not re-call — the budget is already exhausted. Instead, emit a structured
`agent.blocked` event with reason `empty_after_reasoning` so the harness can
distinguish budget-burned-empty from honest blocks.

This is a calibration improvement, not a capability addition.
"""
