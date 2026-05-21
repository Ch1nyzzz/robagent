# REPORT — v1

(Deliverable required by harness-grow SKILL Step 9; mission spec lists it under "What done looks like".)

## Failure landscape (v0 baseline -> input to v1)

| mechanism | n on v0 | addressed in v1 |
|---|---|---|
| M1_truncation (empty content + finish_reason=length) | 76 | yes — max_tokens 2048 -> 8192 |
| M2_blind_file (no file capability, fabricated guess) | 36 wrong + 2 lucky | yes — honest BLOCKED with agent.blocked.reason=needs_file_capability |
| M3_unsupported_source (no retrieval, fabricated) | ~30 wrong | yes — honest BLOCKED with agent.blocked.reason=needs_retrieval_capability |
| M4_format_drift (wrap quotes, trailing period, prefix) | ~5 | yes — normalize_answer strips before submit |
| M5_genuine_reasoning_miss (no quick win) | ~20 | unchanged (LLM-only) |

## Workflow (v1 state machine)

```
NEW
 -> parse_task (det, emit task.parsed) -> PARSED
     -> route_by_extras (det, emit task.routed{route})
         |- NEEDS_FILE -> emit agent.blocked{reason=needs_file_capability} -> BLOCKED (answer="")
         |- NEEDS_RETRIEVAL -> emit agent.blocked{reason=needs_retrieval_capability} -> BLOCKED (answer="")
         |_ DIRECT
             -> answer_direct (LLM, max_tokens=8192) -> emit llm.responded -> ANSWERED
                 -> extract_final_answer (det) -> normalize_answer (det)
                     -> emit answer.normalized -> SUBMIT
SUBMIT -> emit answer.emitted{answer, stage, block_reason, route}
```

## Deterministic helpers accepted

| helper | replay agreement on v0 |
|---|---|
| route_by_extras | n/a (no LLM equivalent; unit-tested structurally) |
| extract_final_answer | 76/76 = 100% on v0 non-empty contents |
| normalize_answer | idempotent on terse v0 answers; strips wrappers when present |

## LLM nodes

answer_direct — input: question; budget: max_tokens=8192, temperature=0.0.

## Audit gates (cumulative for iter 1)

1. gate_truncation — 76 fail / 0 correct on v0 (100% precision)
2. gate_blind_file_answer — 19 fail / 0 correct on v0 with structured-answer filter
3. gate_unsupported_source_claim — depends on task.routed (v1+ traces)

## Score (filled at run completion)

See ITERATION_LOG.md for the final scoreboard.
