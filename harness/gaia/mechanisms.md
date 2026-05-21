# Failure mechanisms — GAIA v0 (iter 1 input)

| mechanism_id | description | n (v0) | example traces |
|---|---|---|---|
| M1_truncation | LLM emits empty content with `finish_reason=length` — all 2048 tokens consumed by hidden reasoning | 76 | `gaia__32102e3e-…` `gaia__17b5a6a3-…` `gaia__7dd30055-…` |
| M2_blind_file | Question references attached file; v0 has no file reader; emits a fabricated guess | 36 wrong (+2 lucky correct) | `gaia__5cfb274c-…` `gaia__7cc4acfa-…` `gaia__04a04a9b-…` |
| M3_unsupported_source | Question cites authoritative source (URL / site name / "according to"); v0 has no retrieval; emits a fabricated answer | 30+ wrong | `gaia__7a4a336d-…` `gaia__0512426f-…` `gaia__0bdb7c40-…` |
| M4_format_drift | LLM output is correct semantically but has wrapping quotes / trailing period / leading "Answer:" so the scorer's exact-match branch rejects | ~5 | (caught and fixed by `normalize_answer`) |
| M5_genuine_reasoning_miss | LLM produced a confident wrong answer with no retrieval/file dependency | ~20 | (residual; LLM-only node has no quick win) |

## v1 mechanisms addressed

- M1 → solved by raising `max_tokens` from 2048 to 8192 in `answer_direct`.
- M2 → solved by routing to `NEEDS_FILE`-BLOCKED with structured event; agent emits empty answer + `agent.blocked` event instead of fabricating.
- M3 → solved by routing to `NEEDS_RETRIEVAL`-BLOCKED with structured event.
- M4 → solved by `normalize_answer` post-LLM.
- M5 → unaddressed in v1 (LLM-only territory).

## Tradeoff

v1 deliberately gives up the ~2 lucky correct file guesses (M2 false-positive corrects in v0)
because honest blocking provides a calibration signal that the harness can act on in v2+.
v0 → v1 score can drop on file/retrieval tasks but should rise sharply on direct tasks
that were previously truncated.
