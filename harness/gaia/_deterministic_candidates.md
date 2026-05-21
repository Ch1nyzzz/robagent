# Deterministic candidates — GAIA iter 1 (from v0 traces)

Observation set: 165 v0 traces, 32 correct / 133 wrong.

| Node | Pattern | n_obs | Proposed helper | Confidence | Accepted? |
|---|---|---|---|---|---|
| `eval.normalize` (implicit) | scorer strips whitespace, lowercases, removes punct (string branch) | 165 | `normalize_answer(s)` mirroring scorer rules | high | ✓ accepted |
| `answer.emitted` post `llm.responded` | Most non-empty LLM outputs are already terse (single word/number) — prompt enforces that | 82 | `extract_final_answer(content)` regex on boxed / `Answer:` patterns plus last-non-empty-line fallback | high | ✓ accepted (low intervention rate — most pass through unchanged, but it catches reasoning-leak cases) |
| router | `file_name` extra ↔ "needs file capability"; URL / TLD / citation phrase ↔ "needs retrieval"; else `DIRECT` | 165 | `route_by_extras(question, extras)` | high | ✓ accepted |
| `decide_plan` | n/a — v0 is single-shot, no plan event to extract; the plan is implicit in the LLM's chain of thought | n/a | LLM node retained (`answer_direct`) | — | LLM only |

Rejected / deferred candidates:
- "Parse the answer type (number vs string vs list) from question" — high false-positive risk (questions don't always announce type). Deferred to v2 once LLM begins emitting `answer.expected_format` events the harness can leverage.
- "Auto-extract numbers from prose" — would clash with legitimately-named entities. Deferred.

## Validation (replay)
Helpers run against the recorded v0 LLM content:
- `extract_final_answer`: on 82 non-empty contents from v0, 80/82 produced an output equal to the v0 `answer.emitted.answer` value (98% agreement). 2 disagreements: cases where v0 emitted the entire content and `extract_final_answer` picked the last line — still semantically equivalent.
- `normalize_answer`: idempotent on already-terse v0 answers (no change in 79/82 cases). Strips wrapper period/quote when present.
- `route_by_extras`: deterministic, no LLM to compare against; structural correctness verified in unit tests.

## Iter 6-9 additions

| Node | Pattern | Proposed helper | Confidence | Accepted? |
|---|---|---|---|---|
| `verify_claim` (post answer_with_evidence) | claim must trace to one of the emitted source_ids; UNKNOWN/missing/orphan → reject | `verify_claim_graph(claim_obj, opened_ids)` | high | ✓ v6 |
| `query_extract` (when LLM JSON parse fails) | longest capitalized multi-word span in question = best wiki query | `deterministic_query_from_question(q)` | high | ✓ v9 (4/4 unit cases pass) |
| `arithmetic` (questions ending with computable expression) | parse → restricted AST eval | `safe_arith(expr)` + `verify_arithmetic` tool | medium | ✓ v8 (no live usage but available for future routing) |
| `file_route` (NEEDS_FILE dispatch by extension) | image extensions → vision; text-extractable → file reader; else BLOCKED | inline `_IMAGE_EXTS` set + extension dispatch | high | ✓ v8 |

Rejected:
- "Auto-detect arithmetic in question" — would need a question classifier; deferred since arithmetic-only questions are rare in GAIA.
