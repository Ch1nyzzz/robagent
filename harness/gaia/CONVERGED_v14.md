# CONVERGED v14 — GAIA harness phase D (re-entry on v15 traces)

## Why we stopped

§V "Full-validation gap": **3 new mechanisms** acted on since the last full
benchmark run (v15 = 48/165). Skill emits `.pending_validation` for v14 and
hands off to the external orchestrator for the next full-165 run.

## Scoreboard

| iter | agent | scope | score | acc | notes |
|---|---|---|---|---|---|
| 0 | v0 | full 165 | 32/165 | 0.194 | single-LLM baseline |
| 10 | v10 | full 165 | 44/165 | 0.267 | + tools, claim-graph |
| 15 | v13 | **full 165** | **48/165** | **0.291** | + rate-limiter + multi-source + extra readers |
| 16 | v14 | sample-10 | 3/10 | 0.300 | -1 vs v15 sample (stochastic) — full validation pending |

The v14 sample regression of 1 task (`2dfc4c37`: v15="6", v14="4") is from
the parametric-fallback LLM giving a different answer to the same prompt
with the same model. Same NEEDS_RETRIEVAL route in both runs, no v14 code
on that path executed (planner truncated, evidence UNKNOWN, parametric
fallback in both). Below the §III EXPERIMENTAL threshold (need ≥-2).

## Deltas applied in v14

| category | change | trace evidence (v15) |
|---|---|---|
| bug fix | `vision_describe` → routes through `chat()` (rate-slot wiring) | 10 image tasks blocked with `_acquire_rate_slot() missing 1 required positional argument: 'token_estimate'` |
| classification | `vision_describe` 400 "non-serverless" → `model_capability_gap:vision_unavailable` | §IV probe confirmed Together AI gates the free vision endpoint behind paid dedicated |
| token budget | `plan_retrieval` MAX_TOKENS 256 → 1024 | 57/73 NEEDS_RETRIEVAL had `finish_reason=length` on plan_retrieval; 34 ended BLOCKED |
| retry-on-empty | `answer_direct` retries once at 16384 if primary returned length+empty | 11 DIRECT tasks ended BLOCKED with `empty_after_reasoning` from length-truncation |

**3 deterministic patches. 0 new LLM contracts. 0 new tools. 0 new audit
gates** (existing (claim, source) protocol covers the unchanged event
shapes).

## Pytest

97/97 PASS. v14 adds 10 unit tests; AST-whitelist extended to scan all v14
files; 14 new structural keywords whitelisted (no task-entity literals
leaked).

## Open mechanisms (post-v14, for the next strategic decision)

1. **`parametric_unknown` (34 tasks blocked in v15)** — multi-source
   retrieval returns UNKNOWN-claim on hard tasks, then the parametric LLM
   fallback also UNKNOWNs. One-line diagnosis: retrieval coverage is the
   bottleneck on questions whose answer is not in the top-2 wiki hits
   AND not in DDG's first 3 snippets AND not in arXiv's top-2 abstracts.

2. **`wrong_answer/DIRECT` (21 in v15, down from v10's 35)** — DIRECT-route
   tasks the model answers but gets wrong. The router can't help here
   (these are not retrieval-needing questions); the model itself is the
   bottleneck. One-line diagnosis: LLM correctness gap.

3. **Vision (~11 tasks)** — confirmed `model_capability_gap` via §IV
   probe on v14. `do_not_revisit` already records this. Resolving it
   requires a paid vision endpoint or a provider switch, which is a
   strategic decision (out of scope for this skill).

## Suggested next strategic direction (≤3 bullets)

- **Try a fourth retrieval source on `retrieval.empty`-after-all-variants
  tasks**: Google Scholar (no-key) or Crossref REST. This directly attacks
  the `parametric_unknown` bucket (34 tasks). Risk: adds new tool surface;
  requires §IV check that the LLM can extract claims from search-snippet
  density (already proven for wiki/DDG/arXiv, so likely yes).
- **Tighten the claim-graph rejection threshold**: 12 of v15's 30
  `claim_unverified` blocks have at least one source opened but the
  claim is "UNKNOWN" — i.e., the LLM read the sources and judged them
  insufficient. Some of those may be over-conservative; loosening the
  reject criterion (e.g., accept when source content contains the
  question's key proper noun AND confidence>=0.3) could recover a few.
- **Accept the model-correctness ceiling on DIRECT failures**: 21
  wrong-DIRECT cases are not deterministic-fixable. The honest signal
  is the calibration improvement from BLOCKED→answered, not the score
  itself. Consider stopping deterministic iteration on DIRECT and
  accepting the v0→v15 +16-task gain as the ceiling for this model.

## Handoff signal

`harness/gaia/.pending_validation` contains `v14`. The external runner
should execute the full 165-task validation on v14 and write a fresh
`.last_validation.json`. The skill's next invocation will diff that
result against this state to decide phase E (or `.globally_converged`).
