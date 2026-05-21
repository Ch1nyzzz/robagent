"""v14 — deterministic refinements on top of v13 stack.

Three targeted fixes derived from the v15 full-165 trace landscape:

1. Vision tool signature fix: `_acquire_rate_slot` requires `token_estimate`
   and `deadline` keyword arguments (introduced in iter 9). v8's vision.py
   still called the v8 zero-arg signature, surfacing as 11 NEEDS_FILE tasks
   blocked with `vision_error:_acquire_rate_slot() missing 1 required
   positional argument: 'token_estimate'`. Fix routes the call through the
   public `chat()` API for the vision model so the rate-slot wiring lives
   in one place.

2. plan_retrieval token budget: 256 was too tight for a reasoning model
   (DeepSeek-V4-Pro burns thinking tokens before content). 57 / 73
   NEEDS_RETRIEVAL tasks hit `finish_reason=length` on plan_retrieval in
   v15, and 34 of those ended BLOCKED. Raised to 1024 (v6 default) which
   gives the model headroom while still trimming relative to answer_direct.

3. answer_direct retry-on-empty: 11 DIRECT tasks ended with
   `finish_reason=length` AND empty content (reasoning budget exhausted
   before any visible answer). When this fires, retry once with
   max_tokens=16384 to clear the cliff.

No new LLM contracts, no new tools. Three deterministic code edits.
"""
