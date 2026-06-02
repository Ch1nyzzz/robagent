"""Mechanism layer: nudge the agent to commit at its actual final turn.

OBSERVED FAILURE MODE
---------------------
On 8 of the 11 train-30 failures in the iter1 frontier (e.g. 16d825ff,
9b54f9d9, 48eb8242, 72e110e7, 0b260a57, 5f982798, e142056d, 114d5fd0),
the FC loop hits `run.exhausted_iterations` at MAX_ITERATIONS=15 with
`finish_reason='tool_calls'` and `raw_answer=''`. The agent kept calling
tools through iter 15 and never emitted a `FINAL ANSWER:` line. The
existing `exhaustion_answer_recovery` (iter1 frontier) then has to guess
from a truncated trace JSONL summary — by construction it sees less
context than the agent itself had on its last turn (no `reasoning_content`,
tool results clipped to 1500 chars, capped at 12 pairs).

MECHANISM
---------
Anchored to two stable system fields and one LLM API field:
  - `ctx.current_iter`            : 1-based FC turn counter
                                    (agent/component_runtime/base.py L337)
  - `agent.base.MAX_ITERATIONS`   : module-level int = 15
                                    (single source of truth for the loop bound)
  - `ctx.shared['finish_reason']` : OpenAI-compatible API field set from
                                    the locked SUT chat result
                                    (agent/component_runtime/base.py L364)

Fires at most ONCE per task, at the SINGLE moment where the warning can
still reach the LLM:

    `post_llm_response` of iter (MAX_ITERATIONS - 1)
    AND `finish_reason == 'tool_calls'`

i.e. the agent just finished iter 14 by issuing more tool_calls — there
is exactly ONE more LLM turn (iter 15) before the FC loop hits the
`else: run.exhausted_iterations` branch. The hook injects a system note
that the runtime drains at the top of iter 15's pre-LLM phase, so the
locked SUT sees an explicit "this is your final turn — commit now"
signal on its last call.

WHY THIS IS NOT A REPEAT OF iter2's turn_budget_warner
------------------------------------------------------
iter2's component fired at iters 12, 13, 14 (three warnings, landing at
iters 13, 14, 15). Train-30 dropped 18→17: at least two tasks the agent
would have finished naturally got pushed into premature commitment.

This component fires at iter 14 ONLY, AND only if iter 14 itself still
issued tool_calls (so the agent is provably in the exhaustion failure
path, not in "I'm about to commit" territory). It fires at most once per
task. The text is informational ("you have ONE turn left, please ensure
the response ends with FINAL ANSWER:"), not commanding.

INTERACTION WITH exhaustion_answer_recovery
-------------------------------------------
- If this hook helps the LLM commit on iter 15 → ctx.raw_response is
  non-empty → `_extract_final_answer` returns a real answer → the
  exhaustion_recovery matcher (`not ctx.answer.strip()`) returns False
  → recovery doesn't fire. The agent's own committed answer wins (better
  than recovery's blind guess, since the agent has full reasoning_content
  context and recovery only has truncated trace events).
- If this hook is ignored and the LLM still empties out on iter 15 →
  exhaustion_answer_recovery fires as before, no behaviour change.

So the worst case here is "no change from iter1 frontier" — the warning
just becomes one extra system message that the LLM ignored.
"""
from __future__ import annotations

from agent.base import MAX_ITERATIONS
from agent.component_runtime.types import (
    Component, ComponentClass, ComponentContext,
    Decision, Trust,
)


_FINAL_TURN_NOTICE = (
    "[Runtime notice] You have just completed iteration "
    f"{MAX_ITERATIONS - 1} of {MAX_ITERATIONS}. Your NEXT response is the "
    "FINAL turn — after it, the function-calling loop terminates and no "
    "further tool calls are possible. Please make sure your next response "
    "ends with a single line in the required format:\n"
    "  FINAL ANSWER: <value>\n"
    "Use the evidence you have already gathered. If you are still "
    "uncertain, output your best estimate in the required format — an "
    "educated guess scores higher than an empty answer (which scores 0)."
)


def _matches(ctx: ComponentContext) -> bool:
    if ctx.current_iter != MAX_ITERATIONS - 1:
        return False
    if (ctx.shared.get("finish_reason") or "") != "tool_calls":
        return False
    # Idempotent: only fire once per task. Uses the per-task,
    # per-component scratchpad on ctx.state.
    return not ctx.state.get("fired", False)


def _handler(ctx: ComponentContext) -> Decision:
    ctx.state["fired"] = True
    return Decision.inject_context(_FINAL_TURN_NOTICE)


COMPONENT = Component(
    name="final_turn_commit_notice",
    cls=ComponentClass.MECHANISM_LAYER,
    listens="post_llm_response",
    matcher=_matches,
    handler=_handler,
    priority=120,
    emits=(),
    trust=Trust(
        evidence_anchor=(
            "Three stable runtime fields outside the evidence traces: "
            "(1) ComponentContext.current_iter — 1-based FC iteration "
            "counter set at agent/component_runtime/base.py per-turn and "
            "declared on the ComponentContext dataclass in types.py; "
            "(2) agent.base.MAX_ITERATIONS — fixed module-level int (=15), "
            "single source of truth for the FC-loop termination bound, "
            "re-imported by the component runtime so both code paths "
            "agree; (3) ctx.shared['finish_reason'] — OpenAI-compatible "
            "completion API field, populated from the locked SUT chat "
            "result. The mechanism (informing the agent its turn budget "
            "is exhausted on its final call) is a general control-loop "
            "stabilization pattern independent of any task content."
        ),
        blast_radius="local",
        rollback_when=(
            "train-30 accuracy decreases by more than the iter1 frontier "
            "score (18/30) — i.e., the final-turn nudge causes the LLM "
            "to commit prematurely with a wrong answer on tasks where "
            "exhaustion_answer_recovery would otherwise have synthesised "
            "a correct one, or where the LLM at iter 15 would have "
            "naturally produced FINAL ANSWER on its own."
        ),
        out_of_evidence_probe=(
            "Out-of-evidence Case A: a task that commits at iter 7 with "
            "finish_reason='stop' — matcher returns False at iter 7 "
            "(current_iter != MAX_ITERATIONS - 1), no fire, ctx unchanged. "
            "Out-of-evidence Case B: a task on iter 14 whose response "
            "got finish_reason='length' (output budget hit) — matcher "
            "returns False (finish_reason != 'tool_calls'); the existing "
            "length_recovery_guard handles this path. "
            "Out-of-evidence Case C: a task on iter 14 with "
            "finish_reason='stop' that already emitted FINAL ANSWER but "
            "had tool_calls earlier in the same response (mixed) — "
            "matcher returns False; the natural FC-loop exit at top of "
            "iter 15 (no tool_calls means break) handles termination."
        ),
        fallback=(
            "Matcher returns False whenever current_iter != "
            "MAX_ITERATIONS - 1, or finish_reason != 'tool_calls', or "
            "the per-task state flag 'fired' is already True. Handler "
            "returns Decision.inject_context only — never blocks, never "
            "rewrites. A missed fire (matcher false-negative) degrades "
            "to the iter1 frontier behaviour exactly; "
            "exhaustion_answer_recovery remains the safety net at "
            "pre_answer_emit. A false-positive fire (warning ignored by "
            "LLM) is also no-op since the warning is informational."
        ),
    ),
)
