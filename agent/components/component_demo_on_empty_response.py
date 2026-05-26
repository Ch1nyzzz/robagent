"""Phase F demo: on_empty_response → ctx.chat recovery (gaia).

PURPOSE
-------
Proves the Phase D Tier-1 event vocabulary + Phase C ctx.chat() capability
work end-to-end for the GAIA single-shot lifecycle. This component is NOT
part of the active frontier (gaia_main.yaml does not include it) — it is a
deliberate demonstrator that the new expressiveness shippable to proposers
in Phase E is wired through to dispatch.

MECHANISM
---------
The GAIA runtime synthesises `on_empty_response` whenever the SUT returns
an empty content string at POST_LLM_RESPONSE. Before Phase D, a guard had
to subscribe to `post_llm_response` and match on `not ctx.raw_response.strip()`
inside its own matcher; now the runtime already gates the event.

The handler re-issues the SAME messages through `ctx.chat()` (locked SUT
model, larger max_tokens budget) and `Decision.rewrite`s the recovered
content into `ctx.raw_response`. Subsequent dispatch sites (PRE_ANSWER_EMIT)
see the recovered text.

NOT-A-CHANNEL
-------------
This is REACTIVE_GUARD, not CHANNEL: it fires only on the observed failure
signal (empty content), never proactively. Matcher returns True at fire
time because the runtime already gated the event by content emptiness.
"""
from __future__ import annotations

from agent.component_runtime.types import (
    Component, ComponentClass, ComponentContext,
    Decision, Trust,
)


_RECOVERY_SYSTEM_HINT = (
    "Your previous answer was empty. Re-read the task and produce the final "
    "answer. If the expected answer is a number, output the number only. If "
    "it is a short string, output that string only. No commentary."
)


def _matches(ctx: ComponentContext) -> bool:
    # The runtime synthesises `on_empty_response` only when raw_response is
    # empty already, so this matcher is a tautology — kept explicit for the
    # SKILL convention "every component has a matcher".
    return not (ctx.raw_response or "").strip()


def _handler(ctx: ComponentContext) -> Decision:
    messages = [
        {"role": "system", "content": _RECOVERY_SYSTEM_HINT},
        {"role": "user", "content": ctx.prompt or ""},
    ]
    try:
        result = ctx.chat(
            messages,
            max_tokens=4096,
            temperature=0.0,
        )
    except RuntimeError:
        # ctx.chat unwired in some smoke contexts; fall through to ALLOW so
        # the demo doesn't crash the test harness.
        return Decision.allow()
    recovered = (result.get("content") or "").strip()
    if not recovered:
        return Decision.allow()
    return Decision.rewrite(recovered)


COMPONENT = Component(
    name="component_demo_on_empty_response",
    cls=ComponentClass.REACTIVE_GUARD,
    listens="on_empty_response",          # Tier-1 event subscription
    matcher=_matches,
    handler=_handler,
    priority=200,                          # fire AFTER length_recovery_guard
    emits=("on_empty_response_recovered",),  # downstream observers may listen
    trust=Trust(
        evidence_anchor=(
            "Anchored on the OpenAI Chat Completions API field "
            "`choices[0].message.content` being the empty string, which "
            "the runtime gates the `on_empty_response` event on. The "
            "recovery mechanism (re-issue with explicit reformulation "
            "system prompt) is a general algorithm independent of any "
            "task content."
        ),
        blast_radius="local",
        rollback_when=(
            "Disable when the empty-response failure mode disappears from "
            "the train traces, or when the recovery LLM call's added "
            "latency (~2-5s per fire) outweighs the recovery accuracy "
            "gain in the iteration's evolution_summary delta."
        ),
        out_of_evidence_probe=(
            "Out of evidence (no trace fed): if the model returns an empty "
            "content because of a provider transient hiccup, the handler "
            "fires, re-asks once, and on a successful recovery returns "
            "Decision.rewrite(recovered_text). If the second call also "
            "returns empty, the handler falls through to Decision.allow() "
            "and the pipeline continues with the original empty response."
        ),
        fallback=(
            "Matcher=False (raw_response is non-empty) → handler not invoked. "
            "Handler returns allow() on a still-empty recovery → no rewrite."
        ),
    ),
)
