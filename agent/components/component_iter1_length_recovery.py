"""Length-truncation recovery using the agent's own conversation history.

OBSERVED FAILURE MODE
---------------------
The prior implementation (iter1-3 frontier) re-asked the original
task_prompt cold, with no conversation history attached. For any task
that hit `finish_reason="length"` mid-conversation — i.e., after the
agent had already done meaningful tool work and reasoning — the
recovery LLM had to start from scratch. The iter3 train-30 trace
9b54f9d9 is the clearest evidence: the agent successfully read the
attached zip, ran a python_exec to enumerate 90 food items, and was
mid-way through a long synonym-matching reasoning trace when it hit
finish_reason="length" with `content=""`. The prior recovery
re-asked only the original question and produced
"CATEGORIES\\nCURRY" (ground truth: "Soups and Stews", driven by
"turtle soup" being the unmatched item).

MECHANISM
---------
Three stable contracts outside the evidence trace:
  - LLM API `finish_reason == "length"` (OpenAI-compatible completion
    field). The `on_length_truncation` event is fired by the FC loop
    exclusively in this case (agent/component_runtime/base.py L388-391).
  - EventLog `llm.requested` event schema: `fields.messages` is the
    full chat messages array passed to the model that turn
    (agent/component_runtime/base.py L350-351). The last
    llm.requested in the trace at the time of `on_length_truncation`
    contains the conversation history through (but not including) the
    truncated assistant message — i.e., the system prompt, the user
    question, all prior assistant turns (with reasoning_content +
    tool_calls), and all tool responses.
  - General algorithm: re-prompt the locked SUT with its own prior
    conversation context plus an explicit "your previous turn was cut
    off, commit a single FINAL ANSWER line now" instruction. Recovery
    pattern, not specific to any task.

CHANGES VS PRIOR IMPLEMENTATION
-------------------------------
- Subscribes directly to the Tier-1 `on_length_truncation` event
  rather than `post_llm_response` with a `finish_reason` matcher;
  the runtime synthesises this event only on length truncation so
  the matcher reduces to "raw_response is empty".
- Reads the most recent llm.requested.messages from `ctx.log.path`
  (same general algorithm as `exhaustion_answer_recovery`, which
  proved durable in iter1) and passes those messages directly to
  `ctx.chat()` after overriding the system message and appending a
  brief user nudge.
- Drops the dead `_FILE_CONTEXT_MARKER` branch — gaia_file_channel
  is no longer in the frontier; file content now comes from the
  baseline `file_read` tool.
- Returns `rewrite("FINAL ANSWER: <answer>")` so the downstream
  `_extract_final_answer` in the FC loop yields exactly `<answer>`
  (the prior implementation could return arbitrary content that was
  then taken as the full answer, leading to garbage output like
  "CATEGORIES\\nCURRY").
- All failure paths (missing log, unreadable trace, no prior request,
  chat exception, empty recovery, no extractable FINAL ANSWER) fall
  through to `Decision.allow()`, preserving the prior behaviour of
  leaving `ctx.raw_response` empty so `exhaustion_answer_recovery`
  at `pre_answer_emit` still gets its chance to synthesise.
"""
from __future__ import annotations

import json
from typing import Any

from agent.base import _extract_final_answer
from agent.component_runtime.types import (
    Component, ComponentClass, ComponentContext, Decision, Trust,
)


_RECOVERY_MAX_TOKENS = 4096

_RECOVERY_SYSTEM = (
    "Your previous response exceeded the per-turn output token budget "
    "before producing a final answer (typically because of an overlong "
    "reasoning trace). Do NOT continue reasoning. Use the conversation "
    "above (your earlier tool calls, their results, and your own prior "
    "reasoning) to commit the final answer NOW.\n\n"
    "Output exactly one line and nothing else:\n"
    "  FINAL ANSWER: <answer>\n\n"
    "Follow the GAIA answer format strictly:\n"
    "- numbers: digits only, no thousand separators, no units unless asked\n"
    "- strings: no extra prefix or explanation\n"
    "- lists: comma-separated values, no brackets\n"
    "If the evidence so far is insufficient, output your best estimate "
    "in the required format — an educated guess scores higher than an "
    "empty answer (which scores 0). Do not call any tool."
)

_RECOVERY_USER_NUDGE = (
    "Your previous turn ran out of output tokens before producing a "
    "final answer. Based on the conversation above, respond with a "
    "single line: FINAL ANSWER: <answer>."
)


def _read_latest_request_messages(path: str) -> list[dict[str, Any]] | None:
    """Return the `messages` payload of the most recent `llm.requested`
    event in the trace, or None if it cannot be reconstructed."""
    latest: list[dict[str, Any]] | None = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (ev.get("type") or "") != "llm.requested":
                    continue
                fields = ev.get("fields") or {}
                msgs = fields.get("messages")
                if isinstance(msgs, list) and msgs:
                    latest = msgs
    except OSError:
        return None
    return latest


def _build_recovery_messages(
    base_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Take the agent's prior request messages, swap the system prompt
    for the recovery instruction, and append a brief user nudge."""
    recovery: list[dict[str, Any]] = list(base_messages)
    if recovery and (recovery[0].get("role") or "") == "system":
        recovery[0] = {"role": "system", "content": _RECOVERY_SYSTEM}
    else:
        recovery.insert(0, {"role": "system", "content": _RECOVERY_SYSTEM})
    recovery.append({"role": "user", "content": _RECOVERY_USER_NUDGE})
    return recovery


def _matches(ctx: ComponentContext) -> bool:
    # The on_length_truncation event is synthesised by the FC loop only
    # when finish_reason == "length" (agent/component_runtime/base.py
    # L388-391). We additionally require raw_response to be empty so we
    # never stomp a partial response that already carries a FINAL ANSWER
    # line.
    return not (ctx.raw_response or "").strip()


def _handler(ctx: ComponentContext) -> Decision:
    log = getattr(ctx, "log", None)
    path = getattr(log, "path", None) if log is not None else None
    if not path:
        return Decision.allow()

    base_messages = _read_latest_request_messages(str(path))
    if not base_messages:
        return Decision.allow()

    recovery_messages = _build_recovery_messages(base_messages)
    try:
        result = ctx.chat(
            recovery_messages,
            max_tokens=_RECOVERY_MAX_TOKENS,
            temperature=0.0,
        )
    except Exception:
        return Decision.allow()

    content = (result.get("content") or "").strip()
    if not content:
        return Decision.allow()

    candidate = _extract_final_answer(content)
    if not candidate or not candidate.strip():
        return Decision.allow()

    return Decision.rewrite(f"FINAL ANSWER: {candidate.strip()}")


COMPONENT = Component(
    name="length_recovery_guard",
    cls=ComponentClass.REACTIVE_GUARD,
    listens="on_length_truncation",
    matcher=_matches,
    handler=_handler,
    priority=100,
    emits=(),
    trust=Trust(
        evidence_anchor=(
            "Three stable contracts outside the evidence trace: "
            "(1) LLM API field finish_reason='length' — the "
            "on_length_truncation event is fired by the FC loop "
            "exclusively when this field equals 'length' "
            "(agent/component_runtime/base.py L388-391); "
            "(2) EventLog 'llm.requested' event schema with "
            "fields.messages being the full chat messages array passed "
            "to the model that turn (agent/component_runtime/base.py "
            "L350-351; agent/events.py::EventLog); "
            "(3) the general algorithm 'when the model truncates mid-"
            "reasoning, re-prompt it with its own prior conversation "
            "context plus a directive to commit a concise final answer' "
            "— a recovery pattern independent of any task content. "
            "Sub-LLM helper ctx.chat() routes through the locked SUT "
            "model (agent/component_runtime/base.py::_make_chat_impl) so "
            "the recovery model identity matches the agent itself."
        ),
        blast_radius="local",
        rollback_when=(
            "train-30 accuracy decreases below the iter3 frontier score "
            "(21/30) — i.e., the conversation-history recovery "
            "produces a confidently wrong FINAL ANSWER on a task where "
            "the prior empty-prompt recovery would have failed-quietly "
            "(left raw_response empty so exhaustion_answer_recovery at "
            "pre_answer_emit could synthesise a correct answer), or the "
            "extra context causes the recovery model itself to hit "
            "length truncation more often, producing the same empty "
            "recovery output but burning more tokens."
        ),
        out_of_evidence_probe=(
            "Out-of-evidence Case A: a task that hits "
            "finish_reason='length' on iteration 1 with no prior tool "
            "calls — the latest llm.requested.messages is just "
            "[system, user]; the recovery messages become "
            "[recovery_system, user, user_nudge] which is functionally "
            "equivalent to the old cold-re-ask path. "
            "Out-of-evidence Case B: a task that finishes with "
            "finish_reason='stop' and non-empty content — the runtime "
            "does not emit on_length_truncation at all; the matcher "
            "never fires; no behavioural change. "
            "Out-of-evidence Case C: a task whose latest "
            "llm.requested.messages contains an assistant message with "
            "tool_calls but no following tool messages (malformed "
            "history) — by construction this is impossible: the FC "
            "loop logs llm.requested for turn N+1 only AFTER appending "
            "all tool responses for turn N (agent/component_runtime/"
            "base.py L350-451), so the recovery messages are always "
            "well-formed with respect to OpenAI tool_call rules. "
            "Out-of-evidence Case D: trace file missing or unreadable, "
            "or the recovery chat call raises, or recovery content is "
            "empty / has no extractable FINAL ANSWER — handler returns "
            "Decision.allow(); ctx.raw_response stays empty so "
            "exhaustion_answer_recovery at pre_answer_emit retains its "
            "safety-net role."
        ),
        fallback=(
            "Matcher returns False whenever raw_response is non-empty "
            "(partial-content length truncation is left alone). Handler "
            "returns Decision.allow() on any of: missing ctx.log.path, "
            "unreadable / empty trace file, no llm.requested events in "
            "the trace, ctx.chat() exception, empty recovery content, "
            "or no extractable FINAL ANSWER. A missed fire degrades "
            "exactly to the prior iter3 behaviour: ctx.raw_response "
            "stays empty, _extract_final_answer returns '', and "
            "exhaustion_answer_recovery at pre_answer_emit fires as the "
            "secondary recovery path."
        ),
    ),
)
