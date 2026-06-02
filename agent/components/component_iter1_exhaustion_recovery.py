"""Reactive guard: recover an empty final answer with a sub-LLM synthesis pass.

Fires at `pre_answer_emit` when `ctx.answer` is empty. In the iter0 frontier
trace set, 13 of 14 failures landed at this exact state: the FC loop hit
`run.exhausted_iterations` (MAX_ITERATIONS=15) without the model ever emitting
a `FINAL ANSWER:` line, so `_extract_final_answer(ctx.raw_response)` returned
the empty string. An empty answer scores 0 on the GAIA scorer by definition;
a synthesised best-guess answer at least has a chance.

The handler reads the same JSONL trace the EventLog has been flushing (its
path is the stable `ctx.log.path` runtime contract) to reconstruct the
question + the most recent tool-call evidence, then issues ONE locked-model
chat via `ctx.chat()` asking only for `FINAL ANSWER: <x>` formatted per the
GAIA SYSTEM_PROMPT. The result is run through `_extract_final_answer` and
rewritten back into `ctx.answer`.

This is REACTIVE_GUARD on the observed empty-answer signal — not CHANNEL,
not capability expansion. The main agent is still the protagonist; we're
just letting it take one more shot when iterations ran out before it could
summarise. Identical pattern to `length_recovery_guard`, applied one event
later in the lifecycle.
"""
from __future__ import annotations

import json
from typing import Any

from agent.base import _extract_final_answer
from agent.component_runtime.types import (
    Component, ComponentClass, ComponentContext,
    Decision, Trust,
)


_MAX_TOOL_RESULT_CHARS = 1500
_MAX_TOOL_EVENTS = 12
_RECOVERY_MAX_TOKENS = 4096

_RECOVERY_SYSTEM = (
    "You are recovering a benchmark answer that the main agent failed to emit "
    "before its turn budget ran out. The gathered tool evidence is below. "
    "Read the evidence and produce exactly one line:\n"
    "  FINAL ANSWER: <answer>\n"
    "Follow the GAIA answer format strictly:\n"
    "- numbers: digits only, no thousand separators, no units unless asked\n"
    "- strings: no extra prefix or explanation\n"
    "- lists: comma-separated values, no brackets\n"
    "If the evidence is insufficient, give your best single-token guess in the "
    "expected format — never refuse, never explain, never output anything other "
    "than the FINAL ANSWER line."
)


def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    head = limit - 60
    return text[:head] + f"\n... [truncated {len(text) - head} chars] ..."


def _read_trace_events(path: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return events


def _build_evidence_block(events: list[dict[str, Any]]) -> str:
    question = ""
    tool_pairs: list[tuple[str, dict, str]] = []
    last_assistant_text = ""

    pending_call: dict[str, Any] | None = None
    for ev in events:
        et = ev.get("type") or ""
        fields = ev.get("fields") or {}
        if et == "run.started" and not question:
            question = str(fields.get("question") or "")
        elif et == "llm.responded":
            content = fields.get("content") or ""
            if content:
                last_assistant_text = content
        elif et == "tool.called":
            pending_call = {
                "name": fields.get("name") or "",
                "args": fields.get("args") or {},
            }
        elif et == "tool.responded":
            if pending_call is not None:
                tool_pairs.append((
                    pending_call.get("name", ""),
                    pending_call.get("args", {}),
                    str(fields.get("result") or ""),
                ))
                pending_call = None

    recent = tool_pairs[-_MAX_TOOL_EVENTS:]
    lines: list[str] = []
    for idx, (name, args, result) in enumerate(recent, 1):
        try:
            args_str = json.dumps(args, ensure_ascii=False)
        except (TypeError, ValueError):
            args_str = str(args)
        args_str = _truncate(args_str, 400)
        result_str = _truncate(result, _MAX_TOOL_RESULT_CHARS)
        lines.append(
            f"[Tool {idx}: {name}]\n"
            f"  args: {args_str}\n"
            f"  result: {result_str}"
        )
    evidence = "\n\n".join(lines) if lines else "(no tool calls were made)"
    tail = _truncate(last_assistant_text, 1500)
    if tail:
        evidence += f"\n\n[Last assistant draft text — no FINAL ANSWER line]\n{tail}"
    return question, evidence


def _matches(ctx: ComponentContext) -> bool:
    return not (ctx.answer or "").strip()


def _handler(ctx: ComponentContext) -> Decision:
    log = getattr(ctx, "log", None)
    path = getattr(log, "path", None) if log is not None else None
    if not path:
        return Decision.allow()

    events = _read_trace_events(str(path))
    question, evidence = _build_evidence_block(events)
    if not question:
        question = ctx.prompt or ""
    if not question:
        return Decision.allow()

    user_msg = (
        f"Task:\n{question}\n\n"
        f"Gathered evidence from the agent's prior tool calls (most recent last):\n"
        f"{evidence}\n\n"
        f"Output the single FINAL ANSWER line now."
    )
    messages = [
        {"role": "system", "content": _RECOVERY_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    try:
        result = ctx.chat(
            messages,
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
    return Decision.rewrite(candidate.strip())


COMPONENT = Component(
    name="exhaustion_answer_recovery",
    cls=ComponentClass.REACTIVE_GUARD,
    listens="pre_answer_emit",
    matcher=_matches,
    handler=_handler,
    priority=110,
    trust=Trust(
        evidence_anchor=(
            "Anchored on three stable runtime contracts: "
            "(1) agent.base._extract_final_answer returns '' iff the final "
            "LLM turn carried no content — direct consequence of the FC loop "
            "exhausting MAX_ITERATIONS without emitting `FINAL ANSWER:`; "
            "(2) ComponentContext.log.path is a flushed-per-emit JSONL trace "
            "file (agent/events.py::EventLog) with a published schema "
            "containing `run.started`, `tool.called`, `tool.responded`, "
            "`llm.responded` event types; "
            "(3) the GAIA scorer awards 0 to an empty string by construction, "
            "so a sub-LLM synthesis from gathered tool evidence is "
            "weakly-dominant. The mechanism (re-prompt the locked SUT with "
            "(question, recent tool results) and extract a FINAL ANSWER line) "
            "is a general algorithm independent of any task content."
        ),
        blast_radius="local",
        rollback_when=(
            "train-30 accuracy decreases vs the iter0 frontier — i.e. the "
            "synthesis ever overrides a correct ctx.answer (matcher should "
            "make this impossible since it requires empty answer) or the "
            "recovery model produces consistently mis-formatted output "
            "leading to scorer false-negatives on previously-passing tasks."
        ),
        out_of_evidence_probe=(
            "Out-of-evidence case A: a task where the FC loop emits "
            "`FINAL ANSWER: 42` cleanly — ctx.answer = '42', matcher returns "
            "False, handler never fires, answer unchanged. "
            "Out-of-evidence case B: a task where ctx.answer is empty AND the "
            "agent made zero tool calls (e.g. early refusal) — handler still "
            "runs but `evidence` is the sentinel string '(no tool calls were "
            "made)' so the locked-model is forced to attempt an answer from "
            "the question alone. Worst case the answer is wrong (still 0); "
            "best case it recovers an easy lookup question."
        ),
        fallback=(
            "Matcher returns False when ctx.answer is non-empty. Handler "
            "returns Decision.allow() when: log.path is missing, the trace "
            "file can't be read, the recovery chat raises, or the recovery "
            "content is empty / has no extractable FINAL ANSWER."
        ),
    ),
)
