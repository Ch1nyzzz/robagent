from __future__ import annotations

import re
from typing import Any

from agent.events import EventLog, new_run_id, traces_dir
from agent.llm import chat, DEFAULT_MODEL


# Ask the model to externalise reasoning into visible tokens and mark the answer
# clearly.  Prior iterations used "no explanations, just the answer", which
# forced ALL reasoning into hidden internal CoT tokens — exhausting the 8192-token
# budget on hard tasks before any visible output was produced.  By allowing (and
# requiring) brief visible reasoning + a structured "FINAL ANSWER:" line, we give
# the model a way to produce a capturable answer even when the full reasoning
# chain is long.
SYSTEM_PROMPT = (
    "You are a precise problem-solving assistant. "
    "Work through the problem concisely — write only the key reasoning steps. "
    "Then output the final answer on its own line in this exact format:\n"
    "FINAL ANSWER: <value>\n"
    "where <value> is the answer only, with no extra text or punctuation."
)

# Recovery prompt when the primary call exhausts all tokens: revert to the
# iter3-proven "raw answer only" approach with a tight 1024-token budget.
RECOVERY_SYSTEM_PROMPT = (
    "You are an assistant. Output ONLY the final answer value to the question — "
    "no reasoning, no explanation, no thinking steps. "
    "Just the raw answer: a number, a short string, or a few words."
)

MAX_TOKENS = 8192
RECOVERY_MAX_TOKENS = 1024
RECOVERY_TEMPERATURE = 0.0


def _extract_answer(content: str, finish_reason: str) -> str:
    """Extract the answer from the model's response.

    Prefer the explicit 'FINAL ANSWER:' marker.  If absent:
    - finish_reason='stop'   → fall back to the last non-empty line.
    - finish_reason='length' → treat as empty (truncated before the marker).
    """
    if content:
        m = re.search(r"FINAL ANSWER:\s*(.+)", content, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip().rstrip(".")
            if candidate:
                return candidate
        if finish_reason == "stop":
            lines = [ln.strip() for ln in content.strip().splitlines() if ln.strip()]
            return lines[-1] if lines else ""
    return ""


def run_task(
    *,
    benchmark: str,
    task_id: str,
    task_prompt: str,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_id = new_run_id()
    log = EventLog(run_id=run_id, benchmark=benchmark, task_id=task_id, out_dir=traces_dir())
    extras = extras or {}

    root = log.emit(
        "run.started",
        question=task_prompt,
        model=DEFAULT_MODEL,
        extras=extras,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_prompt},
    ]

    call = log.emit("llm.requested", parent=root, messages=messages, model=DEFAULT_MODEL)

    try:
        result = chat(messages=messages, max_tokens=MAX_TOKENS)
    except Exception as e:
        log.emit("llm.failed", parent=call, error=repr(e))
        log.emit("run.failed", parent=root, error=repr(e))
        log.close()
        return {"run_id": run_id, "answer": None, "error": repr(e), "trace_path": str(log.path)}

    log.emit(
        "llm.responded",
        parent=call,
        content=result["content"],
        finish_reason=result["finish_reason"],
        usage=result["usage"],
    )

    answer = _extract_answer(result["content"] or "", result["finish_reason"])

    # Recovery: when the primary call exhausted all tokens on hidden CoT and
    # produced no usable visible output, fall back to a tight 1024-token call
    # with the direct answer-only prompt (reverting to iter3's proven recovery).
    if not answer:
        recovery_messages = [
            {"role": "system", "content": RECOVERY_SYSTEM_PROMPT},
            {"role": "user", "content": task_prompt},
        ]
        recovery_call = log.emit(
            "llm.requested",
            parent=root,
            messages=recovery_messages,
            model=DEFAULT_MODEL,
            recovery=True,
            recovery_max_tokens=RECOVERY_MAX_TOKENS,
        )
        try:
            recovery_result = chat(
                messages=recovery_messages,
                max_tokens=RECOVERY_MAX_TOKENS,
                temperature=RECOVERY_TEMPERATURE,
            )
        except Exception as e:
            log.emit("llm.failed", parent=recovery_call, error=repr(e), recovery=True)
        else:
            log.emit(
                "llm.responded",
                parent=recovery_call,
                content=recovery_result["content"],
                finish_reason=recovery_result["finish_reason"],
                usage=recovery_result["usage"],
                recovery=True,
            )
            recovered = (recovery_result["content"] or "").strip()
            if recovered:
                answer = recovered

    log.emit("answer.emitted", parent=root, answer=answer)

    return {
        "run_id": run_id,
        "answer": answer,
        "error": None,
        "trace_path": str(log.path),
        "log": log,
        "root_event_id": root,
    }
