from __future__ import annotations

from typing import Any

from agent.events import EventLog, new_run_id, traces_dir
from agent.llm import chat, DEFAULT_MODEL


SYSTEM_PROMPT = (
    "You are an assistant solving a single benchmark task. "
    "Read the task carefully and produce the final answer only. "
    "Do not include explanations, prefixes, or extra text. "
    "If the expected answer is a number, output the number only. "
    "If the expected answer is a short string, output that string only."
)

# DeepSeek-V4-Pro generates hidden CoT tokens proportional to max_tokens.
# At max_tokens=16384 (mh_iter1), reasoning for hard tasks exceeds the 150s
# per-call API timeout, causing APITimeoutError. At max_tokens=2048 (v0),
# the budget is too small and the model outputs empty content (finish_reason=
# "length"). max_tokens=8192 targets the midpoint: enough thinking to
# complete reasoning (~82s at ~100 tok/s) without exceeding the 150s timeout.
MAX_TOKENS = 8192


def run_task(
    *,
    benchmark: str,
    task_id: str,
    task_prompt: str,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_id = new_run_id()
    log = EventLog(run_id=run_id, benchmark=benchmark, task_id=task_id, out_dir=traces_dir())

    root = log.emit(
        "run.started",
        question=task_prompt,
        model=DEFAULT_MODEL,
        extras=extras or {},
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

    answer = (result["content"] or "").strip()
    log.emit("answer.emitted", parent=root, answer=answer)

    return {
        "run_id": run_id,
        "answer": answer,
        "error": None,
        "trace_path": str(log.path),
        "log": log,
        "root_event_id": root,
    }
