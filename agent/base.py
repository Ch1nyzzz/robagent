from __future__ import annotations

from typing import Any

from .events import EventLog, new_run_id, traces_dir
from .llm import chat, DEFAULT_MODEL


SYSTEM_PROMPT = (
    "You are an assistant solving a single benchmark task. "
    "Read the task carefully and produce the final answer only. "
    "Do not include explanations, prefixes, or extra text. "
    "If the expected answer is a number, output the number only. "
    "If the expected answer is a short string, output that string only."
)


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
        result = chat(messages=messages)
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
