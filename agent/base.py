"""GAIA v0 baseline — a tool-using FC loop over the locked SUT model.

Architecture: the LLM drives, the loop dispatches its tool_calls, and the
result strings are fed back as `tool` messages until the model emits a
turn with no tool_calls (final answer). Tools live in `agent.tools` and
are auto-registered via `TOOL_SPECS`.

This is the orientation: give the main agent a real toolbox upfront.
Stabilization belongs to the component runtime; this file is the
capability layer.
"""
from __future__ import annotations

import re
from typing import Any

from .events import EventLog, new_run_id, traces_dir
from .llm import chat, DEFAULT_MODEL
from .tools import TOOL_SPECS, dispatch_tool


SYSTEM_PROMPT = (
    "You are an assistant solving a single benchmark task.\n\n"
    "Tools available: file_read (read an attached file by file_name), "
    "url_fetch (HTTP GET a URL and return its text), "
    "web_search (query a search engine), "
    "python_exec (run Python code, capturing stdout/stderr). "
    "Use them as needed to gather information and compute.\n\n"
    "When you have the final answer, end your response with one line:\n"
    "  FINAL ANSWER: <answer>\n\n"
    "The <answer> must be exactly the value the question asks for:\n"
    "- numbers: digits only, no thousand separators, no units unless asked\n"
    "- strings: no extra prefix or explanation\n"
    "- lists: comma-separated values, no brackets\n"
    "Do not write anything after the FINAL ANSWER line."
)

MAX_ITERATIONS = 15
# deepseek-v4-pro thinking mode emits a lot of reasoning_content; 8192 leaves
# headroom for thinking + content + tool_calls per turn. base.py can override
# at the chat() call site.
PER_TURN_MAX_TOKENS = 8192

_FINAL_RE = re.compile(r"FINAL ANSWER:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def _extract_final_answer(text: str) -> str:
    if not text:
        return ""
    matches = list(_FINAL_RE.finditer(text))
    if matches:
        return matches[-1].group(1).strip()
    return text.strip()


def run_task(
    *,
    benchmark: str,
    task_id: str,
    task_prompt: str,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    extras = extras or {}
    run_id = new_run_id()
    log = EventLog(run_id=run_id, benchmark=benchmark, task_id=task_id,
                   out_dir=traces_dir())

    root = log.emit(
        "run.started",
        question=task_prompt,
        model=DEFAULT_MODEL,
        extras=extras,
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": task_prompt},
    ]

    raw_content = ""
    finish_reason = None
    last_result: dict[str, Any] | None = None

    for iteration in range(1, MAX_ITERATIONS + 1):
        call_id = log.emit(
            "llm.requested",
            parent=root,
            iteration=iteration,
            model=DEFAULT_MODEL,
            messages=messages,
        )
        try:
            result = chat(messages=messages, tools=TOOL_SPECS, tool_choice="auto",
                          max_tokens=PER_TURN_MAX_TOKENS)
        except Exception as e:
            log.emit("llm.failed", parent=call_id, error=repr(e))
            log.emit("run.failed", parent=root, error=repr(e))
            log.close()
            return {"run_id": run_id, "answer": None, "error": repr(e),
                    "trace_path": str(log.path)}

        last_result = result
        raw_content = result.get("content") or ""
        finish_reason = result.get("finish_reason")

        log.emit(
            "llm.responded",
            parent=call_id,
            iteration=iteration,
            content=raw_content,
            finish_reason=finish_reason,
            tool_calls=result.get("tool_calls"),
            usage=result.get("usage"),
        )

        messages.append(result["assistant_message"])

        tool_calls = result.get("tool_calls") or []
        if not tool_calls:
            break

        for tc in tool_calls:
            tool_name = tc.get("name") or ""
            tool_args = tc.get("arguments") if tc.get("arguments") is not None else {}
            tool_call_id = tc.get("id") or f"call_{iteration}_{tool_name}"
            tcall_id = log.emit(
                "tool.called",
                parent=call_id,
                iteration=iteration,
                name=tool_name,
                args=tool_args,
                tool_call_id=tool_call_id,
            )
            tool_result = dispatch_tool(tool_name, tool_args if isinstance(tool_args, dict) else {})
            log.emit(
                "tool.responded",
                parent=tcall_id,
                name=tool_name,
                result=tool_result,
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": tool_result,
            })
    else:
        log.emit("run.exhausted_iterations", parent=root, iterations=MAX_ITERATIONS)

    answer = _extract_final_answer(raw_content)
    log.emit(
        "answer.emitted",
        parent=root,
        answer=answer,
        raw_answer=raw_content,
        finish_reason=finish_reason,
        iterations_used=iteration if last_result else 0,
    )

    return {
        "run_id": run_id,
        "answer": answer,
        "error": None,
        "trace_path": str(log.path),
        "log": log,
        "root_event_id": root,
    }
