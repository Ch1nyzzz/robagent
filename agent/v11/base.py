"""v11 base — entry point for run_benchmark.py."""
from __future__ import annotations

from typing import Any

from agent.events import EventLog, new_run_id, traces_dir
from agent.llm import DEFAULT_MODEL

from .state import AgentState
from .workflow import run_workflow


def run_task(
    *,
    benchmark: str,
    task_id: str,
    task_prompt: str,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_id = new_run_id()
    log = EventLog(run_id=run_id, benchmark=benchmark, task_id=task_id, out_dir=traces_dir())

    extras = dict(extras or {})
    extras["task_id"] = task_id

    root = log.emit(
        "run.started",
        question=task_prompt,
        model=DEFAULT_MODEL,
        extras=extras,
        agent_version="v11",
    )

    state = AgentState(stage="NEW", question=task_prompt, extras=extras)

    try:
        state = run_workflow(state, log, root)
    except Exception as e:
        log.emit("run.failed", parent=root, error=repr(e))
        log.close()
        return {"run_id": run_id, "answer": None, "error": repr(e), "trace_path": str(log.path)}

    answer = state.answer or ""
    log.emit(
        "answer.emitted",
        parent=root,
        answer=answer,
        stage=state.stage,
        block_reason=state.block_reason,
        route=state.route,
    )
    return {
        "run_id": run_id,
        "answer": answer,
        "error": None,
        "trace_path": str(log.path),
        "log": log,
        "root_event_id": root,
        "stage": state.stage,
        "block_reason": state.block_reason,
        "route": state.route,
    }
