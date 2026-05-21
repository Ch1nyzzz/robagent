"""v5 workflow — v4 plus strict extractor."""
from __future__ import annotations

from typing import Any

from agent.events import EventLog
from agent.llm import DEFAULT_MODEL

from .llm_nodes import answer_direct, answer_with_parametric
from .reducers import (
    extract_final_answer_strict,
    infer_answer_shape,
    is_unknown_response,
    normalize_answer,
    reshape_answer,
    route_by_extras,
)
from .state import AgentState


def _run_llm(log: EventLog, root: str, node_name: str, fn, question: str) -> dict[str, Any]:
    call = log.emit("llm.requested", parent=root, model=DEFAULT_MODEL, node=node_name)
    result = fn(question)
    log.emit(
        "llm.responded",
        parent=call,
        content=result["content"],
        finish_reason=result["finish_reason"],
        usage=result["usage"],
        node=node_name,
    )
    return result


def _finalize_answer(log: EventLog, root: str, raw: str, question: str) -> tuple[str, bool]:
    shape = infer_answer_shape(question)
    extracted = extract_final_answer_strict(raw)
    if not extracted.strip():
        return "", True
    reshaped = reshape_answer(extracted, shape)
    final = normalize_answer(reshaped)
    log.emit(
        "answer.normalized",
        parent=root,
        raw_len=len(raw or ""),
        shape=shape,
        extracted=extracted,
        reshaped=reshaped,
        final=final,
    )
    return final, False


def run_workflow(state: AgentState, log: EventLog, root: str) -> AgentState:
    log.emit("task.parsed", parent=root, question_len=len(state.question))
    state = state.advance("PARSED")

    route = route_by_extras(state.question, state.extras)
    log.emit(
        "task.routed",
        parent=root,
        route=route,
        has_file=bool((state.extras or {}).get("file_name")),
    )
    state = state.advance("ROUTED", route=route)

    if state.route == "NEEDS_FILE":
        log.emit("agent.blocked", parent=root, reason="needs_file_capability")
        return state.advance("BLOCKED", answer="", block_reason="needs_file_capability")

    if state.route == "NEEDS_RETRIEVAL":
        result = _run_llm(log, root, "answer_with_parametric", answer_with_parametric, state.question)
        state = state.advance(
            "ANSWERED",
            raw_answer=result["content"],
            finish_reason=result["finish_reason"],
        )
        if is_unknown_response(state.raw_answer):
            log.emit("agent.blocked", parent=root, reason="parametric_unknown")
            return state.advance("BLOCKED", answer="", block_reason="parametric_unknown")
        final, empty = _finalize_answer(log, root, state.raw_answer, state.question)
        if empty:
            log.emit("agent.blocked", parent=root, reason="empty_after_reasoning", finish_reason=state.finish_reason)
            return state.advance("BLOCKED", answer="", block_reason="empty_after_reasoning")
        return state.advance("SUBMIT", answer=final)

    result = _run_llm(log, root, "answer_direct", answer_direct, state.question)
    state = state.advance(
        "ANSWERED",
        raw_answer=result["content"],
        finish_reason=result["finish_reason"],
    )
    final, empty = _finalize_answer(log, root, state.raw_answer, state.question)
    if empty:
        log.emit("agent.blocked", parent=root, reason="empty_after_reasoning", finish_reason=state.finish_reason)
        return state.advance("BLOCKED", answer="", block_reason="empty_after_reasoning")
    return state.advance("SUBMIT", answer=final)


def run_task(
    *,
    benchmark: str,
    task_id: str,
    task_prompt: str,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from agent.events import new_run_id, traces_dir

    run_id = new_run_id()
    log = EventLog(run_id=run_id, benchmark=benchmark, task_id=task_id, out_dir=traces_dir())

    root = log.emit(
        "run.started",
        question=task_prompt,
        model=DEFAULT_MODEL,
        extras=extras or {},
        agent_version="v5",
    )

    state = AgentState(stage="NEW", question=task_prompt, extras=extras or {})

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
