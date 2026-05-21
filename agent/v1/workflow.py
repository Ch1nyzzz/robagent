"""Explicit GAIA agent workflow for v1.

Stages:
  NEW -> parse_task -> PARSED
  PARSED -> route_by_extras -> ROUTED
  ROUTED.DIRECT -> answer_direct (LLM, max_tokens=8192) -> ANSWERED
  ROUTED.{NEEDS_FILE,NEEDS_RETRIEVAL} -> BLOCKED (honest block, structured reason)
  ANSWERED -> extract_final_answer -> normalize_answer -> SUBMIT
  BLOCKED -> SUBMIT (answer empty + block_reason emitted via events)

Every transition is deterministic. The single LLM call is sandwiched between
deterministic prepare (route) and validate (extract + normalize) steps.
"""
from __future__ import annotations

from typing import Any

from agent.events import EventLog
from agent.llm import DEFAULT_MODEL

from .llm_nodes import answer_direct
from .reducers import extract_final_answer, normalize_answer, route_by_extras
from .state import AgentState


def run_workflow(state: AgentState, log: EventLog, root_event_id: str) -> AgentState:
    # NEW -> PARSED (no expensive parsing in v1; placeholder for v2+)
    log.emit("task.parsed", parent=root_event_id, question_len=len(state.question))
    state = state.advance("PARSED")

    # PARSED -> ROUTED
    route = route_by_extras(state.question, state.extras)
    log.emit(
        "task.routed",
        parent=root_event_id,
        route=route,
        has_file=bool((state.extras or {}).get("file_name")),
    )
    state = state.advance("ROUTED", route=route)

    if state.route == "NEEDS_FILE":
        log.emit(
            "agent.blocked",
            parent=root_event_id,
            reason="needs_file_capability",
            file_name=(state.extras or {}).get("file_name", ""),
        )
        return state.advance("BLOCKED", answer="", block_reason="needs_file_capability")

    if state.route == "NEEDS_RETRIEVAL":
        log.emit(
            "agent.blocked",
            parent=root_event_id,
            reason="needs_retrieval_capability",
        )
        return state.advance("BLOCKED", answer="", block_reason="needs_retrieval_capability")

    # ROUTED.DIRECT -> ANSWERED
    call_event = log.emit(
        "llm.requested",
        parent=root_event_id,
        model=DEFAULT_MODEL,
        node="answer_direct",
    )
    try:
        result = answer_direct(state.question)
    except Exception as e:  # pragma: no cover — surfaced as run.failed by caller
        log.emit("llm.failed", parent=call_event, error=repr(e))
        raise

    log.emit(
        "llm.responded",
        parent=call_event,
        content=result["content"],
        finish_reason=result["finish_reason"],
        usage=result["usage"],
    )
    state = state.advance(
        "ANSWERED",
        raw_answer=result["content"],
        finish_reason=result["finish_reason"],
    )

    # ANSWERED -> NORMALIZED -> SUBMIT
    extracted = extract_final_answer(state.raw_answer)
    final = normalize_answer(extracted)
    log.emit(
        "answer.normalized",
        parent=root_event_id,
        raw_len=len(state.raw_answer or ""),
        extracted=extracted,
        final=final,
    )
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
        agent_version="v1",
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
