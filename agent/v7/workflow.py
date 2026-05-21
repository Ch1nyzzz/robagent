"""v7 workflow — v6 plus file reading for NEEDS_FILE route.

For NEEDS_FILE: read_gaia_file -> answer_with_evidence -> verify_claim_graph.
If file resolution / parsing fails (image, audio, etc.), stay BLOCKED.
"""
from __future__ import annotations

from typing import Any

from agent.events import EventLog
from agent.llm import DEFAULT_MODEL

from .llm_nodes import (
    answer_direct,
    answer_with_evidence,
    answer_with_parametric,
    plan_retrieval,
)
from .reducers import (
    extract_final_answer_strict,
    infer_answer_shape,
    is_unknown_response,
    normalize_answer,
    reshape_answer,
    route_by_extras,
    verify_claim_graph,
)
from .state import AgentState
from .tools import read_gaia_file, wikipedia_fetch, wikipedia_search


def _run_llm(log: EventLog, root: str, node_name: str, fn, *args, **kwargs) -> dict[str, Any]:
    call = log.emit("llm.requested", parent=root, model=DEFAULT_MODEL, node=node_name)
    result = fn(*args, **kwargs)
    log.emit(
        "llm.responded",
        parent=call,
        content=result.get("content", ""),
        finish_reason=result.get("finish_reason"),
        usage=result.get("usage"),
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


def _answer_from_sources(
    log: EventLog,
    root: str,
    state: AgentState,
    sources: list[dict[str, Any]],
) -> AgentState:
    opened_ids = {s["source_id"] for s in sources if s.get("source_id")}
    result = _run_llm(
        log, root, "answer_with_evidence", answer_with_evidence, state.question, sources
    )
    claim_obj = {
        "claim": result.get("claim", ""),
        "source_id": result.get("source_id"),
        "source_quote": result.get("source_quote", ""),
        "confidence": result.get("confidence", 0.0),
    }
    log.emit("claim.extracted", parent=root, **claim_obj)
    verdict = verify_claim_graph(claim_obj, opened_ids)
    log.emit("claim.verified", parent=root, ok=verdict["ok"], reason=verdict["reason"])
    if not verdict["ok"]:
        log.emit("agent.blocked", parent=root, reason=f"claim_unverified:{verdict['reason']}")
        return state.advance("BLOCKED", answer="", block_reason="claim_unverified")
    final, empty = _finalize_answer(log, root, claim_obj["claim"], state.question)
    if empty:
        log.emit("agent.blocked", parent=root, reason="empty_after_reasoning")
        return state.advance("BLOCKED", answer="", block_reason="empty_after_reasoning")
    return state.advance("SUBMIT", answer=final)


def _handle_needs_retrieval(log: EventLog, root: str, state: AgentState) -> AgentState:
    plan = _run_llm(log, root, "plan_retrieval", plan_retrieval, state.question)
    query = plan.get("query", "") or state.question[:120]
    log.emit("retrieval.planned", parent=root, query=query)
    hits = wikipedia_search(query, log=log, parent=root, limit=2)
    sources: list[dict[str, Any]] = []
    for h in hits[:2]:
        if not h.get("title"):
            continue
        src = wikipedia_fetch(h["title"], log=log, parent=root)
        if src.get("ok"):
            sources.append(src)
    if not sources:
        log.emit("retrieval.empty", parent=root)
        return _handle_parametric_fallback(log, root, state)
    new_state = _answer_from_sources(log, root, state, sources)
    # If claim verification failed, fall back to parametric.
    if new_state.stage == "BLOCKED" and new_state.block_reason == "claim_unverified":
        return _handle_parametric_fallback(log, root, state)
    return new_state


def _handle_needs_file(log: EventLog, root: str, state: AgentState) -> AgentState:
    file_name = (state.extras or {}).get("file_name") or ""
    task_id = (state.extras or {}).get("task_id") or ""
    src = read_gaia_file(task_id, file_name, log=log, parent=root)
    if not src.get("ok"):
        log.emit(
            "agent.blocked",
            parent=root,
            reason=f"needs_file_capability:{src.get('reason', 'unknown')}",
        )
        return state.advance("BLOCKED", answer="", block_reason="needs_file_capability")
    return _answer_from_sources(log, root, state, [src])


def _handle_parametric_fallback(log: EventLog, root: str, state: AgentState) -> AgentState:
    result = _run_llm(
        log, root, "answer_with_parametric", answer_with_parametric, state.question
    )
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
        log.emit(
            "agent.blocked",
            parent=root,
            reason="empty_after_reasoning",
            finish_reason=state.finish_reason,
        )
        return state.advance("BLOCKED", answer="", block_reason="empty_after_reasoning")
    return state.advance("SUBMIT", answer=final)


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
        return _handle_needs_file(log, root, state)

    if state.route == "NEEDS_RETRIEVAL":
        return _handle_needs_retrieval(log, root, state)

    result = _run_llm(log, root, "answer_direct", answer_direct, state.question)
    state = state.advance(
        "ANSWERED",
        raw_answer=result["content"],
        finish_reason=result["finish_reason"],
    )
    final, empty = _finalize_answer(log, root, state.raw_answer, state.question)
    if empty:
        log.emit(
            "agent.blocked",
            parent=root,
            reason="empty_after_reasoning",
            finish_reason=state.finish_reason,
        )
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

    extras = dict(extras or {})
    extras["task_id"] = task_id

    root = log.emit(
        "run.started",
        question=task_prompt,
        model=DEFAULT_MODEL,
        extras=extras,
        agent_version="v7",
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
