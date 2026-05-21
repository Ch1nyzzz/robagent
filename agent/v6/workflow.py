"""v6 workflow — adds Wikipedia retrieval and (claim, source) protocol for NEEDS_RETRIEVAL.

Capability additions:
  - wikipedia_search + wikipedia_fetch tools (free, public REST API)
  - plan_retrieval LLM node (question -> query)
  - answer_with_evidence LLM node (question + sources -> {claim, source_id, confidence})
  - verify_claim_graph deterministic verifier

For NEEDS_RETRIEVAL tasks: retrieve -> extract claim -> verify -> submit.
If verification fails, fall back to answer_with_parametric (preserves v5 behavior).

DIRECT and NEEDS_FILE routes unchanged from v5.
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
from .tools import wikipedia_fetch, wikipedia_search


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


def _retrieve_evidence(
    log: EventLog,
    root: str,
    question: str,
    max_pages: int = 2,
) -> list[dict[str, Any]]:
    """plan -> search -> fetch top pages. Returns list of source dicts."""
    plan = _run_llm(log, root, "plan_retrieval", plan_retrieval, question)
    query = plan.get("query", "") or question[:120]
    log.emit("retrieval.planned", parent=root, query=query)
    hits = wikipedia_search(query, log=log, parent=root, limit=max_pages)
    sources: list[dict[str, Any]] = []
    for h in hits[:max_pages]:
        if not h.get("title"):
            continue
        src = wikipedia_fetch(h["title"], log=log, parent=root)
        if src.get("ok"):
            sources.append(src)
    return sources


def _handle_needs_retrieval(
    log: EventLog,
    root: str,
    state: AgentState,
) -> AgentState:
    sources = _retrieve_evidence(log, root, state.question)
    opened_ids = {s["source_id"] for s in sources if s.get("source_id")}

    if not sources:
        # No evidence retrieved — fall back to parametric memory (v5 behavior).
        log.emit("retrieval.empty", parent=root)
        return _handle_parametric_fallback(log, root, state)

    result = _run_llm(
        log, root, "answer_with_evidence", answer_with_evidence, state.question, sources
    )
    claim_obj = {
        "claim": result.get("claim", ""),
        "source_id": result.get("source_id"),
        "source_quote": result.get("source_quote", ""),
        "confidence": result.get("confidence", 0.0),
    }
    log.emit(
        "claim.extracted",
        parent=root,
        **claim_obj,
    )
    verdict = verify_claim_graph(claim_obj, opened_ids)
    log.emit("claim.verified", parent=root, ok=verdict["ok"], reason=verdict["reason"])

    if not verdict["ok"]:
        # Unsourced/unsupported claim — fall back to parametric.
        log.emit(
            "agent.blocked",
            parent=root,
            reason=f"claim_unverified:{verdict['reason']}",
        )
        return _handle_parametric_fallback(log, root, state)

    # Source-backed answer — normalize and submit.
    final, empty = _finalize_answer(log, root, claim_obj["claim"], state.question)
    if empty:
        log.emit("agent.blocked", parent=root, reason="empty_after_reasoning")
        return state.advance("BLOCKED", answer="", block_reason="empty_after_reasoning")
    return state.advance("SUBMIT", answer=final)


def _handle_parametric_fallback(
    log: EventLog,
    root: str,
    state: AgentState,
) -> AgentState:
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
        log.emit("agent.blocked", parent=root, reason="needs_file_capability")
        return state.advance("BLOCKED", answer="", block_reason="needs_file_capability")

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

    root = log.emit(
        "run.started",
        question=task_prompt,
        model=DEFAULT_MODEL,
        extras=extras or {},
        agent_version="v6",
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
