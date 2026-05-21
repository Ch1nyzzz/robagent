"""v14 workflow — inherits v13's dispatch, replaces the imports that point at
v14's overlays:

- `vision_describe` (v14 tools) — fixes the `_acquire_rate_slot()` signature
  bug by routing the multimodal call through `agent.llm.chat`.
- `plan_retrieval` (v14 llm_nodes) — raises MAX_TOKENS 256 -> 1024 to give
  the reasoning model room to emit the visible JSON.
- `answer_direct` (v14 llm_nodes) — retries once at 16384 on empty-length
  truncation.

No state-machine / route-graph changes; the deterministic flow is identical.
"""
from __future__ import annotations

from typing import Any

from agent.events import EventLog
from agent.llm import DEFAULT_MODEL
from agent.v12.workflow import (
    _IMAGE_EXTS,
    _MAX_SOURCES,
    _finalize_answer,
    _handle_parametric_fallback,
    _run_llm,
    _wiki_sources,
)
from agent.v13.workflow import _EXTRA_EXTS

from .llm_nodes import (
    answer_direct,
    answer_with_evidence,
    answer_with_parametric,
    plan_retrieval,
)
from .reducers import (
    rewrite_query_variants,
    route_by_extras,
    verify_claim_graph,
)
from .state import AgentState
from .tools import (
    arxiv_search,
    duckduckgo_html_search,
    gaia_file_path,
    read_extra_file,
    read_gaia_file,
    vision_describe,
    wikipedia_fetch,
    wikipedia_search,
)


def _multi_source_retrieve(
    question: str, query: str, log: EventLog, root: str
) -> list[dict[str, Any]]:
    """v14: identical fanout to v12, but rebound here so it uses the v14
    wikipedia / ddg / arxiv overlays (currently the same modules, but the
    indirection lets us swap any one of them later without forking v12).
    """
    sources: list[dict[str, Any]] = []
    variants = rewrite_query_variants(question, query, k=3)
    for v in variants:
        if len(sources) >= _MAX_SOURCES:
            break
        sources.extend(_wiki_sources(v, log, root, k=2))
        if sources:
            break
    if len(sources) >= _MAX_SOURCES:
        return sources[:_MAX_SOURCES]
    if not sources:
        log.emit("retrieval.empty", parent=root, source="wikipedia")
    for v in variants:
        if len(sources) >= _MAX_SOURCES:
            break
        hits = duckduckgo_html_search(v, log=log, parent=root, limit=3)
        sources.extend(hits[: _MAX_SOURCES - len(sources)])
        if hits:
            break
    if len(sources) >= _MAX_SOURCES:
        return sources[:_MAX_SOURCES]
    for v in variants:
        if len(sources) >= _MAX_SOURCES:
            break
        hits = arxiv_search(v, log=log, parent=root, limit=2)
        sources.extend(hits[: _MAX_SOURCES - len(sources)])
        if hits:
            break
    return sources[:_MAX_SOURCES]


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
    query = plan.get("query", "") or state.question[:80]
    log.emit("retrieval.planned", parent=root, query=query)
    sources = _multi_source_retrieve(state.question, query, log, root)
    if not sources:
        log.emit("retrieval.empty", parent=root, source="all")
        return _handle_parametric_fallback(log, root, state)
    new_state = _answer_from_sources(log, root, state, sources)
    if new_state.stage == "BLOCKED" and new_state.block_reason == "claim_unverified":
        return _handle_parametric_fallback(log, root, state)
    return new_state


def _handle_needs_file(log: EventLog, root: str, state: AgentState) -> AgentState:
    file_name = (state.extras or {}).get("file_name") or ""
    task_id = (state.extras or {}).get("task_id") or ""
    ext = file_name.lower().rsplit(".", 1)[-1] if "." in file_name else ""

    if ext in _IMAGE_EXTS:
        path = gaia_file_path(task_id, file_name)
        if not path:
            log.emit("agent.blocked", parent=root, reason="needs_file_capability:path_not_found")
            return state.advance("BLOCKED", answer="", block_reason="needs_file_capability")
        src = vision_describe(path, state.question, log=log, parent=root)
        if not src.get("ok"):
            log.emit(
                "agent.blocked",
                parent=root,
                reason=f"needs_file_capability:{src.get('reason','vision_failed')}",
            )
            return state.advance("BLOCKED", answer="", block_reason="needs_file_capability")
        return _answer_from_sources(log, root, state, [src])

    if ext in _EXTRA_EXTS:
        path = gaia_file_path(task_id, file_name)
        if not path:
            log.emit("agent.blocked", parent=root, reason="needs_file_capability:path_not_found")
            return state.advance("BLOCKED", answer="", block_reason="needs_file_capability")
        src = read_extra_file(task_id, file_name, path, ext, log=log, parent=root)
        if not src.get("ok"):
            log.emit(
                "agent.blocked",
                parent=root,
                reason=f"needs_file_capability:{src.get('reason','unknown')}",
            )
            return state.advance("BLOCKED", answer="", block_reason="needs_file_capability")
        return _answer_from_sources(log, root, state, [src])

    src = read_gaia_file(task_id, file_name, log=log, parent=root)
    if not src.get("ok"):
        log.emit(
            "agent.blocked",
            parent=root,
            reason=f"needs_file_capability:{src.get('reason','unknown')}",
        )
        return state.advance("BLOCKED", answer="", block_reason="needs_file_capability")
    return _answer_from_sources(log, root, state, [src])


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
