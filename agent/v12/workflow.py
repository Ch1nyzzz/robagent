"""v12 workflow — v11 + multi-source deterministic retrieval.

Retrieval flow (NEEDS_RETRIEVAL route):
1. plan_retrieval (LLM, unchanged)
2. wikipedia_search(query) -> up to 2 wiki pages opened as sources
3. If 0 wiki hits, fan out to (in order, until ≥1 source):
   - duckduckgo_html_search(query) -> up to 3 snippet sources
   - arxiv_search(query) -> up to 3 abstract sources
4. If still 0 hits, deterministically rewrite the query via
   `rewrite_query_variants` and replay step 2 (wiki) + step 3 (ddg+arxiv).
5. answer_with_evidence on whatever sources accumulated (≤4)
6. Same claim/source verification as v11

No new LLM nodes; only deterministic dispatch around the existing ones.
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
    rewrite_query_variants,
    verify_claim_graph,
)
from .state import AgentState
from .tools import (
    arxiv_search,
    duckduckgo_html_search,
    gaia_file_path,
    read_gaia_file,
    vision_describe,
    wikipedia_fetch,
    wikipedia_search,
)


_IMAGE_EXTS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}
_MAX_SOURCES = 4


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


def _wiki_sources(query: str, log: EventLog, root: str, k: int = 2) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    hits = wikipedia_search(query, log=log, parent=root, limit=k)
    for h in hits[:k]:
        if not h.get("title"):
            continue
        src = wikipedia_fetch(h["title"], log=log, parent=root)
        if src.get("ok"):
            out.append(src)
    return out


def _multi_source_retrieve(
    question: str, query: str, log: EventLog, root: str
) -> list[dict[str, Any]]:
    """Fan out across (wiki, duckduckgo, arxiv) with deterministic query
    rewriting. Returns up to _MAX_SOURCES source dicts."""
    sources: list[dict[str, Any]] = []
    variants = rewrite_query_variants(question, query, k=3)
    # Wikipedia first (highest precision for entities we want)
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
    # DuckDuckGo HTML (snippets — broad coverage)
    for v in variants:
        if len(sources) >= _MAX_SOURCES:
            break
        hits = duckduckgo_html_search(v, log=log, parent=root, limit=3)
        sources.extend(hits[: _MAX_SOURCES - len(sources)])
        if hits:
            break
    if len(sources) >= _MAX_SOURCES:
        return sources[:_MAX_SOURCES]
    # arXiv (high-precision for paper / preprint questions)
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

    src = read_gaia_file(task_id, file_name, log=log, parent=root)
    if not src.get("ok"):
        log.emit(
            "agent.blocked",
            parent=root,
            reason=f"needs_file_capability:{src.get('reason','unknown')}",
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
