"""v13 workflow — v12 + deterministic readers for mp3 / zip / pdb.

Only `_handle_needs_file` changes: when the file extension matches v13's
extra-reader set, dispatch through `read_extra_file` and surface the result
through the same (claim, source) pipeline.
"""
from __future__ import annotations

from typing import Any

from agent.events import EventLog
from agent.llm import DEFAULT_MODEL
from agent.v12.workflow import (
    _IMAGE_EXTS,
    _MAX_SOURCES,
    _answer_from_sources,
    _finalize_answer,
    _handle_needs_retrieval,
    _handle_parametric_fallback,
    _multi_source_retrieve,
    _run_llm,
    _wiki_sources,
)

from .llm_nodes import answer_direct
from .reducers import (
    extract_final_answer_strict,
    infer_answer_shape,
    normalize_answer,
    reshape_answer,
    route_by_extras,
)
from .state import AgentState
from .tools import (
    gaia_file_path,
    read_extra_file,
    read_gaia_file,
    vision_describe,
)

_EXTRA_EXTS = {"mp3", "wav", "m4a", "flac", "ogg", "zip", "pdb"}


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
