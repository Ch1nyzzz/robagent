"""GAIA audit gates.

Each gate is a pure function over an event trace (list of event dicts as emitted
by `agent.events.EventLog`). Returns a `GateResult` describing whether the gate
fired and why. Gates are added cumulatively across iterations; a gate is only
removed if it has 0% precision on the new trace set.

All gates here are structural — they reason about event types / shapes,
not about task entity strings. (Static check: AST scan of this file may not
contain any literal that appears with frequency <= 2 in `_task_corpus.txt`.)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


Event = dict[str, Any]


@dataclass(frozen=True)
class GateResult:
    name: str
    fired: bool
    detail: str = ""

    def __bool__(self) -> bool:  # truthy when violation
        return self.fired


def _events_by_type(events: list[Event]) -> dict[str, list[Event]]:
    out: dict[str, list[Event]] = {}
    for e in events:
        out.setdefault(e.get("type", ""), []).append(e)
    return out


def _last_field(events: list[Event], type_: str, key: str, default: Any = None) -> Any:
    for e in reversed(events):
        if e.get("type") == type_:
            return (e.get("fields") or {}).get(key, default)
    return default


# ---------------------------------------------------------------------------
# Gate 1 — truncation
# Fires when an LLM response is `finish_reason=length` and the visible content
# is empty. v0 saw this on 76/165 tasks (all of them wrong). The gate flags
# the trace as a budget bug, not an LLM bug.
# ---------------------------------------------------------------------------
# Nodes whose truncation does NOT invalidate the run (utility / planner nodes
# whose output is consumed by a deterministic fallback). The truncation gate
# only fires when an answer-producing node is truncated.
_UTILITY_NODES = {"plan_retrieval"}


def gate_truncation(events: list[Event]) -> GateResult:
    by = _events_by_type(events)
    for e in by.get("llm.responded", []):
        f = e.get("fields") or {}
        node = f.get("node") or ""
        if node in _UTILITY_NODES:
            continue
        if f.get("finish_reason") == "length" and not (f.get("content") or "").strip():
            return GateResult("gate_truncation", True, "llm.responded with finish_reason=length and empty content")
    return GateResult("gate_truncation", False)


# ---------------------------------------------------------------------------
# Gate 2 — blind file answer
# Fires when the task has a non-empty `file_name` extra, the agent never emits
# a `file.read` event, AND the agent still emits a structured (multi-token)
# final answer. Single-token / binary answers are excluded because the agent
# may guess them correctly by chance, which would violate the precision
# requirement that gates fire on 0 correct tasks.
# ---------------------------------------------------------------------------
import re as _re

_BINARY_ANSWERS = {"yes", "no", "true", "false"}


def _is_structured_answer(s: str) -> bool:
    s = (s or "").strip().lower()
    if not s:
        return False
    if s in _BINARY_ANSWERS:
        return False
    # tokens that look like words / numbers
    toks = _re.findall(r"[A-Za-z0-9]+", s)
    return len(toks) >= 2 or (len(toks) == 1 and len(toks[0]) >= 5)


def gate_blind_file_answer(events: list[Event]) -> GateResult:
    by = _events_by_type(events)
    started = by.get("run.started", [])
    if not started:
        return GateResult("gate_blind_file_answer", False)
    extras = (started[0].get("fields") or {}).get("extras") or {}
    if not extras.get("file_name"):
        return GateResult("gate_blind_file_answer", False)
    # v7+ semantics: a source.opened with kind starting with file:* or vision
    # counts as having read the file. Legacy file.read event also accepted.
    has_file_read = bool(by.get("file.read"))
    if not has_file_read:
        for s in by.get("source.opened", []):
            k = (s.get("fields") or {}).get("kind", "") or ""
            if k.startswith("file:") or k == "vision":
                has_file_read = True
                break
    final = _last_field(events, "answer.emitted", "answer", "")
    if not has_file_read and _is_structured_answer(final or ""):
        return GateResult(
            "gate_blind_file_answer",
            True,
            "file_name present, no file.read or source.opened(file/vision), structured answer emitted",
        )
    return GateResult("gate_blind_file_answer", False)


# ---------------------------------------------------------------------------
# Gate 3 — unsupported source claim
# Fires when the question routed to NEEDS_RETRIEVAL but the agent emitted a
# non-empty answer with no `source.opened` event. This catches fabrication on
# tasks that explicitly cite an authoritative source the agent never accessed.
# ---------------------------------------------------------------------------
def gate_unsupported_source_claim(events: list[Event]) -> GateResult:
    by = _events_by_type(events)
    routed = by.get("task.routed", [])
    if not routed:
        return GateResult("gate_unsupported_source_claim", False)
    route = (routed[0].get("fields") or {}).get("route")
    if route != "NEEDS_RETRIEVAL":
        return GateResult("gate_unsupported_source_claim", False)
    has_source = bool(by.get("source.opened"))
    # If the agent attempted parametric memory (an `answer_with_parametric`
    # LLM call), the answer is sourced from model weights and the gate must
    # not flag it as "fabricated without retrieval".
    used_parametric = any(
        (e.get("fields") or {}).get("node") == "answer_with_parametric"
        for e in by.get("llm.responded", [])
    )
    final = _last_field(events, "answer.emitted", "answer", "")
    if not has_source and not used_parametric and (final or "").strip():
        return GateResult(
            "gate_unsupported_source_claim",
            True,
            "route=NEEDS_RETRIEVAL, no source.opened, no parametric attempt, non-empty answer",
        )
    return GateResult("gate_unsupported_source_claim", False)


# ---------------------------------------------------------------------------
# Gate 4 — empty-after-reasoning (v4+)
# Fires when the LLM consumed budget on reasoning (finish_reason=stop or length)
# but the extractor recovered no answer, and v4 emitted a structured
# `empty_after_reasoning` block. Distinguishes budget-burned-empty from honest
# capability blocks.
# ---------------------------------------------------------------------------
def gate_empty_after_reasoning(events: list[Event]) -> GateResult:
    by = _events_by_type(events)
    for e in by.get("agent.blocked", []):
        if (e.get("fields") or {}).get("reason") == "empty_after_reasoning":
            return GateResult(
                "gate_empty_after_reasoning",
                True,
                "agent emitted empty_after_reasoning block",
            )
    return GateResult("gate_empty_after_reasoning", False)


# ---------------------------------------------------------------------------
# Gate 5 — unsourced claim (v6+)
# Fires when answer.emitted has a non-empty answer for a NEEDS_RETRIEVAL route
# AND there is no preceding claim.extracted event with a matching source.opened
# ancestor (same source_id). This is the (claim, source) protocol enforcer:
# any answer that survives to emission must be traceable to a fetched source.
# ---------------------------------------------------------------------------
def gate_unsourced_claim(events: list[Event]) -> GateResult:
    by = _events_by_type(events)
    routed = by.get("task.routed", [])
    if not routed:
        return GateResult("gate_unsourced_claim", False)
    route = (routed[0].get("fields") or {}).get("route")
    if route != "NEEDS_RETRIEVAL":
        return GateResult("gate_unsourced_claim", False)
    final = _last_field(events, "answer.emitted", "answer", "")
    if not (final or "").strip():
        return GateResult("gate_unsourced_claim", False)
    # parametric path is legitimate (model said it knows)
    used_parametric = any(
        (e.get("fields") or {}).get("node") == "answer_with_parametric"
        for e in by.get("llm.responded", [])
    )
    if used_parametric:
        return GateResult("gate_unsourced_claim", False)
    # Find claim.extracted events; check each has a matching source.opened
    opened_ids = {
        (e.get("fields") or {}).get("source_id")
        for e in by.get("source.opened", [])
    }
    opened_ids.discard(None)
    claims = by.get("claim.extracted", [])
    if not claims:
        return GateResult(
            "gate_unsourced_claim",
            True,
            "NEEDS_RETRIEVAL answer with no claim.extracted event",
        )
    backed = any(
        (c.get("fields") or {}).get("source_id") in opened_ids
        for c in claims
    )
    if not backed:
        return GateResult(
            "gate_unsourced_claim",
            True,
            "claim.extracted source_id has no matching source.opened",
        )
    return GateResult("gate_unsourced_claim", False)


# ---------------------------------------------------------------------------
# Gate 6 — tool argument drift (v6+)
# Fires when a tool.called event has args that include the entire raw task
# question (drift / over-eager) instead of a planned query. Catches the case
# where plan_retrieval returns garbage and we fall through to question itself.
# ---------------------------------------------------------------------------
def gate_tool_argument_drift(events: list[Event]) -> GateResult:
    by = _events_by_type(events)
    started = by.get("run.started", [])
    if not started:
        return GateResult("gate_tool_argument_drift", False)
    question = (started[0].get("fields") or {}).get("question") or ""
    if len(question) < 80:
        return GateResult("gate_tool_argument_drift", False)
    for e in by.get("tool.called", []):
        args = (e.get("fields") or {}).get("args") or {}
        q = args.get("query") or args.get("url") or ""
        if isinstance(q, str) and len(q) > 100 and q.startswith(question[:80]):
            return GateResult(
                "gate_tool_argument_drift",
                True,
                "tool args echo the full question (planner failed)",
            )
    return GateResult("gate_tool_argument_drift", False)


ALL_GATES = [
    gate_truncation,
    gate_blind_file_answer,
    gate_unsupported_source_claim,
    gate_empty_after_reasoning,
    gate_unsourced_claim,
    gate_tool_argument_drift,
]


def run_all(events: list[Event]) -> list[GateResult]:
    return [g(events) for g in ALL_GATES]
