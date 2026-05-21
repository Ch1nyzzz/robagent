"""Unit tests for v6 — claim graph verifier and gate_unsourced_claim."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from agent.v6.reducers.verify_claim_graph import verify_claim_graph  # noqa: E402
from harness.gaia.audit_gates import (  # noqa: E402
    gate_tool_argument_drift,
    gate_unsourced_claim,
)


def test_verify_claim_graph_ok():
    obj = {"claim": "Paris", "source_id": "S1", "confidence": 0.9}
    out = verify_claim_graph(obj, {"S1", "S2"})
    assert out["ok"] is True


def test_verify_claim_graph_unknown():
    obj = {"claim": "UNKNOWN", "source_id": "S1", "confidence": 0.9}
    out = verify_claim_graph(obj, {"S1"})
    assert out["ok"] is False
    assert out["reason"] == "empty_or_unknown_claim"


def test_verify_claim_graph_missing_source():
    obj = {"claim": "Paris", "source_id": None, "confidence": 0.9}
    out = verify_claim_graph(obj, {"S1"})
    assert out["ok"] is False


def test_verify_claim_graph_unknown_source():
    obj = {"claim": "Paris", "source_id": "S3", "confidence": 0.9}
    out = verify_claim_graph(obj, {"S1", "S2"})
    assert out["ok"] is False
    assert out["reason"] == "source_id_not_in_opened_sources"


def test_verify_claim_graph_low_confidence():
    obj = {"claim": "Paris", "source_id": "S1", "confidence": 0.1}
    out = verify_claim_graph(obj, {"S1"})
    assert out["ok"] is False


def test_gate_unsourced_claim_fires_on_no_claim_extracted():
    events = [
        {"type": "run.started", "fields": {"extras": {}, "question": "test"}},
        {"type": "task.routed", "fields": {"route": "NEEDS_RETRIEVAL"}},
        {"type": "answer.emitted", "fields": {"answer": "something"}},
    ]
    r = gate_unsourced_claim(events)
    assert r.fired


def test_gate_unsourced_claim_passes_on_parametric():
    events = [
        {"type": "run.started", "fields": {"extras": {}, "question": "test"}},
        {"type": "task.routed", "fields": {"route": "NEEDS_RETRIEVAL"}},
        {"type": "llm.responded", "fields": {"node": "answer_with_parametric"}},
        {"type": "answer.emitted", "fields": {"answer": "something"}},
    ]
    assert not gate_unsourced_claim(events).fired


def test_gate_unsourced_claim_passes_when_sourced():
    events = [
        {"type": "run.started", "fields": {"extras": {}, "question": "test"}},
        {"type": "task.routed", "fields": {"route": "NEEDS_RETRIEVAL"}},
        {"type": "source.opened", "fields": {"source_id": "S_wiki_abc"}},
        {"type": "claim.extracted", "fields": {"source_id": "S_wiki_abc", "claim": "x"}},
        {"type": "answer.emitted", "fields": {"answer": "x"}},
    ]
    assert not gate_unsourced_claim(events).fired


def test_gate_unsourced_claim_skips_direct_route():
    events = [
        {"type": "run.started", "fields": {"extras": {}, "question": "test"}},
        {"type": "task.routed", "fields": {"route": "DIRECT"}},
        {"type": "answer.emitted", "fields": {"answer": "something"}},
    ]
    assert not gate_unsourced_claim(events).fired


def test_gate_tool_argument_drift_fires_on_question_echo():
    q = "What is the abstract of the research article that mentions the British Museum item 2012,5015.17?"
    events = [
        {"type": "run.started", "fields": {"question": q}},
        {"type": "tool.called", "fields": {"args": {"query": q + " extra"}, "tool": "wikipedia_search"}},
    ]
    assert gate_tool_argument_drift(events).fired


def test_gate_tool_argument_drift_passes_short_question():
    events = [
        {"type": "run.started", "fields": {"question": "Where is Paris?"}},
        {"type": "tool.called", "fields": {"args": {"query": "Paris France"}, "tool": "wikipedia_search"}},
    ]
    assert not gate_tool_argument_drift(events).fired
