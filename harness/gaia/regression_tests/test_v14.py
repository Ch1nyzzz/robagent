"""v14 regression tests.

Validates the three deterministic refinements applied on top of the v13 stack:
1. vision_describe routes through the rate-slot-aware `chat()` API, fixing
   the missing-`token_estimate` bug.
2. plan_retrieval raises its visible-token budget to 1024.
3. answer_direct retries once at 16384 tokens on empty-length truncation.

Unit-test scope only; no live network calls (chat is monkeypatched).
"""
from __future__ import annotations

from typing import Any

import pytest

from agent.events import EventLog


# --- 1. vision_describe rate-slot wiring -----------------------------------

def test_vision_uses_shared_chat_no_missing_token_estimate(tmp_path, monkeypatch):
    """The v8 bug surfaced as `_acquire_rate_slot() missing 1 required
    positional argument: 'token_estimate'`. v14 routes through `chat()` which
    handles the slot internally. We verify by monkeypatching `chat` and
    asserting it receives a well-formed multimodal message."""
    from agent.v14.tools import vision_v14 as v14_vision

    img = tmp_path / "x.png"
    # tiny 1x1 PNG header bytes (synthetic; vision_describe never parses)
    img.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00"
        b"\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx"
        b"\x9cc\xf8\xcf\xc0\x00\x00\x00\x03\x00\x01\xc6\xdb\xdb\xa6\x00\x00"
        b"\x00\x00IEND\xaeB`\x82"
    )

    captured: dict[str, Any] = {}

    def fake_chat(messages, *, model, max_tokens, **kw):
        captured["messages"] = messages
        captured["model"] = model
        captured["max_tokens"] = max_tokens
        return {
            "content": "the image shows a single red pixel",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 10, "completion_tokens": 9, "total_tokens": 19},
        }

    monkeypatch.setattr(v14_vision, "chat", fake_chat)

    log = EventLog(run_id="r1", benchmark="gaia", task_id="t1", out_dir=tmp_path)
    parent = log.emit("run.started")
    src = v14_vision.vision_describe(str(img), "What color?", log=log, parent=parent)
    log.close()

    assert src["ok"], src
    assert src["kind"] == "vision"
    # multimodal message shape
    assert isinstance(captured["messages"], list)
    assert captured["messages"][0]["role"] == "user"
    parts = captured["messages"][0]["content"]
    assert any(p.get("type") == "image_url" for p in parts)
    assert any(p.get("type") == "text" for p in parts)


def test_vision_marks_non_serverless_as_capability_gap(tmp_path, monkeypatch):
    """When Together returns 400 'Unable to access non-serverless model ...',
    v14 surfaces a clean `model_capability_gap:vision_unavailable` reason so
    the workflow does not loop and the failure mode is correctly classified
    per §IV (provider gap, not code bug)."""
    from agent.v14.tools import vision_v14 as v14_vision

    img = tmp_path / "z.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")

    def fake_400(messages, *, model, max_tokens, **kw):
        raise RuntimeError(
            "Error code: 400 - Unable to access non-serverless model "
            "meta-llama/Llama-Vision-Free. Please visit https://api.together"
        )

    monkeypatch.setattr(v14_vision, "chat", fake_400)

    log = EventLog(run_id="rcap", benchmark="gaia", task_id="tcap", out_dir=tmp_path)
    parent = log.emit("run.started")
    src = v14_vision.vision_describe(str(img), "irrelevant", log=log, parent=parent)
    log.close()

    assert not src["ok"]
    assert src["reason"] == "model_capability_gap:vision_unavailable"


def test_vision_propagates_chat_exception_as_blocked_reason(tmp_path, monkeypatch):
    from agent.v14.tools import vision_v14 as v14_vision

    img = tmp_path / "y.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0")  # JPEG SOI bytes — content irrelevant

    def boom(messages, *, model, max_tokens, **kw):
        raise RuntimeError("synthetic provider 500")

    monkeypatch.setattr(v14_vision, "chat", boom)

    log = EventLog(run_id="r2", benchmark="gaia", task_id="t2", out_dir=tmp_path)
    parent = log.emit("run.started")
    src = v14_vision.vision_describe(str(img), "irrelevant", log=log, parent=parent)
    log.close()

    assert not src["ok"]
    # error string is wrapped under `vision_error:` so workflow surfaces it
    assert "vision_error" in src["reason"]
    # The new failure mode must NOT be the v8 signature bug
    assert "_acquire_rate_slot" not in src["reason"]
    assert "token_estimate" not in src["reason"]


# --- 2. plan_retrieval max_tokens raised ------------------------------------

def test_plan_retrieval_max_tokens_is_1024():
    from agent.v14.llm_nodes.plan_retrieval_v14 import MAX_TOKENS

    assert MAX_TOKENS == 1024


def test_plan_retrieval_calls_chat_with_1024(monkeypatch):
    from agent.v14.llm_nodes import plan_retrieval_v14 as pr14

    seen: dict[str, Any] = {}

    def fake_chat(messages, *, model, max_tokens, **kw):
        seen["max_tokens"] = max_tokens
        return {
            "content": '{"query": "Finding Nemo"}',
            "finish_reason": "stop",
            "usage": {},
        }

    monkeypatch.setattr(pr14, "chat", fake_chat)
    out = pr14.plan_retrieval("Where was Nemo found?")
    assert seen["max_tokens"] == 1024
    assert out["query"] == "Finding Nemo"


def test_plan_retrieval_falls_back_to_regex_on_empty_content(monkeypatch):
    from agent.v14.llm_nodes import plan_retrieval_v14 as pr14

    def empty_chat(messages, *, model, max_tokens, **kw):
        return {"content": "", "finish_reason": "length", "usage": {}}

    monkeypatch.setattr(pr14, "chat", empty_chat)
    out = pr14.plan_retrieval(
        "Which paper did Alan Turing publish about computable numbers?"
    )
    # Regex falls back to a capitalized span (the v9 fallback semantics carry)
    assert isinstance(out.get("query"), str) and out["query"]
    # query must NOT be the whole question
    assert len(out["query"]) <= 100


# --- 3. answer_direct retry on empty-length -------------------------------

def test_answer_direct_no_retry_on_success(monkeypatch):
    from agent.v14.llm_nodes import answer_direct_v14 as ad14

    calls: list[int] = []

    def fake_chat(messages, *, model, max_tokens, **kw):
        calls.append(max_tokens)
        return {
            "content": "42",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        }

    monkeypatch.setattr(ad14, "chat", fake_chat)
    out = ad14.answer_direct("What is 6 times 7?")
    assert calls == [8192]
    assert out["content"] == "42"


def test_answer_direct_retries_on_empty_length(monkeypatch):
    from agent.v14.llm_nodes import answer_direct_v14 as ad14

    calls: list[int] = []
    responses = [
        {"content": "  ", "finish_reason": "length", "usage": {"completion_tokens": 8192}},
        {"content": "answer", "finish_reason": "stop", "usage": {"completion_tokens": 1}},
    ]

    def fake_chat(messages, *, model, max_tokens, **kw):
        calls.append(max_tokens)
        return responses.pop(0)

    monkeypatch.setattr(ad14, "chat", fake_chat)
    out = ad14.answer_direct("Long reasoning task here.")
    assert calls == [8192, 16384]
    assert out["content"] == "answer"
    assert out["primary_finish_reason"] == "length"
    assert out["primary_completion_tokens"] == 8192


def test_answer_direct_no_retry_on_stop_with_empty(monkeypatch):
    """If the model emits stop+empty (refusal-ish), we don't retry — the empty
    is a deliberate signal, not a budget cliff. The downstream
    `_finalize_answer` will surface it as `empty_after_reasoning`."""
    from agent.v14.llm_nodes import answer_direct_v14 as ad14

    calls: list[int] = []

    def fake_chat(messages, *, model, max_tokens, **kw):
        calls.append(max_tokens)
        return {"content": "", "finish_reason": "stop", "usage": {}}

    monkeypatch.setattr(ad14, "chat", fake_chat)
    out = ad14.answer_direct("trivial")
    assert calls == [8192]
    assert out["content"] == ""
    assert out["finish_reason"] == "stop"


# --- workflow wiring smoke test --------------------------------------------

def test_v14_workflow_uses_v14_overlays():
    from agent.v14.workflow import (
        answer_direct as wf_answer_direct,
        plan_retrieval as wf_plan_retrieval,
        vision_describe as wf_vision_describe,
    )
    from agent.v14.llm_nodes.plan_retrieval_v14 import plan_retrieval as v14_pr
    from agent.v14.llm_nodes.answer_direct_v14 import answer_direct as v14_ad
    from agent.v14.tools.vision_v14 import vision_describe as v14_vd

    assert wf_plan_retrieval is v14_pr
    assert wf_answer_direct is v14_ad
    assert wf_vision_describe is v14_vd


def test_v14_inherits_all_v13_tools_and_reducers():
    """Backwards-compat check: v14 must re-export the v13 surface so anything
    that imported `agent.v13.<x>` still has a v14 mirror."""
    from agent.v14 import reducers as r14
    from agent.v14 import tools as t14

    for name in (
        "extract_final_answer",
        "extract_final_answer_strict",
        "infer_answer_shape",
        "is_unknown_response",
        "normalize_answer",
        "reshape_answer",
        "rewrite_query_variants",
        "route_by_extras",
        "verify_claim_graph",
    ):
        assert hasattr(r14, name), f"reducers.{name} missing"
    for name in (
        "arxiv_search",
        "audio_read",
        "duckduckgo_html_search",
        "gaia_file_path",
        "pdb_read",
        "read_extra_file",
        "read_gaia_file",
        "vision_describe",
        "wikipedia_fetch",
        "wikipedia_search",
        "zip_read",
    ):
        assert hasattr(t14, name), f"tools.{name} missing"
