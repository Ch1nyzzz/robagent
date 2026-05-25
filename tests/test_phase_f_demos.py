"""Phase F smoke: verify the two demonstrator components load + fire end-to-end.

Both demos prove the Phase D Tier-1 event vocabulary + Phase C ctx capability
methods route through dispatch correctly. No real LLM calls — the gaia demo's
ctx.chat is stubbed with a fake response; the toolathlon demo's artifact gate
is a pure filesystem check.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# gaia: component_demo_on_empty_response
# ---------------------------------------------------------------------------


def _load_gaia_demo():
    from agent.component_runtime.registry import load_components_from_dir
    grouped = load_components_from_dir(
        only=["component_demo_on_empty_response"],
    )
    flat = [c for cs in grouped.values() for c in cs]
    assert flat, "gaia demo did not load (registry empty)"
    return flat[0]


def test_gaia_demo_loads_and_passes_policy():
    """Phase A + B + D: the demo's (cls, mount) cell is admitted in policy
    AND its `listens="on_empty_response"` string key is admitted too."""
    comp = _load_gaia_demo()
    assert comp.name == "component_demo_on_empty_response"
    assert comp.listens == "on_empty_response"
    assert "on_empty_response_recovered" in comp.emits

    # Tier-1 string key admission
    from agent.component_runtime.policy import validate_decision
    from agent.component_runtime.types import DecisionKind
    validate_decision(comp.cls, "on_empty_response", DecisionKind.REWRITE)
    validate_decision(comp.cls, "on_empty_response", DecisionKind.ALLOW)


def test_gaia_demo_fires_via_dispatcher_with_stubbed_chat():
    """End-to-end: build a Dispatcher with the demo, emit on_empty_response,
    verify ctx.chat is invoked and Decision.rewrite lands raw_response."""
    from meta_harness.component_runtime_core.dispatcher import Dispatcher
    from agent.component_runtime.policy import validate_decision
    from agent.component_runtime.types import (
        ComponentContext, DecisionKind, Mount,
    )

    comp = _load_gaia_demo()

    # Track that ctx.chat was actually called (proves capability wiring).
    chat_calls: list[dict] = []

    def _fake_chat(messages, *, max_tokens, temperature, system_override, tools):
        chat_calls.append({
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system_override": system_override,
            "tools": tools,
        })
        return {"content": "42", "finish_reason": "stop", "usage": {}}

    # gaia-style apply_decision (lifted minimal subset)
    def _apply(ctx, decision, comp_):
        if decision.kind is DecisionKind.REWRITE:
            ctx.raw_response = str(decision.payload or "")
            return False
        if decision.kind is DecisionKind.BLOCK:
            ctx.blocked = True
            ctx.blocked_reason = decision.reason
            return True
        return False

    disp = Dispatcher(
        [comp],
        validate_decision=lambda c, e, k: validate_decision(c.cls, e, k),
        apply_decision=_apply,
    )

    ctx = ComponentContext(
        mount=Mount.POST_LLM_RESPONSE,
        benchmark="gaia",
        task_id="demo-smoke-1",
        extras={},
        system_prompt="sys",
        prompt="What is the answer to life, the universe, and everything?",
        raw_response="",                        # empty → triggers the event
    )
    ctx._impl_chat = _fake_chat

    disp.emit("on_empty_response", ctx)

    assert chat_calls, "demo did not invoke ctx.chat"
    assert chat_calls[0]["max_tokens"] == 4096
    assert ctx.raw_response == "42", \
        f"demo did not rewrite raw_response; got {ctx.raw_response!r}"
    assert ctx.blocked is False


def test_gaia_demo_handler_returns_allow_when_chat_unwired():
    """If ctx.chat is unwired (capability missing), handler must NOT crash —
    falls through to Decision.allow() so the demo is safe to dry-run."""
    from agent.component_runtime.types import (
        ComponentContext, DecisionKind, Mount,
    )

    comp = _load_gaia_demo()
    ctx = ComponentContext(
        mount=Mount.POST_LLM_RESPONSE,
        benchmark="gaia",
        task_id="demo-smoke-2",
        extras={},
        prompt="x",
        raw_response="",
    )
    # NB: ctx._impl_chat left None → ctx.chat() raises RuntimeError inside
    # the handler; the demo's try/except catches it and returns ALLOW.
    decision = comp.handler(ctx)
    assert decision.kind is DecisionKind.ALLOW


# ---------------------------------------------------------------------------
# toolathlon: component_demo_artifact_gate
# ---------------------------------------------------------------------------


def _load_toolathlon_demo():
    from agent_toolathlon.component_runtime.registry import load_components_from_dir
    grouped = load_components_from_dir(
        "agent_toolathlon/components",
        only=["component_demo_artifact_gate"],
    )
    flat = [c for cs in grouped.values() for c in cs]
    assert flat, "toolathlon demo did not load (registry empty)"
    return flat[0]


def test_toolathlon_demo_loads_and_passes_policy():
    comp = _load_toolathlon_demo()
    assert comp.name == "component_demo_artifact_gate"
    assert comp.listens == "on_explicit_terminate"

    from agent_toolathlon.component_runtime.policy import validate_decision
    from agent_toolathlon.component_runtime.types import DecisionKind
    validate_decision(comp.cls, "on_explicit_terminate", DecisionKind.BLOCK)
    validate_decision(comp.cls, "on_explicit_terminate", DecisionKind.ALLOW)


def test_toolathlon_demo_blocks_when_artifact_missing():
    """User asked for report.md, workspace exists, file absent → BLOCK."""
    from agent_toolathlon.component_runtime.types import (
        ComponentContext, DecisionKind, Mount,
    )

    comp = _load_toolathlon_demo()
    with tempfile.TemporaryDirectory() as tmp:
        ctx = ComponentContext(
            mount=Mount.STOP,
            task_id="demo-art-1",
            history=[
                {"role": "user", "content": "please write a report to report.md"},
            ],
        )
        ctx.shared["_agent_workspace"] = tmp

        assert comp.matcher(ctx) is True, "matcher did not fire on report needle"
        decision = comp.handler(ctx)
        assert decision.kind is DecisionKind.BLOCK
        assert "artifact_gate" in decision.reason


def test_toolathlon_demo_allows_when_artifact_present():
    """Same task shape, but the file exists → ALLOW (termination proceeds)."""
    from agent_toolathlon.component_runtime.types import (
        ComponentContext, DecisionKind, Mount,
    )

    comp = _load_toolathlon_demo()
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "report.md"), "w") as f:
            f.write("# Report\n\nFindings.\n")

        ctx = ComponentContext(
            mount=Mount.STOP,
            task_id="demo-art-2",
            history=[
                {"role": "user", "content": "please write a report to report.md"},
            ],
        )
        ctx.shared["_agent_workspace"] = tmp

        decision = comp.handler(ctx)
        assert decision.kind is DecisionKind.ALLOW


def test_toolathlon_demo_matcher_false_on_unrelated_task():
    """User did not ask for a report → matcher returns False; demo no-ops."""
    from agent_toolathlon.component_runtime.types import (
        ComponentContext, Mount,
    )

    comp = _load_toolathlon_demo()
    ctx = ComponentContext(
        mount=Mount.STOP,
        task_id="demo-art-3",
        history=[
            {"role": "user", "content": "list the prime numbers under 20"},
        ],
    )
    assert comp.matcher(ctx) is False
