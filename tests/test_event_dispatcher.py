"""End-to-end test for the Phase-B event dispatcher + EventContext.

Covers:
  - basic emit → matcher → handler → apply_decision pipeline
  - priority ordering within an event bucket
  - BLOCK short-circuits subsequent components
  - mount → event-name alias (legacy components default listens=mount.value)
  - explicit listens="custom_event_name" works
  - in-handler ctx.emit re-enters the dispatcher synchronously
  - recursion depth cap fires when components form a loop

Smoke-only — full sibling integration lands in Phase C/D when the
runtime swaps its ad-hoc `_dispatch_*` calls for `dispatcher.emit`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.component_runtime.types import (  # noqa: E402
    Component, ComponentClass, ComponentContext, Decision, DecisionKind,
    Mount, Trust,
)
from meta_harness.component_runtime_core.dispatcher import Dispatcher  # noqa: E402
from meta_harness.component_runtime_core.event_context import EventContext  # noqa: E402


# --------------------------------------------------------------------------- #
# Test helpers
# --------------------------------------------------------------------------- #

def _trust() -> Trust:
    return Trust(
        evidence_anchor="test anchor",
        blast_radius="local",
        rollback_when="test cleanup",
    )


def _make_component(
    *,
    name: str,
    mount: Mount,
    listens: str = "",
    priority: int = 100,
    matcher=None,
    handler=None,
    cls: ComponentClass = ComponentClass.MECHANISM_LAYER,
    emits: tuple[str, ...] = (),
) -> Component:
    return Component(
        name=name,
        cls=cls,
        mount=mount,
        matcher=matcher,
        handler=handler or (lambda ctx: Decision.allow()),
        trust=_trust(),
        priority=priority,
        listens=listens,
        emits=emits,
    )


def _validate(comp, event_name, kind):
    # Permissive validator for tests — accept anything except None.
    if kind is None:
        raise RuntimeError("decision.kind must not be None")


def _apply_default(ctx: ComponentContext, decision: Decision, comp: Component) -> bool:
    """Track which decisions reached apply, with a BLOCK short-circuit."""
    ctx.shared.setdefault("applied", []).append((comp.name, decision.kind.value))
    return decision.kind is DecisionKind.BLOCK


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_emit_fires_matching_subscriber():
    """Basic happy path: emit → matcher True → handler runs → apply called."""
    seen: list[str] = []

    def matcher(ctx):
        return True

    def handler(ctx):
        seen.append("handler-ran")
        return Decision.inject_context("hello")

    comp = _make_component(
        name="c1",
        mount=Mount.PRE_PROMPT_BUILD,
        matcher=matcher,
        handler=handler,
    )
    disp = Dispatcher(
        [comp],
        validate_decision=_validate,
        apply_decision=_apply_default,
    )

    ctx = ComponentContext(
        mount=Mount.PRE_PROMPT_BUILD, benchmark="test", task_id="t1", extras={},
    )
    disp.emit("pre_prompt_build", ctx)

    assert seen == ["handler-ran"]
    assert ctx.shared["applied"] == [("c1", "inject_context")]


def test_priority_ordering_within_bucket():
    """Lower priority fires first."""
    order: list[str] = []

    def make_handler(label):
        def h(ctx):
            order.append(label)
            return Decision.allow()
        return h

    comps = [
        _make_component(name="low", mount=Mount.POST_LLM_RESPONSE,
                        priority=10, handler=make_handler("low")),
        _make_component(name="high", mount=Mount.POST_LLM_RESPONSE,
                        priority=200, handler=make_handler("high")),
        _make_component(name="mid", mount=Mount.POST_LLM_RESPONSE,
                        priority=100, handler=make_handler("mid")),
    ]
    disp = Dispatcher(
        comps,
        validate_decision=_validate,
        apply_decision=_apply_default,
    )
    ctx = ComponentContext(
        mount=Mount.POST_LLM_RESPONSE, benchmark="test", task_id="t1", extras={},
    )
    disp.emit("post_llm_response", ctx)

    assert order == ["low", "mid", "high"]


def test_block_short_circuits_remaining_components():
    """A BLOCK decision stops further components at the same event."""

    def handler_block(ctx):
        return Decision.block("nope")

    def handler_never(ctx):
        raise AssertionError("should not run after BLOCK")

    comps = [
        _make_component(name="first", mount=Mount.PRE_PROMPT_BUILD,
                        priority=10, handler=handler_block),
        _make_component(name="second", mount=Mount.PRE_PROMPT_BUILD,
                        priority=20, handler=handler_never),
    ]
    disp = Dispatcher(
        comps, validate_decision=_validate, apply_decision=_apply_default,
    )
    ctx = ComponentContext(
        mount=Mount.PRE_PROMPT_BUILD, benchmark="test", task_id="t1", extras={},
    )
    disp.emit("pre_prompt_build", ctx)
    assert ctx.shared["applied"] == [("first", "block")]


def test_mount_to_event_name_alias():
    """Legacy component (no explicit listens) subscribes to mount.value."""
    comp = _make_component(name="legacy", mount=Mount.PRE_PROMPT_BUILD)
    assert comp.listens == "pre_prompt_build"  # __post_init__ filled it
    disp = Dispatcher(
        [comp], validate_decision=_validate, apply_decision=_apply_default,
    )
    assert disp.known_events() == ["pre_prompt_build"]


def test_explicit_custom_event_subscription():
    """A component with listens='custom_x' is registered to that event,
    independent of its Mount value."""
    comp = _make_component(
        name="new_style",
        mount=Mount.SESSION_END,  # placeholder; not the actual subscription
        listens="on_length_truncation",
    )
    disp = Dispatcher(
        [comp], validate_decision=_validate, apply_decision=_apply_default,
    )
    assert disp.known_events() == ["on_length_truncation"]
    # And it does NOT fire on the mount-value event.
    ctx = ComponentContext(
        mount=Mount.SESSION_END, benchmark="test", task_id="t1", extras={},
    )
    disp.emit("session_end", ctx)
    assert "applied" not in ctx.shared


def test_in_handler_emit_reenters_dispatcher():
    """ctx.emit() from within a handler re-fires another component listening
    to the custom event, synchronously, in the same task.

    Uses EventContext directly because Phase C wires sibling
    ComponentContext → EventContext inheritance; in Phase B the
    dispatcher's `ctx` is duck-typed (Any), so any context exposing
    `.shared` and `.emit` works.
    """
    fire_log: list[str] = []

    def a_handler(ctx):
        fire_log.append("A")
        ctx.emit("on_a_done", payload="x")
        return Decision.allow()

    def b_handler(ctx):
        fire_log.append("B")
        return Decision.allow()

    comp_a = _make_component(
        name="A", mount=Mount.PRE_PROMPT_BUILD, handler=a_handler,
        emits=("on_a_done",),
    )
    comp_b = _make_component(
        name="B", mount=Mount.SESSION_END, listens="on_a_done",
        handler=b_handler,
    )
    disp = Dispatcher(
        [comp_a, comp_b], validate_decision=_validate, apply_decision=_apply_default,
    )

    ctx = EventContext(event="pre_prompt_build", task_id="t1")
    ctx._impl_emit = lambda name, fields: disp.emit(name, ctx)  # noqa: SLF001
    disp.emit("pre_prompt_build", ctx)

    assert fire_log == ["A", "B"]


def test_emit_recursion_depth_cap():
    """A component that emits the event it listens to triggers the cap."""

    def looping_handler(ctx):
        ctx.emit("loop", payload="x")
        return Decision.allow()

    comp = _make_component(
        name="loop_comp", mount=Mount.PRE_PROMPT_BUILD, listens="loop",
        handler=looping_handler,
    )
    disp = Dispatcher(
        [comp], validate_decision=_validate, apply_decision=_apply_default,
        max_recursion_depth=5,
    )

    ctx = EventContext(event="loop", task_id="t1")
    ctx._impl_emit = lambda name, fields: disp.emit(name, ctx)  # noqa: SLF001
    with pytest.raises(RuntimeError, match="recursion depth"):
        disp.emit("loop", ctx)
