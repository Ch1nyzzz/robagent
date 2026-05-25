"""ComponentDispatcher + ComponentAgentHooks + ComponentRunHooks.

The dispatcher holds the (matcher, handler) loop for every mount the SDK
exposes during a tool-using run, plus the inline variants used by v2's
MCP tool wrapping (`tool_wrappers.py`). Two SDK hook classes wrap it for
the lifecycle surfaces SDK calls directly:

  * `on_tool_start(ctx, agent, tool)` → PRE_TOOL_USE
  * `on_tool_end(ctx, agent, tool, result)` → POST_TOOL_USE

When `tool_wrappers.py` has wrapped MCP tools as FunctionTools, the
wrapper does the dispatch itself with the **real arguments** (which SDK
does not give us at `on_tool_start`). To avoid double-firing, the wrapped
FunctionTool carries a `_cr_wrapped` marker attribute; ComponentAgentHooks
checks for it and skips its own dispatch in that case.

The SDK passes a `RunContextWrapper`. Its `.context` attribute is the
dict we own (the task_agent's `shared_context`). We use it both to read
snapshots and to queue/flush POST_TOOL_USE INJECT_CONTEXT payloads under
`_cr_pending_post_tool_use`.

In v1 (no wrap): PRE_TOOL_USE BLOCK is advisory — queue a system note
for the next outer user turn (single_turn_mode tasks never see this).
POST_TOOL_USE INJECT_CONTEXT is queued the same way (same caveat).

In v2 (wrap on): the wrapper short-circuits BLOCK before the tool runs
and concatenates POST_TOOL_USE INJECT_CONTEXT directly to the tool's
result string — so LLM sees the injection on its next inference,
single_turn or not.

PRE_CONTEXT_BUILD / SESSION_START / USER_PROMPT_SUBMIT are NOT dispatched
from SDK hooks — they happen in CrTaskAgent before the SDK Runner even
sees the turn. See `task_agent.py` for those.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from agents import AgentHooks, RunHooks

from agent.llm import chat as _bench_chat
from meta_harness.component_runtime_core.dispatcher import Dispatcher as _CoreDispatcher

from .policy import validate_decision
from .types import (
    Component,
    ComponentContext,
    Decision,
    DecisionKind,
    Mount,
)


def _make_chat_impl():
    """ctx.chat helper bound to the locked SUT model (agent.llm.chat).
    NOTE: this does NOT route through the OpenAI Agents SDK Runner /
    ModelProvider — it's a direct call to our locked-name chat() so a
    sub-LLM verifier component sees the same model the task SUT does."""
    def _impl(messages, *, max_tokens, temperature, system_override, tools):
        if system_override:
            messages = (
                [{"role": "system", "content": system_override}]
                + [m for m in messages if m.get("role") != "system"]
            )
        return _bench_chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
        )
    return _impl


def _trace_event(component_name: str, event_name: str,
                 decision_kind_value: str, extra: dict) -> None:
    """Trace sink for Tier-1 events (core.Dispatcher signature)."""
    rec = {
        "ts": time.time(),
        "component": component_name,
        "event": event_name,
        "mount": event_name,
        "decision": decision_kind_value,
        **extra,
    }
    try:
        with (_trace_dir() / "fired.jsonl").open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _apply_tier1_decision(ctx: ComponentContext, decision: "Decision",
                          comp: "Component") -> bool:
    """Decision applier for Tier-1 events in toolathlon.

    INJECT_CONTEXT @ pre-LLM events → ctx.shared['tier1_prompt_inject']
                                       (task_agent reads before next turn)
    INJECT_CONTEXT @ post-* events  → ctx.shared[_PENDING_KEY] queue
                                       (already-existing post-tool inject path)
    BLOCK                           → ctx.blocked + stop=True
    Other kinds                     → no-op (Tier-1 events don't directly
                                       rewrite tau2-shaped objects).
    """
    event = ctx.event or ""
    kind = decision.kind
    if kind is DecisionKind.ALLOW:
        return False
    if kind is DecisionKind.INJECT_CONTEXT:
        text = str(decision.payload or "")
        if event in ("task_received", "pre_context_build", "pre_agent_construct",
                     "pre_llm_request"):
            ctx.shared.setdefault("tier1_prompt_inject", []).append(text)
        else:
            ctx.shared.setdefault(_PENDING_KEY, []).append(text)
        return False
    if kind is DecisionKind.BLOCK:
        ctx.blocked = True
        ctx.blocked_reason = decision.reason or f"{comp.name}: block"
        return True
    return False


def _validate_for_tier1(comp: "Component", event_name: str,
                        kind: DecisionKind) -> None:
    validate_decision(comp.cls, event_name, kind)


# --- trace sink --------------------------------------------------------------


def _trace_dir() -> Path:
    tag = os.environ.get("COMPONENT_RUN_TAG", "default")
    d = Path(os.environ.get("COMPONENT_STATE_DIR", ".component-state-toolathlon")) / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def _trace(component_name: str, mount: Mount, decision_kind: DecisionKind,
           extra: dict) -> None:
    rec = {
        "ts": time.time(),
        "component": component_name,
        "mount": mount.value,
        "decision": decision_kind.value,
        **extra,
    }
    with (_trace_dir() / "fired.jsonl").open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --- ctx helpers -------------------------------------------------------------


_PENDING_KEY = "_cr_pending_post_tool_use"


def _queue_post_tool_use_text(shared: dict, text: str) -> None:
    bucket = shared.setdefault(_PENDING_KEY, [])
    bucket.append(text)


def drain_pending_post_tool_use(shared: dict) -> list[str]:
    """Called by CrTaskAgent at the top of every turn. Drains the queue
    of system-note strings to inject into `self.logs`."""
    bucket = shared.get(_PENDING_KEY)
    if not bucket:
        return []
    out = list(bucket)
    shared[_PENDING_KEY] = []
    return out


# --- dispatch core -----------------------------------------------------------


class ComponentDispatcher:
    """Shared dispatch core for both AgentHooks-driven (v1) and
    FunctionTool-wrapper-driven (v2) flows.

    The v1 surface (`fire_pre_tool_use`, `fire_post_tool_use`) is used by
    `ComponentAgentHooks` and only sees `tool_name` (no args). The v2
    surface (`fire_pre_tool_use_with_args`,
    `fire_post_tool_use_inline`) is used by `ComponentMCPToolWrapper` in
    `tool_wrappers.py` and sees real args + result, so PRE_TOOL_USE can
    actually rewrite/block and POST_TOOL_USE injection is returned to
    the wrapper for inline concatenation (immediately visible to the LLM).
    """

    def __init__(
        self,
        by_mount: dict[Mount, list[Component]],
        session_state: dict[str, dict],
        domain_policy: str,
        tool_names: tuple[str, ...],
    ):
        self._by_mount = by_mount
        self._session_state = session_state
        self._domain_policy = domain_policy
        self._tool_names = tool_names

        # Phase B/C/D: parallel core dispatcher for Tier-1 events. The
        # existing v1/v2 `fire_*` paths handle PRE_TOOL_USE / POST_TOOL_USE
        # via tau2-shaped ctx unchanged; Tier-1 events route through here.
        all_comps: list[Component] = [
            c for comps in by_mount.values() for c in comps
        ]
        self._core = _CoreDispatcher(
            all_comps,
            validate_decision=_validate_for_tier1,
            apply_decision=_apply_tier1_decision,
            trace_sink=_trace_event,
        )

    def _ctx(self, mount: Mount, shared: dict, **extra: Any) -> ComponentContext:
        return ComponentContext(
            mount=mount,
            domain_policy=self._domain_policy,
            tool_names=self._tool_names,
            history=list(shared.get("_logs_snapshot", []) or []),
            shared=shared,
            state=self._session_state,
            **extra,
        )

    # ---- Tier-1 event surface (Phase C/D) -------------------------------

    def make_tier1_ctx(self, event_name: str, mount: Mount,
                       shared: dict, **extra: Any) -> ComponentContext:
        """Build a ComponentContext with Tier-1 capability hooks wired."""
        ctx = self._ctx(mount, shared, event=event_name, **extra)
        ctx._impl_chat = _make_chat_impl()
        ctx._impl_emit = lambda name, payload: self._core.emit(name, ctx)
        return ctx

    def emit(self, event_name: str, ctx: ComponentContext) -> None:
        """Fire a Tier-1 event through the parallel dispatcher.

        Called by task_agent.py at setup_agent / post-Runner / save_results
        boundaries. The 5 SDK-internal events (`pre_llm_request`,
        `pre_tool_arg_validation`, `post_tool_result_raw`, `on_tool_error`,
        `on_no_tool_call_emitted`) are NOT emitted in v1; subscribers to
        them load but never fire."""
        self._core.emit(event_name, ctx)

    def wire_capabilities(self, ctx: ComponentContext) -> None:
        """Attach per-task capability implementations to a ctx the legacy
        `fire_*` path constructs. ctx.chat → agent.llm.chat (locked SUT
        model); ctx.emit → re-enter this dispatcher (with depth cap)."""
        ctx._impl_chat = _make_chat_impl()
        ctx._impl_emit = lambda name, payload: self._core.emit(name, ctx)

    # ---- v2 surface (real args; returns Decision to wrapper) ------------

    def fire_pre_tool_use_with_args(
        self,
        shared: dict,
        tool_name: str,
        args: dict,
    ) -> Decision:
        """Used by tool_wrappers.py: dispatch PRE_TOOL_USE with REAL args
        and return the (last-wins) Decision so the wrapper can act on it.

        If no component matches, returns Decision.allow(). If multiple
        components match, REWRITE_TOOL_ARGS / BLOCK from the last one
        wins; earlier components' rewrites compose into the args we pass
        to the next (sequential rewrite). BLOCK short-circuits the loop.
        """
        comps = self._by_mount.get(Mount.PRE_TOOL_USE, [])
        if not comps:
            return Decision.allow()
        current_args = dict(args)
        last_decision = Decision.allow()
        for comp in comps:
            tool_proxy = {"name": tool_name, "arguments": current_args}
            ctx = self._ctx(
                Mount.PRE_TOOL_USE,
                shared,
                tool_call=tool_proxy,
            )
            if comp.matcher is not None and not comp.matcher(ctx):
                continue
            decision = comp.handler(ctx)
            validate_decision(comp.cls, comp.mount, decision.kind)
            _trace(comp.name, comp.mount, decision.kind, {"tool": tool_name})
            if decision.kind is DecisionKind.ALLOW:
                continue
            if decision.kind is DecisionKind.REWRITE_TOOL_ARGS:
                current_args = dict(decision.payload)
                last_decision = Decision(
                    DecisionKind.REWRITE_TOOL_ARGS,
                    payload=current_args,
                    reason=decision.reason,
                )
                continue
            if decision.kind is DecisionKind.BLOCK:
                return decision
            # DEFER is rejected at registration in v2; if we get here
            # something is misconfigured. Fall through to ALLOW.
        return last_decision

    def fire_post_tool_use_inline(
        self,
        shared: dict,
        tool_name: str,
        args: dict,
        result_str: str,
    ) -> Decision:
        """Used by tool_wrappers.py: dispatch POST_TOOL_USE inline (after
        real tool invocation, before result returns to SDK) and collect
        INJECT_CONTEXT payloads into a single concatenated decision.

        Returns a Decision: INJECT_CONTEXT(joined_text) if at least one
        component injected, else ALLOW.
        """
        comps = self._by_mount.get(Mount.POST_TOOL_USE, [])
        if not comps:
            return Decision.allow()
        incoming = {"tool_name": tool_name, "args": dict(args), "output": result_str}
        parts: list[str] = []
        for comp in comps:
            ctx = self._ctx(
                Mount.POST_TOOL_USE,
                shared,
                incoming_message=incoming,
            )
            if comp.matcher is not None and not comp.matcher(ctx):
                continue
            decision = comp.handler(ctx)
            validate_decision(comp.cls, comp.mount, decision.kind)
            _trace(comp.name, comp.mount, decision.kind, {"tool": tool_name})
            if decision.kind is DecisionKind.INJECT_CONTEXT:
                parts.append(str(decision.payload))
        if not parts:
            return Decision.allow()
        return Decision.inject_context("\n\n".join(parts))


# Backwards-compatible alias for the v1 internal name.
_Dispatcher = ComponentDispatcher


# --- SDK hook subclasses -----------------------------------------------------


def _shared_from_ctx(context: Any) -> dict:
    inner = getattr(context, "context", None)
    if isinstance(inner, dict):
        return inner
    return {}


class ComponentAgentHooks(AgentHooks):
    """Per-agent SDK hooks placeholder. v2 dispatches PRE_TOOL_USE /
    POST_TOOL_USE entirely inside `ComponentMCPToolWrapper` (see
    `tool_wrappers.py`) with REAL args + result, so the SDK's
    on_tool_start / on_tool_end have nothing left to do — they would
    only see post-wrap FunctionTools without arguments and would
    double-fire. This subclass keeps the SDK interface populated
    without doing any work; new SDK lifecycle mounts that don't have a
    wrapper-side equivalent (e.g. handoff) can override methods here."""

    def __init__(self, dispatcher: ComponentDispatcher):
        super().__init__()
        self._dispatcher = dispatcher


class ComponentRunHooks(RunHooks):
    """Per-Runner.run hooks. Empty in v1 — present so future mounts
    (handoff, etc.) have a place to live without re-plumbing build_agent."""

    def __init__(self, dispatcher: ComponentDispatcher):
        super().__init__()
        self._dispatcher = dispatcher


# --- factory -----------------------------------------------------------------


def build_dispatcher(
    by_mount: dict[Mount, list[Component]],
    session_state: dict[str, dict],
    domain_policy: str,
    tool_names: tuple[str, ...],
) -> ComponentDispatcher:
    return ComponentDispatcher(by_mount, session_state, domain_policy, tool_names)


def build_hooks(
    by_mount: dict[Mount, list[Component]],
    session_state: dict[str, dict],
    domain_policy: str,
    tool_names: tuple[str, ...],
) -> tuple[ComponentAgentHooks, ComponentRunHooks, ComponentDispatcher]:
    """v2: also return the dispatcher so task_agent / tool_wrappers can
    share the same instance (single source of truth for matcher loops)."""
    disp = ComponentDispatcher(by_mount, session_state, domain_policy, tool_names)
    return ComponentAgentHooks(disp), ComponentRunHooks(disp), disp
