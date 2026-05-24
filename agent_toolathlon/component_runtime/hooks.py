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

from .policy import validate_decision
from .types import (
    Component,
    ComponentContext,
    Decision,
    DecisionKind,
    Mount,
)


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

    # ---- v1 surface (no args; advisory BLOCK only) ----------------------

    def fire_pre_tool_use(self, shared: dict, tool: Any) -> None:
        """SDK on_tool_start: arguments NOT available. We can only match
        by tool name. ALLOW = no-op. BLOCK = advisory inject for next
        turn (see file docstring)."""
        comps = self._by_mount.get(Mount.PRE_TOOL_USE, [])
        if not comps:
            return
        tool_name = getattr(tool, "name", None) or str(tool)
        tool_proxy = {"name": tool_name, "arguments": {}}
        for comp in comps:
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
            if decision.kind is DecisionKind.BLOCK:
                _queue_post_tool_use_text(
                    shared,
                    f"<component_block component={comp.name} tool={tool_name}>\n"
                    f"This tool call should not have been made. "
                    f"Component reason: {decision.reason or '(no reason given)'}\n"
                    f"Do not call this tool again with the same pattern."
                    f"\n</component_block>",
                )

    def fire_post_tool_use(self, shared: dict, tool: Any, result: str) -> None:
        comps = self._by_mount.get(Mount.POST_TOOL_USE, [])
        if not comps:
            return
        tool_name = getattr(tool, "name", None) or str(tool)
        incoming = {"tool_name": tool_name, "output": result}
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
                _queue_post_tool_use_text(shared, str(decision.payload))

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


def _is_cr_wrapped(tool: Any) -> bool:
    """Returns True if `tool` was produced by tool_wrappers.py; the
    wrapper does its own dispatch with real args, so AgentHooks should
    skip to avoid double-firing."""
    return bool(getattr(tool, "_cr_wrapped", False))


class ComponentAgentHooks(AgentHooks):
    """Per-agent hooks. Set on `Agent(hooks=...)` inside `setup_agent`."""

    def __init__(self, dispatcher: ComponentDispatcher):
        super().__init__()
        self._dispatcher = dispatcher

    async def on_tool_start(self, context, agent, tool) -> None:
        if _is_cr_wrapped(tool):
            return  # v2 wrapper already dispatched with real args
        self._dispatcher.fire_pre_tool_use(_shared_from_ctx(context), tool)

    async def on_tool_end(self, context, agent, tool, result) -> None:
        if _is_cr_wrapped(tool):
            return  # v2 wrapper already dispatched and inlined any INJECT
        self._dispatcher.fire_post_tool_use(_shared_from_ctx(context), tool, result)


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
