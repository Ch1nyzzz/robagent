"""ComponentAgentHooks + ComponentRunHooks.

These two classes subclass the OpenAI Agents SDK lifecycle hook bases
(`agents.lifecycle.AgentHooks` / `RunHooks`) and dispatch component
matchers/handlers at the two surfaces the SDK actually exposes during a
tool-using run:

  * `on_tool_start(ctx, agent, tool)` → PRE_TOOL_USE
  * `on_tool_end(ctx, agent, tool, result)` → POST_TOOL_USE

The SDK passes a `RunContextWrapper`. Its `.context` attribute is the
dict we own (the task_agent's `shared_context`). We use it to:

  * read snapshots (`history`, etc.) for matchers
  * queue POST_TOOL_USE INJECT_CONTEXT payloads under the key
    `_cr_pending_post_tool_use` so the CrTaskAgent flushes them into
    `self.logs` at the top of the next turn

PRE_TOOL_USE BLOCK is advisory in v1: the SDK has already decided to
call the tool by the time `on_tool_start` fires; we cannot abort the
invocation without wrapping the tool itself. So BLOCK is recorded into
`_cr_pending_post_tool_use` as a strong system note instructing the
model to stop using that tool. v2 will switch to FunctionTool wrapping
for true enforcement.

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


class _Dispatcher:
    """Shared dispatch logic for both AgentHooks and RunHooks subclasses.

    The two SDK hook classes have identical method signatures on the
    surfaces we care about (`on_tool_start`, `on_tool_end`), so we keep
    the matcher/handler loop in one place and instantiate two thin
    wrappers below.
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
                # v1: advisory only. Queue a strong system note for the
                # next turn so the model sees the block reason.
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


# --- SDK hook subclasses -----------------------------------------------------


def _shared_from_ctx(context: Any) -> dict:
    """`context` here is a RunContextWrapper. Its `.context` field is
    whatever we passed as `context=` to Runner.run — i.e. our
    shared_context dict."""
    inner = getattr(context, "context", None)
    if isinstance(inner, dict):
        return inner
    # In rare paths (e.g. Agent-only AgentHooks before Runner.run sets up
    # the wrapper), fall back to an empty dict so we never crash.
    return {}


class ComponentAgentHooks(AgentHooks):
    """Per-agent hooks. Set on `Agent(hooks=...)` inside `setup_agent`."""

    def __init__(self, dispatcher: _Dispatcher):
        super().__init__()
        self._dispatcher = dispatcher

    async def on_tool_start(self, context, agent, tool) -> None:
        self._dispatcher.fire_pre_tool_use(_shared_from_ctx(context), tool)

    async def on_tool_end(self, context, agent, tool, result) -> None:
        self._dispatcher.fire_post_tool_use(_shared_from_ctx(context), tool, result)


class ComponentRunHooks(RunHooks):
    """Per-Runner.run hooks. v1 keeps these empty — the AgentHooks
    subclass above already covers tool_start/tool_end. ComponentRunHooks
    is wired so future mounts (e.g. agent handoff) have a place to live
    without re-plumbing build_agent."""

    def __init__(self, dispatcher: _Dispatcher):
        super().__init__()
        self._dispatcher = dispatcher


# --- factory -----------------------------------------------------------------


def build_hooks(
    by_mount: dict[Mount, list[Component]],
    session_state: dict[str, dict],
    domain_policy: str,
    tool_names: tuple[str, ...],
) -> tuple[ComponentAgentHooks, ComponentRunHooks]:
    disp = _Dispatcher(by_mount, session_state, domain_policy, tool_names)
    return ComponentAgentHooks(disp), ComponentRunHooks(disp)
