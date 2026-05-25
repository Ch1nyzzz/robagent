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


def _apply_decision(ctx: ComponentContext, decision: "Decision",
                    comp: "Component") -> bool:
    """Unified decision applier for every toolathlon event. Event-aware
    so the same DecisionKind means the right thing per lifecycle point.

      INJECT_CONTEXT @ setup events       → accumulate into BOTH
            ctx.proposed_system_prompt (so subsequent subscribers see
            prior contributions) AND ctx.shared['tier1_prompt_inject']
            (so the outer setup_agent code can drain the list).
      INJECT_CONTEXT @ user_prompt_submit → ctx.shared['_toolathlon_user_prompt_inject']
      INJECT_CONTEXT @ post_tool_use*     → ctx.shared['_toolathlon_post_tool_injections']
                                            (drained by fire_post_tool_use_inline)
      INJECT_CONTEXT @ other (legacy)     → ctx.shared[_PENDING_KEY] queue
      REWRITE_TOOL_ARGS @ pre_tool_use*   → rewrite ctx.tool_call (dict shape)
                                            and mark _toolathlon_args_rewritten
      BLOCK @ pre_tool_use*               → mark _toolathlon_block_reason,
                                            STOP further subscribers; the
                                            wrapper returns Decision.block()
      BLOCK @ anything else               → ctx.blocked, terminate.
    """
    event = ctx.event or ""
    kind = decision.kind
    if kind is DecisionKind.ALLOW:
        return False

    if kind is DecisionKind.INJECT_CONTEXT:
        text = str(decision.payload or "")
        if event in ("task_received", "pre_context_build", "session_start",
                     "pre_agent_construct", "pre_llm_request"):
            if ctx.proposed_system_prompt is not None:
                ctx.proposed_system_prompt = (
                    ctx.proposed_system_prompt + "\n\n" + text
                )
            ctx.shared.setdefault("tier1_prompt_inject", []).append(text)
        elif event == "user_prompt_submit":
            ctx.shared.setdefault(
                "_toolathlon_user_prompt_inject", []
            ).append(text)
        elif event in ("post_tool_use", "post_tool_result_raw"):
            ctx.shared.setdefault(
                "_toolathlon_post_tool_injections", []
            ).append(text)
        else:
            ctx.shared.setdefault(_PENDING_KEY, []).append(text)
        return False

    if kind is DecisionKind.REWRITE_TOOL_ARGS:
        # toolathlon's ctx.tool_call is a dict {"name", "arguments"}.
        if event in ("pre_tool_use", "pre_tool_arg_validation"):
            tc = ctx.tool_call
            if isinstance(tc, dict):
                ctx.tool_call = {
                    "name": tc.get("name"),
                    "arguments": dict(decision.payload),
                }
                ctx.shared["_toolathlon_args_rewritten"] = True
        return False

    if kind is DecisionKind.BLOCK:
        if event in ("pre_tool_use", "pre_tool_arg_validation"):
            ctx.shared["_toolathlon_block_reason"] = (
                decision.reason or f"{comp.name}: block"
            )
            return True
        ctx.blocked = True
        ctx.blocked_reason = decision.reason or f"{comp.name}: block"
        return True

    # DEFER is rejected at registration in v2; if we get here, treat as ALLOW.
    return False


# Legacy alias preserved for external introspection.
_apply_tier1_decision = _apply_decision


def _validate_event(comp: "Component", event_name: str,
                    kind: DecisionKind) -> None:
    validate_decision(comp.cls, event_name, kind)


# Legacy alias.
_validate_for_tier1 = _validate_event


# --- trace sink --------------------------------------------------------------


def _trace_dir() -> Path:
    tag = os.environ.get("COMPONENT_RUN_TAG", "default")
    d = Path(os.environ.get("COMPONENT_STATE_DIR", ".component-state-toolathlon")) / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


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
        components: list[Component],
        session_state: dict[str, dict],
        domain_policy: str,
        tool_names: tuple[str, ...],
    ):
        self._session_state = session_state
        self._domain_policy = domain_policy
        self._tool_names = tool_names

        # Single core dispatcher — components bucketed by their string
        # `listens` field. Tool-wrapping fire_*_with_args / inline
        # surfaces below route every PRE_TOOL_USE / POST_TOOL_USE call
        # through this same dispatcher.
        self._core = _CoreDispatcher(
            list(components),
            validate_decision=_validate_event,
            apply_decision=_apply_decision,
            trace_sink=_trace_event,
        )

    def _ctx(self, event_name: str, shared: dict,
              **extra: Any) -> ComponentContext:
        ctx = ComponentContext(
            domain_policy=self._domain_policy,
            tool_names=self._tool_names,
            history=list(shared.get("_logs_snapshot", []) or []),
            shared=shared,
            state=self._session_state,
            event=event_name,
            **extra,
        )
        ctx._impl_chat = _make_chat_impl()
        ctx._impl_emit = lambda name, payload: self._core.emit(name, ctx)
        return ctx

    # ---- Event surface --------------------------------------------------

    def make_tier1_ctx(self, event_name: str, shared: dict, **extra: Any) -> ComponentContext:
        """Build a ComponentContext with capability hooks wired. Kept
        under the `make_tier1_ctx` name for backward-compat with callers;
        internally delegates to `_ctx` which now does the wiring."""
        return self._ctx(event_name, shared, **extra)

    def emit(self, event_name: str, ctx: ComponentContext) -> None:
        """Fire an event through the unified core dispatcher. The 5
        SDK-internal Tier-1 events (`pre_llm_request`,
        `pre_tool_arg_validation`, `post_tool_result_raw`, `on_tool_error`,
        `on_no_tool_call_emitted`) are NOT emitted by the toolathlon
        runtime in v1; subscribers to them load but never fire."""
        self._core.emit(event_name, ctx)

    def wire_capabilities(self, ctx: ComponentContext) -> None:
        """Attach per-task capability implementations to a ctx built
        outside `_ctx`. ctx.chat → agent.llm.chat (locked SUT model);
        ctx.emit → re-enter this dispatcher (with depth cap)."""
        ctx._impl_chat = _make_chat_impl()
        ctx._impl_emit = lambda name, payload: self._core.emit(name, ctx)

    # ---- tool-wrapper entry points (v2 path; PRE/POST_TOOL_USE) ----------

    def fire_pre_tool_use_with_args(
        self,
        shared: dict,
        tool_name: str,
        args: dict,
    ) -> Decision:
        """Used by tool_wrappers.py: dispatch PRE_TOOL_USE through the
        unified core dispatcher with REAL args, then translate the
        post-emit ctx state back into a single Decision for the wrapper.

        Multi-component composition: REWRITE_TOOL_ARGS rewrites mutate
        `ctx.tool_call` in place (via `_apply_decision`); subsequent
        subscribers see the previous subscriber's rewritten args.
        BLOCK short-circuits via `_apply_decision` returning stop=True.
        """
        ctx = self._ctx(
            "pre_tool_use", "pre_tool_use", shared,
            tool_call={"name": tool_name, "arguments": dict(args)},
        )
        ctx.shared.pop("_toolathlon_args_rewritten", None)
        ctx.shared.pop("_toolathlon_block_reason", None)
        self._core.emit("pre_tool_use", ctx)

        block_reason = ctx.shared.pop("_toolathlon_block_reason", None)
        if block_reason is not None:
            return Decision.block(block_reason)
        if ctx.shared.pop("_toolathlon_args_rewritten", False) \
                and isinstance(ctx.tool_call, dict):
            return Decision.rewrite_tool_args(
                dict(ctx.tool_call.get("arguments", {}))
            )
        return Decision.allow()

    def fire_post_tool_use_inline(
        self,
        shared: dict,
        tool_name: str,
        args: dict,
        result_str: str,
    ) -> Decision:
        """Used by tool_wrappers.py: dispatch POST_TOOL_USE through the
        unified core dispatcher inline (after real tool invocation,
        before result returns to SDK). `_apply_decision` collects
        INJECT_CONTEXT payloads into `_toolathlon_post_tool_injections`;
        return a single concatenated Decision for the wrapper.
        """
        ctx = self._ctx(
            "post_tool_use", "post_tool_use", shared,
            incoming_message={
                "tool_name": tool_name,
                "args": dict(args),
                "output": result_str,
            },
        )
        ctx.shared.pop("_toolathlon_post_tool_injections", None)
        self._core.emit("post_tool_use", ctx)

        parts = ctx.shared.pop("_toolathlon_post_tool_injections", []) or []
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
    components: list[Component],
    session_state: dict[str, dict],
    domain_policy: str,
    tool_names: tuple[str, ...],
) -> ComponentDispatcher:
    return ComponentDispatcher(components, session_state, domain_policy, tool_names)


def build_hooks(
    components: list[Component],
    session_state: dict[str, dict],
    domain_policy: str,
    tool_names: tuple[str, ...],
) -> tuple[ComponentAgentHooks, ComponentRunHooks, ComponentDispatcher]:
    """Return (agent_hooks, run_hooks, dispatcher). All three share the
    same ComponentDispatcher instance (single source of truth)."""
    disp = ComponentDispatcher(components, session_state, domain_policy, tool_names)
    return ComponentAgentHooks(disp), ComponentRunHooks(disp), disp
