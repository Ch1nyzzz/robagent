"""EnterpriseOps dispatcher: runtime side of the EnterpriseOpsAgent component bridge.

Unlike GAIA's component runtime which provides a complete `run_task(...)`
entry point, this runtime exposes a `Dispatcher` that the agent calls at
each lifecycle event in its function-calling loop. The agent owns the loop
(wrapping upstream `orchestrators/react.py`); the dispatcher owns the
policy + mutation logic.

Component activation: every `.py` file inside the sibling `components_<domain>/`
directory is active. There is no workflow YAML or frontier snapshot — the
outer loop maintains frontier state at the directory level (symlink to the
current v_N), and each iter forks the entire directory.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from threading import Lock
from typing import Optional

from agent.llm import chat as _bench_chat
from ballast.component_runtime_core.dispatcher import Dispatcher as _CoreDispatcher

from .policy import validate_decision
from .registry import load_components_from_dir
from .types import (
    Component,
    ComponentContext,
    DecisionKind,
)


_RUN_TAG_ENV  = "ENTERPRISEOPS_COMPONENT_RUN_TAG"
_STATE_ENV    = "ENTERPRISEOPS_COMPONENT_STATE_DIR"


# Process-level cache: one Dispatcher per components directory.
_CACHE_LOCK = Lock()
_CACHE: dict[str, "Dispatcher"] = {}


def _make_chat_impl():
    """ctx.chat helper bound to the locked SUT model. The wrapper turns
    EventContext.chat's `system_override` into a messages-list rewrite."""
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


def _apply_decision_enterpriseops(ctx: ComponentContext, decision,
                                   comp: Component) -> bool:
    """EnterpriseOps-specific decision applier (passed to core.Dispatcher).

    Routing mirrors sopbench (identical mount + Tier-1 event vocabulary).
    BLOCK @ pre_tool_use / pre_tool_arg_validation sets skip-flag without
    halting the loop; BLOCK @ anything else sets ctx.blocked + stop=True.
    """
    event = ctx.event or ""
    kind = decision.kind
    if kind is DecisionKind.ALLOW:
        return False
    if kind is DecisionKind.INJECT_CONTEXT:
        text = str(decision.payload or "")
        if event in (
            "session_start",
            "pre_prompt_build",
            "pre_context_build",
            "task_received",
            "pre_agent_construct",
            "pre_llm_request",
            "pre_llm_turn",
        ):
            ctx.system_prompt = (ctx.system_prompt + "\n\n" + text).strip()
        else:
            ctx.shared.setdefault("post_llm_inject", []).append(text)
        return False
    if kind is DecisionKind.REWRITE:
        payload = decision.payload
        if event in ("pre_prompt_build", "pre_context_build"):
            ctx.user_prompt = str(payload or "")
        elif event == "pre_llm_turn":
            if isinstance(payload, list):
                ctx.messages = list(payload)
        elif event in (
            "post_llm_response",
            "post_llm_response_raw",
            "on_length_truncation",
            "on_empty_response",
            "on_no_tool_call_emitted",
        ):
            ctx.raw_response = str(payload or "")
        elif event in ("pre_tool_use", "pre_tool_arg_validation"):
            if isinstance(payload, dict):
                ctx.current_tool_args = dict(payload)
        elif event in ("post_tool_use", "post_tool_result_raw", "on_tool_error"):
            ctx.current_tool_result_str = str(payload or "")
        elif event == "pre_final_emit":
            ctx.final_output = "" if payload is None else str(payload)
        return False
    if kind is DecisionKind.BLOCK:
        if event in ("pre_tool_use", "pre_tool_arg_validation"):
            ctx.shared["skip_current_tool"] = True
            ctx.shared["skip_current_tool_reason"] = (
                decision.reason or f"{comp.name}: block"
            )
            return False
        ctx.blocked = True
        ctx.blocked_reason = decision.reason or f"{comp.name}: block"
        return True
    return False


def _validate_for_core(comp: Component, event_name: str,
                       kind: DecisionKind) -> None:
    validate_decision(comp.cls, event_name, kind)


class Dispatcher:
    """Dispatcher used by EnterpriseOpsAgent.

    Wraps `core.Dispatcher`; adds `emit(event_name, ctx)` and
    `wire_capabilities(ctx)` for the event-style integrations.
    """

    def __init__(
        self,
        *,
        components: list[Component],
        run_tag: str = "default",
        state_dir: Optional[Path] = None,
    ) -> None:
        self.run_tag = run_tag
        self._state_dir = state_dir
        self._core = _CoreDispatcher(
            list(components),
            validate_decision=_validate_for_core,
            apply_decision=_apply_decision_enterpriseops,
            trace_sink=self._trace_sink,
        )

    def _trace_dir(self) -> Path:
        if self._state_dir is not None:
            d = self._state_dir / self.run_tag
        else:
            base = Path(os.environ.get(_STATE_ENV, ".component-state-enterpriseops"))
            d = base / self.run_tag
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _trace_sink(self, component_name: str, event_name: str,
                    decision_kind_value: str, extra: dict) -> None:
        rec = {
            "ts": time.time(),
            "component": component_name,
            "event": event_name,
            "mount": event_name,
            "decision": decision_kind_value,
            **extra,
        }
        try:
            with (self._trace_dir() / "fired.jsonl").open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass

    def emit(self, event_name: str, ctx: ComponentContext) -> None:
        if event_name == "pre_tool_use":
            ctx.shared.pop("skip_current_tool", None)
        self._core.emit(event_name, ctx)

    def wire_capabilities(self, ctx: ComponentContext) -> None:
        ctx._impl_chat = _make_chat_impl()
        ctx._impl_emit = lambda name, fields: self._core.emit(name, ctx)


def build_dispatcher(
    *,
    comp_dir: Path,
    run_tag: Optional[str] = None,
    state_dir: Optional[Path] = None,
) -> Dispatcher:
    """Return a cached Dispatcher for `comp_dir`.

    Every `.py` file in `comp_dir` (excluding those starting with `_`) is
    loaded. The cache lets parallel-worker eval re-use the same parsed
    Component objects.
    """
    cd = Path(comp_dir)
    tag = run_tag or os.environ.get(_RUN_TAG_ENV, "default")

    key = str(cd.resolve())
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            return hit

        components = load_components_from_dir(cd)

        disp = Dispatcher(
            components=components,
            run_tag=tag,
            state_dir=state_dir,
        )
        _CACHE[key] = disp
        return disp


def clear_dispatcher_cache() -> None:
    """Test helper: drop the process-level dispatcher cache."""
    with _CACHE_LOCK:
        _CACHE.clear()
