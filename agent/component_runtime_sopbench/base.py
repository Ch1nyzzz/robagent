"""SOP-Bench dispatcher: the runtime side of the SopBenchAgent component bridge.

Unlike GAIA's `agent/component_runtime/base.py` which provides a complete
`run_task(...)` entry point, the SOP-Bench runtime exposes a `Dispatcher`
that SopBenchAgent.execute() calls at each Mount in its FC loop. The agent
owns the loop; the dispatcher owns the policy + mutation logic.

Workflow source on disk:
  ballast/workflows/sopbench_<domain>.yaml  (override with
  `SOPBENCH_COMPONENT_WORKFLOW` env var)

Active component set = workflow's `active_nodes()`. The outer loop keeps
the YAML pinned to the accepted frontier; the runtime trusts it.
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

from .policy import ComponentPolicyError, validate_decision
from .registry import COMPONENTS_DIR_DEFAULT, load_components_from_dir
from .types import (
    Component,
    ComponentContext,
    DecisionKind,
)
from .workflow import Workflow


ROOT = Path(__file__).resolve().parent.parent.parent

_WORKFLOW_ENV = "SOPBENCH_COMPONENT_WORKFLOW"
_NAMES_ENV    = "SOPBENCH_COMPONENT_NAMES"
_RUN_TAG_ENV  = "SOPBENCH_COMPONENT_RUN_TAG"
_DIR_ENV      = "SOPBENCH_COMPONENT_DIR"
_STATE_ENV    = "SOPBENCH_COMPONENT_STATE_DIR"


def resolve_workflow_path(default: Optional[Path] = None) -> Optional[Path]:
    raw = os.environ.get(_WORKFLOW_ENV)
    if raw:
        return Path(raw)
    return default


# Process-level cache: one Dispatcher per workflow+components selection.
_CACHE_LOCK = Lock()
_CACHE: dict[tuple[str, str], "Dispatcher"] = {}


def _make_chat_impl():
    """Build the per-task ctx.chat implementation. The locked SUT model is
    enforced inside `agent.llm.chat` itself (raises on `model=` override).
    The wrapper translates EventContext.chat's `system_override` kwarg into
    a messages-list rewrite so the upstream API accepts it."""
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


def _apply_decision_sopbench(ctx: ComponentContext, decision,
                              comp: Component) -> bool:
    """Sopbench-specific decision applier (passed to core.Dispatcher).

    Event-name routing (covers both legacy mount.value strings and Tier-1
    event names emitted in Phase D):

      INJECT_CONTEXT @ pre-LLM-ish events  → ctx.system_prompt
      INJECT_CONTEXT @ post-LLM-ish / tool events → ctx.shared['post_llm_inject']
      REWRITE        @ pre_prompt_build/pre_context_build       → ctx.user_prompt
      REWRITE        @ pre_llm_turn                             → ctx.messages
      REWRITE        @ post_llm_response[_raw] / on_*           → ctx.raw_response
      REWRITE        @ pre_tool_use / pre_tool_arg_validation   → ctx.current_tool_args
      REWRITE        @ post_tool_use / post_tool_result_raw /
                       on_tool_error                            → ctx.current_tool_result_str
      REWRITE        @ pre_final_emit                           → ctx.final_output
      BLOCK          @ pre_tool_use / pre_tool_arg_validation   → skip_current_tool flag
                                                                   (does NOT terminate the loop)
      BLOCK          @ anything else                            → ctx.blocked + stop=True
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
        # Per-tool-call BLOCK: skip the tool, don't halt the whole task.
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
    """Adapter: core.Dispatcher passes `event_name: str`, sibling policy
    accepts Mount enum OR string (see policy._normalise_key)."""
    validate_decision(comp.cls, event_name, kind)


class Dispatcher:
    """Single-workflow dispatcher used by SopBenchAgent.

    Phase B/C migration: internally wraps `core.Dispatcher` (the
    event-keyed dispatcher) while keeping the existing `dispatch(mount,
    ctx)` public surface so SopBenchAgent doesn't change shape. Adds
    `emit(event_name, ctx)` (Phase D) and `wire_capabilities(ctx)`
    (Phase C, hooks ctx.chat / ctx.emit) for the new event-style
    integrations.
    """

    def __init__(
        self,
        *,
        workflow: Workflow,
        components: list[Component],
        run_tag: str = "default",
        state_dir: Optional[Path] = None,
    ) -> None:
        self.workflow = workflow
        self.run_tag = run_tag
        self._state_dir = state_dir
        self._core = _CoreDispatcher(
            list(components),
            validate_decision=_validate_for_core,
            apply_decision=_apply_decision_sopbench,
            trace_sink=self._trace_sink,
        )

    # --- trace sink -------------------------------------------------------

    def _trace_dir(self) -> Path:
        if self._state_dir is not None:
            d = self._state_dir / self.run_tag
        else:
            base = Path(os.environ.get(_STATE_ENV, ".component-state-sopbench"))
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
            # Trace failure must never break the agent loop.
            pass

    # --- dispatch ---------------------------------------------------------

    def emit(self, event_name: str, ctx: ComponentContext) -> None:
        """Fire `event_name` through the unified core dispatcher.
        Per-event side-effects (skip_current_tool reset before
        pre_tool_use, etc.) are handled here or in
        `_apply_decision_sopbench` based on `ctx.event`."""
        if event_name == "pre_tool_use":
            ctx.shared.pop("skip_current_tool", None)
        self._core.emit(event_name, ctx)

    def wire_capabilities(self, ctx: ComponentContext) -> None:
        """Phase C: attach per-task capability implementations.

        - ctx.chat → agent.llm.chat (locked SUT model)
        - ctx.emit → re-enter this same dispatcher (with depth cap)
        - ctx.fetch / ctx.read_file intentionally None in v1 (not yet
          wired by this sibling).
        """
        ctx._impl_chat = _make_chat_impl()
        ctx._impl_emit = lambda name, fields: self._core.emit(name, ctx)


def _coerce_active_names(workflow: Workflow) -> set[str]:
    """If SOPBENCH_COMPONENT_NAMES is set, sanity-check it matches workflow.

    Mirrors GAIA's safety check that the outer loop pinned the YAML
    before launching the eval.
    """
    raw = os.environ.get(_NAMES_ENV, "").strip()
    if not raw:
        return set(workflow.active_nodes())
    declared = {n for n in raw.split(",") if n}
    yaml_active = set(workflow.active_nodes())
    if declared != yaml_active:
        raise SystemExit(
            f"{_NAMES_ENV} {sorted(declared)} != workflow active "
            f"{sorted(yaml_active)}; the outer loop must pin the workflow yaml."
        )
    return declared


def build_dispatcher(
    *,
    workflow_path: Optional[Path] = None,
    comp_dir: Optional[Path] = None,
    run_tag: Optional[str] = None,
    state_dir: Optional[Path] = None,
) -> Dispatcher:
    """Return a cached Dispatcher for the (workflow, components_dir) pair.

    Reading components from disk and validating each at load time is mildly
    expensive; the cache makes parallel-worker eval re-use the same parsed
    Component objects.
    """
    wf_path = workflow_path or resolve_workflow_path()
    cd = Path(comp_dir or os.environ.get(_DIR_ENV) or COMPONENTS_DIR_DEFAULT)
    tag = run_tag or os.environ.get(_RUN_TAG_ENV, "default")

    key = (str(wf_path) if wf_path else "", str(cd))
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            return hit

        if wf_path and Path(wf_path).exists():
            wf = Workflow.from_yaml(wf_path)
        else:
            wf = Workflow()
        _coerce_active_names(wf)  # raises SystemExit on mismatch

        active = list(wf.active_nodes())
        components = load_components_from_dir(cd, only=active) if active else []

        disp = Dispatcher(
            workflow=wf,
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
