"""EnterpriseOps dispatcher: runtime side of the EnterpriseOpsAgent component bridge.

Unlike GAIA's `agent/component_runtime/base.py` which provides a complete
`run_task(...)` entry point, this runtime exposes a `Dispatcher` that the
EnterpriseOpsAgent calls at each Mount in its function-calling loop. The
agent owns the loop (wrapping upstream `orchestrators/react.py`); the
dispatcher owns the policy + mutation logic.

Workflow source on disk:
  meta_harness/workflows/enterpriseops_<domain>.yaml  (override with
  `ENTERPRISEOPS_COMPONENT_WORKFLOW` env var)

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
from meta_harness.component_runtime_core.dispatcher import Dispatcher as _CoreDispatcher

from .policy import ComponentPolicyError, validate_decision
from .registry import COMPONENTS_DIR_DEFAULT, load_components_from_dir
from .types import (
    Component,
    ComponentContext,
    DecisionKind,
    Mount,
)
from .workflow import Workflow


ROOT = Path(__file__).resolve().parent.parent.parent

_WORKFLOW_ENV = "ENTERPRISEOPS_COMPONENT_WORKFLOW"
_NAMES_ENV    = "ENTERPRISEOPS_COMPONENT_NAMES"
_RUN_TAG_ENV  = "ENTERPRISEOPS_COMPONENT_RUN_TAG"
_DIR_ENV      = "ENTERPRISEOPS_COMPONENT_DIR"
_STATE_ENV    = "ENTERPRISEOPS_COMPONENT_STATE_DIR"


def resolve_workflow_path(default: Optional[Path] = None) -> Optional[Path]:
    raw = os.environ.get(_WORKFLOW_ENV)
    if raw:
        return Path(raw)
    return default


# Process-level cache: one Dispatcher per workflow+components selection.
_CACHE_LOCK = Lock()
_CACHE: dict[tuple[str, str], "Dispatcher"] = {}


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
    """Adapter: core dispatcher passes event_name string, policy accepts
    Mount enum OR string (see policy._normalise_key)."""
    validate_decision(comp.cls, event_name, kind)


class Dispatcher:
    """Single-workflow dispatcher used by EnterpriseOpsAgent.

    Phase B/C migration: wraps `core.Dispatcher` internally while keeping
    the existing `dispatch(mount, ctx)` surface. Adds `emit(event_name,
    ctx)` (Phase D) and `wire_capabilities(ctx)` (Phase C) for the new
    event-style integrations.
    """

    def __init__(
        self,
        *,
        workflow: Workflow,
        components_by_mount: dict[Mount, list[Component]],
        run_tag: str = "default",
        state_dir: Optional[Path] = None,
    ) -> None:
        self.workflow = workflow
        self.components_by_mount = components_by_mount
        self.run_tag = run_tag
        self._state_dir = state_dir
        flat: list[Component] = []
        for mount, comps in components_by_mount.items():
            flat.extend(comps)
        self._core = _CoreDispatcher(
            flat,
            validate_decision=_validate_for_core,
            apply_decision=_apply_decision_enterpriseops,
            trace_sink=self._trace_sink,
        )

    # --- trace sink -------------------------------------------------------

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

    # --- dispatch ---------------------------------------------------------

    def emit(self, event_name: str, ctx: ComponentContext,
             *, sync_mount: Optional[Mount] = None) -> None:
        """Unified dispatch entry. Both legacy mount-aligned events
        (caller passes `event_name=Mount.X.value`, `sync_mount=Mount.X`)
        and Tier-1 events (caller passes just the event-name string)
        route here. `sync_mount` updates `ctx.mount` so legacy matchers
        reading `ctx.mount` see the right value; the PRE_TOOL_USE
        skip-flag reset stays attached to its mount."""
        if sync_mount is not None:
            ctx.mount = sync_mount
            if sync_mount is Mount.PRE_TOOL_USE:
                ctx.shared.pop("skip_current_tool", None)
        self._core.emit(event_name, ctx)

    def wire_capabilities(self, ctx: ComponentContext) -> None:
        """Phase C: attach per-task capability implementations.
        ctx.chat → agent.llm.chat (locked SUT model);
        ctx.emit → re-enter this dispatcher (with depth cap).
        ctx.fetch / ctx.read_file intentionally None in v1."""
        ctx._impl_chat = _make_chat_impl()
        ctx._impl_emit = lambda name, fields: self._core.emit(name, ctx)


def _coerce_active_names(workflow: Workflow) -> set[str]:
    """If ENTERPRISEOPS_COMPONENT_NAMES is set, sanity-check against workflow.

    Mirrors the safety check that the outer loop pinned the YAML before
    launching the eval.
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
        if active:
            by_mount = load_components_from_dir(cd, only=active)
        else:
            by_mount = {m: [] for m in Mount}

        disp = Dispatcher(
            workflow=wf,
            components_by_mount=by_mount,
            run_tag=tag,
            state_dir=state_dir,
        )
        _CACHE[key] = disp
        return disp


def clear_dispatcher_cache() -> None:
    """Test helper: drop the process-level dispatcher cache."""
    with _CACHE_LOCK:
        _CACHE.clear()
