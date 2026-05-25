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


class Dispatcher:
    """Single-workflow dispatcher used by EnterpriseOpsAgent.

    Holds a parsed Workflow + components-by-mount mapping. Stateless per
    task: every task constructs its own ComponentContext and threads it
    through the dispatch calls.

    Construction is intentionally cheap-to-reuse; prefer `build_dispatcher`
    which caches by (workflow_path, comp_dir) so multi-threaded eval loops
    don't reload component modules for every task.
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

    # --- trace sink -------------------------------------------------------

    def _trace_dir(self) -> Path:
        if self._state_dir is not None:
            d = self._state_dir / self.run_tag
        else:
            base = Path(os.environ.get(_STATE_ENV, ".component-state-enterpriseops"))
            d = base / self.run_tag
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _trace(self, component_name: str, mount: Mount,
               decision_kind: DecisionKind, extra: dict) -> None:
        rec = {
            "ts": time.time(),
            "component": component_name,
            "mount": mount.value,
            "decision": decision_kind.value,
            **extra,
        }
        try:
            with (self._trace_dir() / "fired.jsonl").open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except Exception:
            # Trace failure must never break the agent loop.
            pass

    # --- dispatch ---------------------------------------------------------

    def dispatch(self, mount: Mount, ctx: ComponentContext) -> None:
        """Fire all active components at `mount` in priority order.

        Decisions mutate `ctx` in place:
          INJECT_CONTEXT @ SESSION_START / PRE_PROMPT_BUILD
              → append to ctx.system_prompt
          INJECT_CONTEXT @ POST_LLM_RESPONSE
              → append to ctx.shared['post_llm_inject'] (read by agent at
                start of next turn and added as system reminder)
          REWRITE @ PRE_PROMPT_BUILD   → ctx.user_prompt = payload
          REWRITE @ PRE_LLM_TURN       → ctx.messages = list(payload)
          REWRITE @ POST_LLM_RESPONSE  → ctx.raw_response = payload
                                         (tool_calls preserved; use
                                         PRE_TOOL_USE BLOCK to drop them)
          REWRITE @ PRE_TOOL_USE       → ctx.current_tool_args = dict(payload)
          REWRITE @ POST_TOOL_USE      → ctx.current_tool_result_str = payload
          REWRITE @ PRE_FINAL_EMIT     → ctx.final_output = payload (None → blocked)
          BLOCK @ PRE_TOOL_USE         → ctx.shared['skip_current_tool'] = True
                                         (agent skips dispatch; ctx.blocked stays False)
          BLOCK @ anything else        → ctx.blocked = True
        """
        ctx.mount = mount
        # Reset per-mount skip-tool flag so a previous PRE_TOOL_USE call
        # doesn't carry over to a later mount.
        if mount is Mount.PRE_TOOL_USE:
            ctx.shared.pop("skip_current_tool", None)

        for comp in self.components_by_mount.get(mount, []):
            if ctx.blocked:
                return
            if comp.matcher is not None:
                try:
                    if not comp.matcher(ctx):
                        continue
                except Exception as e:
                    self._trace(comp.name, mount, DecisionKind.ALLOW,
                                {"matcher_error": repr(e)})
                    continue
            try:
                decision = comp.handler(ctx)
            except Exception as e:
                self._trace(comp.name, mount, DecisionKind.ALLOW,
                            {"handler_error": repr(e)})
                continue
            try:
                validate_decision(comp.cls, comp.mount, decision.kind)
            except ComponentPolicyError as e:
                self._trace(comp.name, mount, decision.kind,
                            {"policy_error": str(e)})
                continue

            kind = decision.kind
            if kind is DecisionKind.ALLOW:
                self._trace(comp.name, mount, kind, {})
                continue
            if kind is DecisionKind.INJECT_CONTEXT:
                text = str(decision.payload or "")
                if mount in (Mount.SESSION_START, Mount.PRE_PROMPT_BUILD):
                    ctx.system_prompt = (ctx.system_prompt + "\n\n" + text).strip()
                else:
                    ctx.shared.setdefault("post_llm_inject", []).append(text)
                self._trace(comp.name, mount, kind, {"chars": len(text)})
            elif kind is DecisionKind.REWRITE:
                payload = decision.payload
                if mount is Mount.PRE_PROMPT_BUILD:
                    ctx.user_prompt = str(payload or "")
                elif mount is Mount.PRE_LLM_TURN:
                    if isinstance(payload, list):
                        ctx.messages = list(payload)
                elif mount is Mount.POST_LLM_RESPONSE:
                    ctx.raw_response = str(payload or "")
                elif mount is Mount.PRE_TOOL_USE:
                    if isinstance(payload, dict):
                        ctx.current_tool_args = dict(payload)
                elif mount is Mount.POST_TOOL_USE:
                    ctx.current_tool_result_str = str(payload or "")
                elif mount is Mount.PRE_FINAL_EMIT:
                    ctx.final_output = "" if payload is None else str(payload)
                self._trace(comp.name, mount, kind, {})
            elif kind is DecisionKind.BLOCK:
                if mount is Mount.PRE_TOOL_USE:
                    ctx.shared["skip_current_tool"] = True
                    ctx.shared["skip_current_tool_reason"] = (
                        decision.reason or f"{comp.name}: block"
                    )
                    self._trace(comp.name, mount, kind,
                                {"reason": ctx.shared["skip_current_tool_reason"]})
                else:
                    ctx.blocked = True
                    ctx.blocked_reason = decision.reason or f"{comp.name}: block"
                    self._trace(comp.name, mount, kind,
                                {"reason": ctx.blocked_reason})


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
