"""GAIA component runtime entry point: `run_task` wrapped with dispatch chain.

The `run_benchmark.py::_resolve_run_task` resolver imports
`agent.component_runtime.base::run_task` when invoked with
`--agent-version component_runtime`. The function signature is identical
to `agent/base.py::run_task` so the orchestrator does not change.

Composition: at each Mount, every active component's matcher is queried;
matching handlers run in priority order; their Decisions mutate the
in-flight state (system_prompt / prompt / raw_response / answer) per the
policy matrix. Component fires append one row per fire to
`.component-state/<run_tag>/fired.jsonl` for the durability audit.

Workflow source on disk:
  meta_harness/workflows/gaia_main.yaml  (default; override with COMPONENT_WORKFLOW env var)

Active component set is the workflow's `active_nodes()` (i.e. excluding
`disabled:`). The outer loop is responsible for keeping the YAML in sync
with the accepted frontier; this runtime trusts it.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from agent.events import EventLog, new_run_id, traces_dir
from agent.llm import chat, DEFAULT_MODEL

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
DEFAULT_WORKFLOW = ROOT / "meta_harness" / "workflows" / "gaia_main.yaml"

SYSTEM_PROMPT = (
    "You are an assistant solving a single benchmark task. "
    "Read the task carefully and produce the final answer only. "
    "Do not include explanations, prefixes, or extra text. "
    "If the expected answer is a number, output the number only. "
    "If the expected answer is a short string, output that string only."
)


# --- trace sink --------------------------------------------------------------


def _trace_dir() -> Path:
    tag = os.environ.get("COMPONENT_RUN_TAG", "default")
    d = Path(os.environ.get("COMPONENT_STATE_DIR", ".component-state")) / tag
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


# --- workflow load (per process; cached) -------------------------------------


_WORKFLOW: Optional[Workflow] = None
_COMPONENTS_BY_MOUNT: Optional[dict[Mount, list[Component]]] = None


def _resolve_workflow_path() -> Path:
    return Path(os.environ.get("COMPONENT_WORKFLOW", str(DEFAULT_WORKFLOW)))


def _load_active() -> tuple[Workflow, dict[Mount, list[Component]]]:
    """Load workflow + active components once per process. Cached."""
    global _WORKFLOW, _COMPONENTS_BY_MOUNT
    if _WORKFLOW is not None and _COMPONENTS_BY_MOUNT is not None:
        return _WORKFLOW, _COMPONENTS_BY_MOUNT

    wf_path = _resolve_workflow_path()
    if wf_path.exists():
        wf = Workflow.from_yaml(wf_path)
    else:
        wf = Workflow()

    names_env = os.environ.get("COMPONENT_NAMES", "").strip()
    names_from_env = {n for n in names_env.split(",") if n}
    if names_from_env and names_from_env != set(wf.active_nodes()):
        raise SystemExit(
            f"COMPONENT_NAMES {sorted(names_from_env)} ≠ workflow active "
            f"{sorted(wf.active_nodes())}; outer loop must pin the workflow."
        )

    comp_dir = os.environ.get("COMPONENT_DIR", str(COMPONENTS_DIR_DEFAULT))
    active = list(wf.active_nodes())
    if active:
        grouped = load_components_from_dir(comp_dir, only=active)
    else:
        grouped = {m: [] for m in Mount}

    _WORKFLOW = wf
    _COMPONENTS_BY_MOUNT = grouped
    return wf, grouped


# --- dispatch ----------------------------------------------------------------


def _dispatch(mount: Mount, ctx: ComponentContext,
              components_by_mount: dict[Mount, list[Component]]) -> None:
    """Fire all components at `mount` whose matcher matches, in priority order.

    Decisions mutate `ctx` in place:
      INJECT_CONTEXT @ SESSION_START / PRE_PROMPT_BUILD → append to ctx.system_prompt
      INJECT_CONTEXT @ POST_LLM_RESPONSE → append to ctx.shared['post_llm_inject']
                                            (the recovery hook reads this; v1: append
                                            to raw_response as a system note marker)
      REWRITE @ PRE_PROMPT_BUILD  → ctx.prompt = payload
      REWRITE @ POST_LLM_RESPONSE → ctx.raw_response = payload
      REWRITE @ PRE_ANSWER_EMIT   → ctx.answer = payload (None marks blocked)
      BLOCK  @ any                 → ctx.blocked = True, ctx.blocked_reason = decision.reason
    """
    for comp in components_by_mount.get(mount, []):
        if ctx.blocked:
            return
        if comp.matcher is not None and not comp.matcher(ctx):
            continue
        decision = comp.handler(ctx)
        try:
            validate_decision(comp.cls, comp.mount, decision.kind)
        except ComponentPolicyError as e:
            # log and treat as ALLOW (don't crash the eval)
            if ctx.log is not None:
                ctx.log.emit("component.policy_error",
                             component=comp.name, error=repr(e))
            _trace(comp.name, mount, decision.kind, {"policy_error": str(e)})
            continue

        kind = decision.kind
        if kind is DecisionKind.ALLOW:
            _trace(comp.name, mount, kind, {})
            continue
        if kind is DecisionKind.INJECT_CONTEXT:
            text = str(decision.payload or "")
            if mount in (Mount.SESSION_START, Mount.PRE_PROMPT_BUILD):
                ctx.system_prompt = (ctx.system_prompt + "\n\n" + text).strip()
            else:
                ctx.shared.setdefault("post_llm_inject", []).append(text)
            _trace(comp.name, mount, kind, {"chars": len(text)})
        elif kind is DecisionKind.REWRITE:
            payload = decision.payload
            if mount is Mount.PRE_PROMPT_BUILD:
                ctx.prompt = str(payload or "")
            elif mount is Mount.POST_LLM_RESPONSE:
                ctx.raw_response = str(payload or "")
            elif mount is Mount.PRE_ANSWER_EMIT:
                ctx.answer = None if payload is None else str(payload)
            _trace(comp.name, mount, kind, {})
        elif kind is DecisionKind.BLOCK:
            ctx.blocked = True
            ctx.blocked_reason = decision.reason or f"{comp.name}: block"
            _trace(comp.name, mount, kind, {"reason": ctx.blocked_reason})


# --- entry point -------------------------------------------------------------


def run_task(
    *,
    benchmark: str,
    task_id: str,
    task_prompt: str,
    extras: Optional[dict] = None,
) -> dict[str, Any]:
    """Component-runtime GAIA agent. Same signature as agent/base.py::run_task."""
    extras = extras or {}
    run_id = new_run_id()
    log = EventLog(run_id=run_id, benchmark=benchmark, task_id=task_id,
                   out_dir=traces_dir())

    root = log.emit(
        "run.started",
        question=task_prompt,
        model=DEFAULT_MODEL,
        extras=extras,
    )

    wf, by_mount = _load_active()

    ctx = ComponentContext(
        mount=Mount.SESSION_START,
        benchmark=benchmark,
        task_id=task_id,
        extras=extras,
        system_prompt=SYSTEM_PROMPT,
        prompt=task_prompt,
        log=log,
    )

    # SESSION_START
    _dispatch(Mount.SESSION_START, ctx, by_mount)
    if ctx.blocked:
        log.emit("agent.blocked", parent=root,
                 reason=ctx.blocked_reason, stage="session_start")
        log.emit("answer.emitted", parent=root, answer=None)
        log.close()
        return {"run_id": run_id, "answer": None, "error": None,
                "trace_path": str(log.path), "log": log, "root_event_id": root,
                "blocked_reason": ctx.blocked_reason}

    # PRE_PROMPT_BUILD
    ctx.mount = Mount.PRE_PROMPT_BUILD
    _dispatch(Mount.PRE_PROMPT_BUILD, ctx, by_mount)
    if ctx.blocked:
        log.emit("agent.blocked", parent=root,
                 reason=ctx.blocked_reason, stage="pre_prompt_build")
        log.emit("answer.emitted", parent=root, answer=None)
        log.close()
        return {"run_id": run_id, "answer": None, "error": None,
                "trace_path": str(log.path), "log": log, "root_event_id": root,
                "blocked_reason": ctx.blocked_reason}

    messages = [
        {"role": "system", "content": ctx.system_prompt},
        {"role": "user", "content": ctx.prompt or ""},
    ]
    call = log.emit("llm.requested", parent=root, messages=messages,
                    model=DEFAULT_MODEL)

    try:
        result = chat(messages=messages)
    except Exception as e:
        log.emit("llm.failed", parent=call, error=repr(e))
        log.emit("run.failed", parent=root, error=repr(e))
        log.close()
        return {"run_id": run_id, "answer": None, "error": repr(e),
                "trace_path": str(log.path)}

    log.emit(
        "llm.responded",
        parent=call,
        content=result["content"],
        finish_reason=result["finish_reason"],
        usage=result["usage"],
    )
    ctx.raw_response = result.get("content") or ""
    ctx.shared["finish_reason"] = result.get("finish_reason")

    # POST_LLM_RESPONSE
    ctx.mount = Mount.POST_LLM_RESPONSE
    _dispatch(Mount.POST_LLM_RESPONSE, ctx, by_mount)
    if ctx.blocked:
        log.emit("agent.blocked", parent=root,
                 reason=ctx.blocked_reason, stage="post_llm_response")
        log.emit("answer.emitted", parent=root, answer=None)
        log.close()
        return {"run_id": run_id, "answer": None, "error": None,
                "trace_path": str(log.path), "log": log, "root_event_id": root,
                "blocked_reason": ctx.blocked_reason}

    # Default extraction: strip whitespace. Components at PRE_ANSWER_EMIT may
    # override (e.g., regex on "FINAL ANSWER:" line).
    ctx.answer = (ctx.raw_response or "").strip()

    # PRE_ANSWER_EMIT
    ctx.mount = Mount.PRE_ANSWER_EMIT
    _dispatch(Mount.PRE_ANSWER_EMIT, ctx, by_mount)
    if ctx.blocked:
        log.emit("agent.blocked", parent=root,
                 reason=ctx.blocked_reason, stage="pre_answer_emit")
        log.emit("answer.emitted", parent=root, answer=None)
        log.close()
        return {"run_id": run_id, "answer": None, "error": None,
                "trace_path": str(log.path), "log": log, "root_event_id": root,
                "blocked_reason": ctx.blocked_reason}

    log.emit("answer.emitted", parent=root, answer=ctx.answer)

    return {
        "run_id": run_id,
        "answer": ctx.answer,
        "error": None,
        "trace_path": str(log.path),
        "log": log,
        "root_event_id": root,
    }
