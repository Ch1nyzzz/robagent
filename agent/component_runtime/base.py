"""GAIA component runtime entry point: `run_task` wrapped with dispatch chain.

The `run_benchmark.py::_resolve_run_task` resolver imports
`agent.component_runtime.base::run_task` when invoked with
`--agent-version component_runtime`. The function signature is identical
to `agent/base.py::run_task` so the orchestrator does not change.

Composition: at each lifecycle event, every active component's matcher
is queried; matching handlers run in priority order; their Decisions
mutate the in-flight state (system_prompt / prompt / raw_response /
answer) per the policy matrix. Component fires append one row per fire
to `.component-state/<run_tag>/fired.jsonl` for the durability audit.

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
from meta_harness.component_runtime_core.dispatcher import Dispatcher

from .policy import ComponentPolicyError, validate_decision
from .registry import COMPONENTS_DIR_DEFAULT, load_components_from_dir
from .types import (
    Component,
    ComponentContext,
    DecisionKind,
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


def _trace_sink(component_name: str, event_name: str, decision_kind_value: str,
                extra: dict) -> None:
    """Append one row per fire to `fired.jsonl`. Phase B/C: the trace shape
    carries both `event` (Tier-1 / Tier-2/3 name) and `mount` (legacy alias
    of the same string) so downstream log readers that still parse the old
    key keep working until they migrate."""
    rec = {
        "ts": time.time(),
        "component": component_name,
        "event": event_name,
        "mount": event_name,
        "decision": decision_kind_value,
        **extra,
    }
    with (_trace_dir() / "fired.jsonl").open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --- workflow load (per process; cached) -------------------------------------


_WORKFLOW: Optional[Workflow] = None
_DISPATCHER: Optional[Dispatcher] = None


def _resolve_workflow_path() -> Path:
    return Path(os.environ.get("COMPONENT_WORKFLOW", str(DEFAULT_WORKFLOW)))


def _load_active() -> tuple[Workflow, Dispatcher]:
    """Load workflow + build the per-process Dispatcher. Cached."""
    global _WORKFLOW, _DISPATCHER
    if _WORKFLOW is not None and _DISPATCHER is not None:
        return _WORKFLOW, _DISPATCHER

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
    flattened: list[Component] = (
        load_components_from_dir(comp_dir, only=active) if active else []
    )

    _WORKFLOW = wf
    _DISPATCHER = Dispatcher(
        flattened,
        validate_decision=_validate_for_dispatcher,
        apply_decision=_apply_decision,
        trace_sink=_trace_sink,
    )
    return wf, _DISPATCHER


# --- dispatch helpers (Phase B/C) --------------------------------------------


def _validate_for_dispatcher(comp: Component, event_name: str,
                             kind: DecisionKind) -> None:
    """Adapter: the core Dispatcher passes `event_name: str`; the policy
    validator takes the same shape."""
    validate_decision(comp.cls, event_name, kind)


def _apply_decision(ctx: ComponentContext, decision: "Decision",  # noqa: F821
                    comp: Component) -> bool:
    """Mutate ctx per Decision; return True to stop firing remaining
    subscribers at this event.

    Event → side-effect mapping:

      INJECT_CONTEXT @ pre-LLM events  → append to ctx.system_prompt
                       (session_start, pre_prompt_build, pre_context_build,
                        task_received, pre_agent_construct, pre_llm_request)
      INJECT_CONTEXT @ post-LLM events → push onto ctx.shared['post_llm_inject']
                       (post_llm_response[_raw], on_length_truncation, on_empty_response)
      REWRITE @ pre_prompt_build / pre_context_build → ctx.prompt
      REWRITE @ post_llm_response[_raw] /
                 on_length_truncation / on_empty_response → ctx.raw_response
      REWRITE @ pre_answer_emit                       → ctx.answer (None = blocked)
      BLOCK   @ any                                   → ctx.blocked + stop=True
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
        ):
            ctx.system_prompt = (ctx.system_prompt + "\n\n" + text).strip()
        else:
            ctx.shared.setdefault("post_llm_inject", []).append(text)
        return False
    if kind is DecisionKind.REWRITE:
        payload = decision.payload
        if event in ("pre_prompt_build", "pre_context_build"):
            ctx.prompt = str(payload or "")
        elif event in (
            "post_llm_response",
            "post_llm_response_raw",
            "on_length_truncation",
            "on_empty_response",
        ):
            ctx.raw_response = str(payload or "")
        elif event == "pre_answer_emit":
            ctx.answer = None if payload is None else str(payload)
        return False
    if kind is DecisionKind.BLOCK:
        ctx.blocked = True
        ctx.blocked_reason = decision.reason or f"{comp.name}: block"
        return True
    return False


def _make_chat_impl():
    """Build the per-task ctx.chat implementation. The locked SUT model is
    enforced inside `agent.llm.chat` itself (raises if a `model=` override
    is passed). The wrapper translates EventContext.chat's `system_override`
    kwarg into a messages-list rewrite the upstream API accepts."""
    def _impl(messages, *, max_tokens, temperature, system_override, tools):
        if system_override:
            messages = (
                [{"role": "system", "content": system_override}]
                + [m for m in messages if m.get("role") != "system"]
            )
        return chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
        )
    return _impl


def _wire_capabilities(ctx: ComponentContext, dispatcher: Dispatcher) -> None:
    """Attach per-sibling capability implementations to the ctx. None for
    fetch / read_file (gaia v1 does not expose those — components that
    declare HTTP_GET / READ_FILE still get the structural contract via
    the manifest, but the runtime stub will raise if called)."""
    ctx._impl_chat = _make_chat_impl()
    ctx._impl_emit = lambda name, fields: dispatcher.emit(name, ctx)
    # ctx._impl_fetch / _impl_read_file intentionally left None for gaia v1.




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

    wf, dispatcher = _load_active()

    ctx = ComponentContext(
        benchmark=benchmark,
        task_id=task_id,
        extras=extras,
        system_prompt=SYSTEM_PROMPT,
        prompt=task_prompt,
        log=log,
    )
    _wire_capabilities(ctx, dispatcher)

    def _blocked_return(stage: str) -> dict[str, Any]:
        log.emit("agent.blocked", parent=root,
                 reason=ctx.blocked_reason, stage=stage)
        log.emit("answer.emitted", parent=root, answer=None)
        log.close()
        return {"run_id": run_id, "answer": None, "error": None,
                "trace_path": str(log.path), "log": log, "root_event_id": root,
                "blocked_reason": ctx.blocked_reason}

    # --- setup phase ---------------------------------------------------------
    # Tier-1 task_received: lifecycle anchor right after ctx construction.
    dispatcher.emit("task_received", ctx)
    if ctx.blocked:
        return _blocked_return("task_received")

    # Legacy SESSION_START mount (static framework-invariant injection).
    dispatcher.emit("session_start", ctx)
    if ctx.blocked:
        return _blocked_return("session_start")

    # Legacy PRE_PROMPT_BUILD mount, paired with Tier-1 alias pre_context_build
    # (per the cross-sibling event vocabulary). Legacy subscribers fire on the
    # mount.value, new event-style subscribers fire on the Tier-1 name.
    dispatcher.emit("pre_prompt_build", ctx)
    if ctx.blocked:
        return _blocked_return("pre_prompt_build")
    dispatcher.emit("pre_context_build", ctx)
    if ctx.blocked:
        return _blocked_return("pre_context_build")

    # Tier-1 pre_agent_construct: last chance to influence the inference
    # request before messages list is sealed. Components can append guidance
    # via INJECT_CONTEXT.
    dispatcher.emit("pre_agent_construct", ctx)
    if ctx.blocked:
        return _blocked_return("pre_agent_construct")

    messages = [
        {"role": "system", "content": ctx.system_prompt},
        {"role": "user", "content": ctx.prompt or ""},
    ]

    # Tier-1 pre_llm_request: anything wired to react just before the SUT call.
    dispatcher.emit("pre_llm_request", ctx)
    if ctx.blocked:
        return _blocked_return("pre_llm_request")

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

    # Legacy POST_LLM_RESPONSE + Tier-1 post_llm_response_raw twin emit.
    dispatcher.emit("post_llm_response", ctx)
    if ctx.blocked:
        return _blocked_return("post_llm_response")
    dispatcher.emit("post_llm_response_raw", ctx)
    if ctx.blocked:
        return _blocked_return("post_llm_response_raw")

    # Tier-1 failure-mode events — synthesised from the response shape so
    # components can attach narrowly (e.g. length_recovery_guard hooks
    # `on_length_truncation` instead of post_llm_response with a finish_reason
    # matcher). Both gates are independent: an empty response with
    # finish_reason=length fires both.
    if (result.get("finish_reason") or "") == "length":
        dispatcher.emit("on_length_truncation", ctx)
        if ctx.blocked:
            return _blocked_return("on_length_truncation")
    if not (ctx.raw_response or "").strip():
        dispatcher.emit("on_empty_response", ctx)
        if ctx.blocked:
            return _blocked_return("on_empty_response")

    # Default extraction: strip whitespace. Components at PRE_ANSWER_EMIT may
    # override (e.g., regex on "FINAL ANSWER:" line).
    ctx.answer = (ctx.raw_response or "").strip()

    # Legacy PRE_ANSWER_EMIT.
    dispatcher.emit("pre_answer_emit", ctx)
    if ctx.blocked:
        return _blocked_return("pre_answer_emit")

    log.emit("answer.emitted", parent=root, answer=ctx.answer)

    # Tier-1 session_end: terminal bookkeeping; decisions are ALLOW-only per
    # policy (any other decision is too late to matter on a single-shot
    # benchmark). Fired after answer emission so handlers see the final ctx.
    dispatcher.emit("session_end", ctx)

    return {
        "run_id": run_id,
        "answer": ctx.answer,
        "error": None,
        "trace_path": str(log.path),
        "log": log,
        "root_event_id": root,
    }
