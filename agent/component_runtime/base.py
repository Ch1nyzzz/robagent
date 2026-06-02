"""GAIA component runtime entry point: `run_task` wrapped with dispatch chain.

Same FC loop and tools as `agent/base.py` — the baseline IS the protagonist;
this module just sprinkles dispatcher.emit calls at each lifecycle anchor so
registered components can stabilize the loop's behavior. Setup events fire
once per task; pre_llm_request / post_llm_response / on_length_truncation /
on_empty_response fire once PER FC TURN; pre_tool_use / post_tool_use /
on_tool_error fire once PER TOOL CALL within a turn.

`run_benchmark.py::_resolve_run_task` imports this module when invoked with
`--agent-version component_runtime`. Function signature matches
`agent/base.py::run_task` so the orchestrator does not change.

Decisions mutate ctx in-place per the policy matrix. Component fires append
one row per fire to `.component-state/<run_tag>/fired.jsonl` for the
durability audit.

Workflow source on disk:
  ballast/workflows/gaia_main.yaml  (default; override via COMPONENT_WORKFLOW)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from agent.base import (
    MAX_ITERATIONS,
    PER_TURN_MAX_TOKENS,
    SYSTEM_PROMPT,
    _extract_final_answer,
)
from agent.events import EventLog, new_run_id, traces_dir
from agent.llm import chat, DEFAULT_MODEL
from agent.tools import TOOL_SPECS, dispatch_tool
from ballast.component_runtime_core.dispatcher import Dispatcher

from .policy import ComponentPolicyError, validate_decision
from .registry import COMPONENTS_DIR_DEFAULT, load_components_from_dir
from .types import (
    Component,
    ComponentContext,
    DecisionKind,
)
from .workflow import Workflow


ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_WORKFLOW = ROOT / "ballast" / "workflows" / "gaia_main.yaml"

# SYSTEM_PROMPT, MAX_ITERATIONS, PER_TURN_MAX_TOKENS, _extract_final_answer
# imported from agent.base — single source of truth so the component runtime
# scores against the SAME baseline the proposer's hooks aim to stabilize.


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


_SETUP_EVENTS = frozenset({
    "session_start", "pre_prompt_build", "pre_context_build",
    "task_received", "pre_agent_construct", "pre_llm_request",
})

_RAW_RESPONSE_EVENTS = frozenset({
    "post_llm_response", "post_llm_response_raw",
    "on_length_truncation", "on_empty_response",
})


def _apply_decision(ctx: ComponentContext, decision: "Decision",  # noqa: F821
                    comp: Component) -> bool:
    """Mutate ctx per Decision; return True to stop firing remaining
    subscribers at this event.

    Event → side-effect mapping:

      INJECT_CONTEXT @ setup events    → append to ctx.system_prompt
      INJECT_CONTEXT @ post-LLM events → push onto ctx.shared['post_llm_inject']
                                          (replayed as a system note next turn)
      INJECT_CONTEXT @ post_tool_use /
                       on_tool_error    → concatenated INTO ctx.current_tool_result
                                          so the LLM sees it on the next turn
      INJECT_CONTEXT @ pre_tool_use    → queued on ctx.shared['post_llm_inject']
                                          (advisory only; tool still runs)

      REWRITE @ pre_prompt_build /
                pre_context_build       → ctx.prompt (str)
      REWRITE @ raw-response events     → ctx.raw_response (str)
      REWRITE @ pre_tool_use            → ctx.current_tool_args (dict)
      REWRITE @ post_tool_use /
                on_tool_error           → ctx.current_tool_result (str)
      REWRITE @ pre_answer_emit         → ctx.answer (None = blocked)

      BLOCK @ pre_tool_use              → skip THIS tool call (task continues);
                                          tool result = "[skipped by <comp>]"
      BLOCK @ any other event           → terminate task (ctx.blocked + stop)
    """
    event = ctx.event or ""
    kind = decision.kind
    if kind is DecisionKind.ALLOW:
        return False
    if kind is DecisionKind.INJECT_CONTEXT:
        text = str(decision.payload or "")
        if event in _SETUP_EVENTS:
            ctx.system_prompt = (ctx.system_prompt + "\n\n" + text).strip()
        elif event in ("post_tool_use", "on_tool_error"):
            base = ctx.current_tool_result or ""
            ctx.current_tool_result = (base + "\n\n" + text).strip() if base else text
        else:
            ctx.shared.setdefault("post_llm_inject", []).append(text)
        return False
    if kind is DecisionKind.REWRITE:
        payload = decision.payload
        if event in ("pre_prompt_build", "pre_context_build"):
            ctx.prompt = str(payload or "")
        elif event in _RAW_RESPONSE_EVENTS:
            ctx.raw_response = str(payload or "")
        elif event == "pre_tool_use":
            if isinstance(payload, dict):
                ctx.current_tool_args = payload
        elif event in ("post_tool_use", "on_tool_error"):
            ctx.current_tool_result = "" if payload is None else str(payload)
        elif event == "pre_answer_emit":
            ctx.answer = None if payload is None else str(payload)
        return False
    if kind is DecisionKind.BLOCK:
        if event == "pre_tool_use":
            # Per-tool skip — task continues. Signal stored in shared for the
            # FC loop to read after dispatcher.emit("pre_tool_use", ctx) returns.
            ctx.shared["_skip_current_tool"] = True
            ctx.shared["_skip_reason"] = decision.reason or f"{comp.name}: block"
            return True
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


def _wire_ctx_helpers(ctx: ComponentContext, dispatcher: Dispatcher) -> None:
    """Wire the per-task ctx helper methods. The baseline FC loop owns
    file_read / url_fetch / web_search as registered tools; ctx.fetch /
    ctx.read_file stay None — components that need to read a file directly
    can still use the standard library or invoke the underlying tool function.
    """
    ctx._impl_chat = _make_chat_impl()
    ctx._impl_emit = lambda name, fields: dispatcher.emit(name, ctx)




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
    _wire_ctx_helpers(ctx, dispatcher)

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

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": ctx.system_prompt},
        {"role": "user", "content": ctx.prompt or ""},
    ]

    last_result: dict[str, Any] | None = None
    iteration = 0

    # --- FC loop -------------------------------------------------------------
    for iteration in range(1, MAX_ITERATIONS + 1):
        ctx.current_iter = iteration

        # Per-turn pre_llm_request hook.
        dispatcher.emit("pre_llm_request", ctx)
        if ctx.blocked:
            return _blocked_return("pre_llm_request")

        # Drain any queued post_llm_inject from prior turn into a system note.
        injects = ctx.shared.pop("post_llm_inject", None)
        if injects:
            messages.append({"role": "system",
                             "content": "\n\n".join(str(t) for t in injects)})

        call = log.emit("llm.requested", parent=root, iteration=iteration,
                        messages=messages, model=DEFAULT_MODEL)
        try:
            result = chat(messages=messages, tools=TOOL_SPECS,
                          tool_choice="auto", max_tokens=PER_TURN_MAX_TOKENS)
        except Exception as e:
            log.emit("llm.failed", parent=call, error=repr(e))
            log.emit("run.failed", parent=root, error=repr(e))
            log.close()
            return {"run_id": run_id, "answer": None, "error": repr(e),
                    "trace_path": str(log.path)}

        last_result = result
        ctx.raw_response = result.get("content") or ""
        ctx.shared["finish_reason"] = result.get("finish_reason")
        log.emit(
            "llm.responded", parent=call, iteration=iteration,
            content=ctx.raw_response,
            finish_reason=result.get("finish_reason"),
            tool_calls=result.get("tool_calls"),
            usage=result.get("usage"),
        )

        messages.append(result["assistant_message"])

        # Per-turn post_llm_response hooks (REWRITE here mutates ctx.raw_response;
        # we sync back into messages[-1] so the model history reflects it).
        prev_raw = ctx.raw_response
        dispatcher.emit("post_llm_response", ctx)
        if ctx.blocked:
            return _blocked_return("post_llm_response")
        dispatcher.emit("post_llm_response_raw", ctx)
        if ctx.blocked:
            return _blocked_return("post_llm_response_raw")
        if ctx.raw_response != prev_raw:
            messages[-1] = {**messages[-1], "content": ctx.raw_response}

        tool_calls = result.get("tool_calls") or []
        if (result.get("finish_reason") or "") == "length":
            dispatcher.emit("on_length_truncation", ctx)
            if ctx.blocked:
                return _blocked_return("on_length_truncation")
        if not (ctx.raw_response or "").strip() and not tool_calls:
            dispatcher.emit("on_empty_response", ctx)
            if ctx.blocked:
                return _blocked_return("on_empty_response")

        # No tool calls → this is the final turn.
        if not tool_calls:
            break

        # Per-tool-call dispatch.
        for tc in tool_calls:
            tool_name = tc.get("name") or ""
            raw_args = tc.get("arguments")
            tool_args = raw_args if isinstance(raw_args, dict) else {}
            tool_call_id = tc.get("id") or f"call_{iteration}_{tool_name}"

            ctx.current_tool_name = tool_name
            ctx.current_tool_args = tool_args
            ctx.current_tool_call_id = tool_call_id
            ctx.current_tool_result = None
            ctx.current_tool_success = True

            dispatcher.emit("pre_tool_use", ctx)
            if ctx.blocked:
                return _blocked_return("pre_tool_use")

            if ctx.shared.pop("_skip_current_tool", False):
                reason = ctx.shared.pop("_skip_reason", "blocked by component")
                tool_result = f"[tool call skipped: {reason}]"
                ctx.current_tool_result = tool_result
                ctx.current_tool_success = False
            else:
                tcall_id = log.emit(
                    "tool.called", parent=call, iteration=iteration,
                    name=tool_name, args=ctx.current_tool_args,
                    tool_call_id=tool_call_id,
                )
                tool_result = dispatch_tool(
                    tool_name,
                    ctx.current_tool_args if isinstance(ctx.current_tool_args, dict) else {},
                )
                ctx.current_tool_result = tool_result
                ctx.current_tool_success = not (tool_result or "").startswith("ERROR")
                log.emit("tool.responded", parent=tcall_id, name=tool_name,
                         result=tool_result, success=ctx.current_tool_success)

            dispatcher.emit("post_tool_use", ctx)
            if ctx.blocked:
                return _blocked_return("post_tool_use")
            if not ctx.current_tool_success:
                dispatcher.emit("on_tool_error", ctx)
                if ctx.blocked:
                    return _blocked_return("on_tool_error")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": ctx.current_tool_result or "",
            })
    else:
        log.emit("run.exhausted_iterations", parent=root,
                 iterations=MAX_ITERATIONS)

    # Default extraction: pull "FINAL ANSWER: <x>" or fallback to last content.
    ctx.answer = _extract_final_answer(ctx.raw_response or "")

    dispatcher.emit("pre_answer_emit", ctx)
    if ctx.blocked:
        return _blocked_return("pre_answer_emit")

    log.emit(
        "answer.emitted", parent=root, answer=ctx.answer,
        raw_answer=ctx.raw_response,
        finish_reason=(last_result.get("finish_reason") if last_result else None),
        iterations_used=iteration,
    )

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
