"""ComponentLLMAgent + `build_agent` entry point.

The runtime composes the active workflow's component set into a single
`ComponentLLMAgent`, which subclasses tau2's stock `LLMAgent`. Dispatch
goes through ONE core event dispatcher (`self._disp`) — the legacy
`_by_mount` bucketing was removed during the event-runtime cleanup.
Existing components keep firing via the `Component.listens = mount.value`
alias (set by `Component.__post_init__`).

Events emitted per lifecycle phase:

  * Setup (one-shot at __init__):
        task_received  →  pre_context_build (= "pre_context_build")
            → session_start (= "session_start") → pre_agent_construct
        INJECT_CONTEXT decisions accumulate into the system_prompt
        extension; subsequent components see prior contributions via
        `ctx.proposed_system_prompt`.

  * Per turn (generate_next_message):
        post_tool_result_raw + (gated) on_tool_error + post_tool_use
            [if incoming message was a ToolMessage]
            → pre_llm_request
            → [LLM call]
            → post_llm_response (= "post_llm_response") — REWRITE_TOOL_ARGS
              rewrites first tool_call; BLOCK clears tool_calls; INJECT_CONTEXT
              queues SystemMessage for next turn
            → post_llm_response_raw (Tier-1 alias) + (gated) on_empty_response /
              on_no_tool_call_emitted
            → pre_tool_arg_validation (per ToolCall) → pre_tool_use
              (= "pre_tool_use"; per ToolCall) — REWRITE_TOOL_ARGS rewrites
              args; BLOCK drops THIS tool call; DEFER falls back to ALLOW

USER_PROMPT_SUBMIT / STOP / SESSION_END are reserved (declared in Mount and
validated by `policy.py`) but not yet dispatched. Adding them is local to
this file — no component code changes.

The runtime appends one JSONL row per component fire to
`.component-state/<run_tag>/fired.jsonl` for durability auditing.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from tau2.agent.llm_agent import LLMAgent, AGENT_INSTRUCTION, SYSTEM_PROMPT
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)

from agent.llm import chat as _bench_chat
from meta_harness.component_runtime_core.dispatcher import Dispatcher as _CoreDispatcher

from .policy import ComponentPolicyError, validate_decision
from .registry import COMPONENTS_DIR_DEFAULT, load_components_from_dir
from .types import (
    Component,
    ComponentContext,
    DecisionKind,
    StateScope,
)
from .workflow import Workflow


def _make_chat_impl():
    """ctx.chat helper bound to the locked SUT model. Used by Tier-1
    event subscribers that declare Capability.LLM_CALL."""
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


def _apply_tau2_decision(ctx: ComponentContext, decision,
                          comp: Component) -> bool:
    """Unified decision applier for every tau2 event (cleanup of the
    parallel legacy `_dispatch_*` paths). Event-aware so the same
    DecisionKind can mean different things at different lifecycle points:

      INJECT_CONTEXT @ pre-LLM events  → accumulate into
            ctx.proposed_system_prompt (so subsequent subscribers see it)
            AND append to ctx.shared['tier1_prompt_inject'] so the outer
            code can read out the joined extension.
      INJECT_CONTEXT @ post_llm_response* → append to
            ctx.shared['_tau2_post_llm_injections']
      INJECT_CONTEXT @ post_tool_use*   → append to
            ctx.shared['_tau2_post_tool_injections']
      REWRITE_TOOL_ARGS @ pre_tool_use* → rewrite ctx.tool_call
      REWRITE_TOOL_ARGS @ post_llm_response* → rewrite ctx.tool_call AND
            mark ctx.shared['_tau2_first_tool_call_rewritten']=True so
            the outer code knows to swap tool_calls[0]
      BLOCK @ pre_tool_use*  → mark ctx.shared['_tau2_drop_this_tool']=True
            and STOP further subscribers (this ToolCall is dropped)
      BLOCK @ post_llm_response* → mark ctx.shared['_tau2_clear_tool_calls']
            =True and STOP further subscribers
      BLOCK @ anything else → ctx.blocked=True, terminate
      DEFER → fall back to ALLOW (v1; replay queue is future work)
    """
    event = ctx.event or ""
    kind = decision.kind
    if kind is DecisionKind.ALLOW or kind is DecisionKind.DEFER:
        return False

    if kind is DecisionKind.INJECT_CONTEXT:
        text = str(decision.payload or "")
        if event in (
            "task_received",
            "pre_context_build",
            "session_start",
            "pre_agent_construct",
            "user_prompt_submit",
            "pre_llm_request",
        ):
            # Accumulate into proposed_system_prompt so subsequent
            # subscribers in the same emit() see the contribution.
            if ctx.proposed_system_prompt is not None:
                ctx.proposed_system_prompt = (
                    ctx.proposed_system_prompt + "\n\n" + text
                )
            ctx.shared.setdefault("tier1_prompt_inject", []).append(text)
        elif event in ("post_tool_use", "post_tool_result_raw"):
            ctx.shared.setdefault("_tau2_post_tool_injections", []).append(text)
        elif event in ("post_llm_response", "post_llm_response_raw",
                        "on_length_truncation", "on_empty_response",
                        "on_no_tool_call_emitted"):
            ctx.shared.setdefault("_tau2_post_llm_injections", []).append(text)
        else:
            ctx.shared.setdefault("post_llm_inject", []).append(text)
        return False

    if kind is DecisionKind.REWRITE_TOOL_ARGS:
        if ctx.tool_call is not None:
            tc = ctx.tool_call
            ctx.tool_call = ToolCall(
                id=tc.id, name=tc.name,
                arguments=dict(decision.payload),
                requestor=tc.requestor,
            )
            if event in ("post_llm_response", "post_llm_response_raw"):
                ctx.shared["_tau2_first_tool_call_rewritten"] = True
        return False

    if kind is DecisionKind.BLOCK:
        if event in ("pre_tool_use", "pre_tool_arg_validation"):
            ctx.shared["_tau2_drop_this_tool"] = True
            return True
        if event in ("post_llm_response", "post_llm_response_raw"):
            ctx.shared["_tau2_clear_tool_calls"] = True
            return True
        ctx.blocked = True
        ctx.blocked_reason = decision.reason or f"{comp.name}: block"
        return True

    return False


def _validate_event(comp: Component, event_name: str,
                    kind: DecisionKind) -> None:
    """Adapter: core dispatcher passes event_name string; policy normalises."""
    validate_decision(comp.cls, event_name, kind)


ROOT = Path(__file__).resolve().parent.parent.parent


# --- trace sink --------------------------------------------------------------


def _trace_dir() -> Path:
    tag = os.environ.get("COMPONENT_RUN_TAG", "default")
    d = Path(os.environ.get("COMPONENT_STATE_DIR", ".component-state")) / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def _trace_event(component_name: str, event_name: str,
                 decision_kind_value: str, extra: dict) -> None:
    """Trace sink for the core.Dispatcher (signature
    `(comp_name, event_name, decision_kind_value, extra)`). Writes one
    JSONL row per fire to `.component-state/<run_tag>/fired.jsonl`."""
    rec = {
        "ts": time.time(),
        "component": component_name,
        "event": event_name,
        "mount": event_name,                   # legacy log-reader compat
        "decision": decision_kind_value,
        **extra,
    }
    with (_trace_dir() / "fired.jsonl").open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --- dispatch ----------------------------------------------------------------


class ComponentLLMAgent(LLMAgent):
    """LLMAgent whose `system_prompt` and `generate_next_message` are
    interposed by the active component set."""

    def __init__(self, tools, domain_policy, llm, llm_args=None,
                 workflow: Optional[Workflow] = None,
                 components_by_name: Optional[dict[str, Component]] = None):
        super().__init__(tools=tools, domain_policy=domain_policy,
                         llm=llm, llm_args=llm_args)
        self._workflow = workflow or Workflow()
        self._components_by_name = components_by_name or {}
        self._tool_names = tuple(getattr(t, "name", "") for t in tools)

        # Per-session SESSION-scope state (keyed by component name).
        self._session_state: dict[str, dict] = {
            n: {} for n in self._workflow.active_nodes()
        }

        # Single core dispatcher — components are bucketed by their
        # string `listens` field. Existing components keep firing via
        # the `Component.__post_init__` mount→listens alias, so the
        # workflow.active_nodes() insertion order + Python's stable
        # priority sort reproduce the legacy (priority, insertion) order.
        active: list[Component] = []
        for name in self._workflow.active_nodes():
            comp = self._components_by_name.get(name)
            if comp is not None:
                active.append(comp)
        self._disp = _CoreDispatcher(
            active,
            validate_decision=_validate_event,
            apply_decision=_apply_tau2_decision,
            trace_sink=_trace_event,
        )

        # Cache the static prompt injections (task_received,
        # pre_context_build, session_start, pre_agent_construct) once.
        self._prompt_injection = self._collect_prompt_injection()

    # ---- event helpers -----------------------------------------------------

    def _make_ctx(self, event_name: str, **fields) -> ComponentContext:
        """Build a ComponentContext with capability hooks wired. The
        dispatcher routes via `ctx.event` / the string-keyed
        `Component.listens` field."""
        ctx = ComponentContext(
            domain_policy=self.domain_policy,
            tool_names=self._tool_names,
            state=self._session_state,
            event=event_name,
            **fields,
        )
        ctx._impl_chat = _make_chat_impl()
        ctx._impl_emit = lambda name, payload: self._disp.emit(name, ctx)
        return ctx

    # ---- prompt -------------------------------------------------------------

    def _collect_prompt_injection(self) -> str:
        """Fire the 4 setup-phase events sequentially; each emit's
        accumulated INJECT_CONTEXT fragments roll into the next event's
        `proposed_system_prompt` so multi-component pipelines compose.

        Events fired (priority-ordered within each):
          task_received       — Tier-1 lifecycle anchor (no mount alias)
          pre_context_build   — also "pre_context_build", so
                                legacy `mount="pre_context_build"`
                                components fire here via `listens` alias
          session_start       — "session_start"
          pre_agent_construct — Tier-1 only
        """
        proposed = SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy,
            agent_instruction=AGENT_INSTRUCTION,
        )
        parts: list[str] = []
        for event_name, mount in (
            ("task_received", "pre_context_build"),
            ("pre_context_build", "pre_context_build"),
            ("session_start", "session_start"),
            ("pre_agent_construct", "session_start"),
        ):
            ctx = self._make_ctx(
                event_name, mount,
                proposed_system_prompt=proposed,
            )
            self._disp.emit(event_name, ctx)
            for fragment in (ctx.shared.get("tier1_prompt_inject") or []):
                if fragment not in parts:
                    parts.append(str(fragment))
            # apply_decision already accumulated injections into
            # ctx.proposed_system_prompt; carry forward to next emit.
            proposed = ctx.proposed_system_prompt or proposed

        return "\n\n".join(parts)

    @property
    def system_prompt(self) -> str:
        base = SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy,
            agent_instruction=AGENT_INSTRUCTION,
        )
        if not self._prompt_injection:
            return base
        return (f"{base}\n\n<components_prompt_injection>\n"
                f"{self._prompt_injection}\n</components_prompt_injection>")

    # ---- turn ---------------------------------------------------------------

    def generate_next_message(self, message, state):
        # POST_TOOL_USE → per-ToolMessage emit; result/error events first
        # to let downstream rewrite injections, then mount.value alias so
        # legacy POST_TOOL_USE components fire.
        if isinstance(message, (ToolMessage, MultiToolMessage)):
            self._fire_post_tool_use(message, state)

        # pre_llm_request: anything wired to react just before the SUT call.
        pre_ctx = self._make_ctx(
            "pre_llm_request", "post_llm_response",
            incoming_message=message,
            history=list(state.messages),
        )
        self._disp.emit("pre_llm_request", pre_ctx)

        assistant_message, state = super().generate_next_message(message, state)

        # POST_LLM_RESPONSE (mount.value + Tier-1 alias) — REWRITE_TOOL_ARGS,
        # BLOCK (clear tool_calls), INJECT_CONTEXT (next-turn SystemMessage).
        assistant_message = self._fire_post_llm_response(
            assistant_message, state
        )

        # Synthesised failure-mode events keyed off the (possibly rewritten)
        # assistant message shape.
        content = getattr(assistant_message, "content", None) or ""
        tcs = list(getattr(assistant_message, "tool_calls", None) or [])
        if not content.strip() and not tcs:
            failure_ctx = self._make_ctx(
                "on_empty_response", "post_llm_response",
                assistant_message=assistant_message,
                history=list(state.messages),
            )
            self._disp.emit("on_empty_response", failure_ctx)
        if not tcs:
            no_tc_ctx = self._make_ctx(
                "on_no_tool_call_emitted", "post_llm_response",
                assistant_message=assistant_message,
                history=list(state.messages),
            )
            self._disp.emit("on_no_tool_call_emitted", no_tc_ctx)

        # PRE_TOOL_USE: per-call rewrite / block.
        if assistant_message.tool_calls:
            assistant_message = self._fire_pre_tool_use(
                assistant_message, state
            )

        # Replace the recorded last message with the (possibly rewritten)
        # assistant message so the conversation history reflects what
        # actually went to the env.
        state.messages[-1] = assistant_message
        return assistant_message, state

    # ---- pre_tool_use -------------------------------------------------------

    def _fire_pre_tool_use(self, msg: AssistantMessage,
                            state) -> AssistantMessage:
        """For each ToolCall, fire pre_tool_arg_validation then
        "pre_tool_use" through the unified dispatcher. apply_decision
        handles REWRITE_TOOL_ARGS (rewrite ctx.tool_call) and BLOCK
        (mark drop). DEFER falls back to ALLOW."""
        kept: list[ToolCall] = []
        for call in (msg.tool_calls or []):
            initial = ToolCall(
                id=call.id, name=call.name,
                arguments=dict(call.arguments),
                requestor=call.requestor,
            )

            # Narrow schema-check phase first; lets a deterministic
            # validator rewrite before the main per-mount components fire.
            arg_ctx = self._make_ctx(
                "pre_tool_arg_validation", "pre_tool_use",
                tool_call=initial,
                history=list(state.messages),
            )
            arg_ctx.shared.pop("_tau2_drop_this_tool", None)
            self._disp.emit("pre_tool_arg_validation", arg_ctx)
            if arg_ctx.shared.get("_tau2_drop_this_tool"):
                continue
            current_call = arg_ctx.tool_call or initial

            # Main pre_tool_use phase (mount.value event = "pre_tool_use").
            main_ctx = self._make_ctx(
                "pre_tool_use", "pre_tool_use",
                tool_call=current_call,
                history=list(state.messages),
            )
            main_ctx.shared.pop("_tau2_drop_this_tool", None)
            self._disp.emit("pre_tool_use", main_ctx)
            if main_ctx.shared.get("_tau2_drop_this_tool"):
                continue
            kept.append(main_ctx.tool_call or current_call)

        return msg.model_copy(update={"tool_calls": kept or None})

    # ---- post_llm_response --------------------------------------------------

    def _fire_post_llm_response(self, msg: AssistantMessage,
                                 state) -> AssistantMessage:
        """One emit per AssistantMessage. apply_decision handles:
          REWRITE_TOOL_ARGS → rewrites ctx.tool_call (the first tool_call),
                              marks `_tau2_first_tool_call_rewritten`.
          BLOCK             → marks `_tau2_clear_tool_calls`.
          INJECT_CONTEXT    → appends to `_tau2_post_llm_injections`.
        After the emit, apply ctx.shared signals to the AssistantMessage
        and queue a SystemMessage for the next turn if injections exist.
        """
        # Skip the work if no subscriber is interested in this event.
        if not (self._disp.subscribers("post_llm_response")
                or self._disp.subscribers("post_llm_response_raw")
                or self._disp.subscribers("on_length_truncation")):
            return msg

        current_tcs = list(msg.tool_calls or [])
        for event_name in ("post_llm_response", "post_llm_response_raw"):
            ctx = self._make_ctx(
                event_name, "post_llm_response",
                assistant_message=msg,
                tool_call=current_tcs[0] if current_tcs else None,
                history=list(state.messages),
            )
            ctx.shared.pop("_tau2_clear_tool_calls", None)
            ctx.shared.pop("_tau2_first_tool_call_rewritten", None)
            self._disp.emit(event_name, ctx)

            if ctx.shared.get("_tau2_clear_tool_calls"):
                current_tcs = []
            elif ctx.shared.get("_tau2_first_tool_call_rewritten") and current_tcs:
                current_tcs[0] = ctx.tool_call

        injections = ctx.shared.get("_tau2_post_llm_injections") or []
        if injections:
            note = SystemMessage(
                role="system",
                content="<components_post_llm_response>\n" +
                        "\n\n".join(injections) +
                        "\n</components_post_llm_response>",
            )
            state.system_messages.append(note)

        if current_tcs != list(msg.tool_calls or []):
            return msg.model_copy(
                update={"tool_calls": current_tcs or None}
            )
        return msg

    # ---- post_tool_use ------------------------------------------------------

    def _fire_post_tool_use(self, message, state) -> None:
        ctx_msgs = (message.tool_messages
                    if isinstance(message, MultiToolMessage) else [message])
        all_injections: list[str] = []
        for tm in ctx_msgs:
            # Per-ToolMessage: emit raw event, then optional error event,
            # then the mount.value event for legacy POST_TOOL_USE components.
            t1 = self._make_ctx(
                "post_tool_result_raw", "post_tool_use",
                incoming_message=tm,
                history=list(state.messages),
            )
            t1.shared.pop("_tau2_post_tool_injections", None)
            self._disp.emit("post_tool_result_raw", t1)
            if getattr(tm, "error", None):
                self._disp.emit("on_tool_error", t1)

            main = self._make_ctx(
                "post_tool_use", "post_tool_use",
                incoming_message=tm,
                history=list(state.messages),
            )
            self._disp.emit("post_tool_use", main)

            for tag in ("_tau2_post_tool_injections",):
                all_injections.extend(t1.shared.get(tag, []) or [])
                all_injections.extend(main.shared.get(tag, []) or [])

        if all_injections:
            note = SystemMessage(
                role="system",
                content="<components_post_tool_use>\n" +
                        "\n\n".join(all_injections) +
                        "\n</components_post_tool_use>",
            )
            state.system_messages.append(note)


# --- entry point -------------------------------------------------------------


def build_agent(tools, domain_policy, **kwargs):
    """tau2 candidate entry. Identical signature to agent_tau2/v0/agent.py.

    Active component set is selected via env vars (set by
    meta_harness_components.py):

      COMPONENT_WORKFLOW  path to YAML workflow file
                          (default: meta_harness/workflows/tau2_main.yaml)
      COMPONENT_NAMES     comma-separated names; defensive check that the
                          outer loop has the workflow pinned correctly.
                          Empty = trust the workflow YAML.
      COMPONENT_DIR       directory of component .py files
                          (default: agent_tau2/components/)
      COMPONENT_FILES     colon-separated explicit paths (overrides
                          dir+workflow; primarily for unit tests).
      COMPONENT_RUN_TAG   namespacing tag for .component-state/<tag>/fired.jsonl
    """
    files_env = os.environ.get("COMPONENT_FILES", "").strip()
    if files_env:
        from .registry import load_components
        grouped = load_components(files_env.split(":"))
        components_by_name = {
            c.name: c for comps in grouped.values() for c in comps
        }
        wf = Workflow(nodes=tuple(components_by_name.keys()))
    else:
        workflow_path = Path(os.environ.get(
            "COMPONENT_WORKFLOW",
            str(ROOT / "meta_harness" / "workflows" / "tau2_main.yaml"),
        ))
        if not workflow_path.exists():
            wf = Workflow()
        else:
            wf = Workflow.from_yaml(workflow_path)

        names_env = os.environ.get("COMPONENT_NAMES", "").strip()
        names_from_env = {n for n in names_env.split(",") if n}
        if names_from_env and names_from_env != set(wf.active_nodes()):
            raise SystemExit(
                f"COMPONENT_NAMES {sorted(names_from_env)} ≠ workflow active "
                f"{sorted(wf.active_nodes())}; the outer loop must apply the "
                f"patch before invoking the runner."
            )

        comp_dir = os.environ.get("COMPONENT_DIR", str(COMPONENTS_DIR_DEFAULT))
        active = list(wf.active_nodes())
        if active:
            grouped = load_components_from_dir(comp_dir, only=active)
            components_by_name = {
                c.name: c for comps in grouped.values() for c in comps
            }
        else:
            components_by_name = {}

    return ComponentLLMAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
        workflow=wf,
        components_by_name=components_by_name,
    )
