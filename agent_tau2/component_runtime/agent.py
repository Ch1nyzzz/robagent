"""ComponentLLMAgent + `build_agent` entry point.

The runtime composes the active workflow's component set into a single
`ComponentLLMAgent`, which subclasses tau2's stock `LLMAgent`. v1 weaves
components into these surfaces:

  * PRE_CONTEXT_BUILD  — INJECT_CONTEXT decisions append to a per-session
                         prompt extension, BEFORE SESSION_START fires.
  * SESSION_START      — INJECT_CONTEXT decisions append to the same
                         extension (mounted into `system_prompt` once).
  * PRE_TOOL_USE       — REWRITE_TOOL_ARGS / BLOCK / DEFER on each emitted
                         ToolCall.
  * POST_LLM_RESPONSE  — REWRITE_TOOL_ARGS / BLOCK / INJECT_CONTEXT on the
                         full AssistantMessage (sub-LLM verifier slot).
  * POST_TOOL_USE      — INJECT_CONTEXT for each incoming ToolMessage.

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

from .policy import ComponentPolicyError, validate_decision
from .registry import COMPONENTS_DIR_DEFAULT, load_components_from_dir
from .types import (
    Component,
    ComponentContext,
    DecisionKind,
    Mount,
    StateScope,
)
from .workflow import Workflow


ROOT = Path(__file__).resolve().parent.parent.parent


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

        # Bucket active components by mount in workflow insertion order.
        self._by_mount: dict[Mount, list[Component]] = {m: [] for m in Mount}
        for name in self._workflow.active_nodes():
            comp = self._components_by_name.get(name)
            if comp is None:
                continue
            self._by_mount[comp.mount].append(comp)
        # Stable sort within each mount: (priority, insertion).
        for mount in Mount:
            self._by_mount[mount].sort(
                key=lambda c, ord_=self._workflow.nodes:
                    (c.priority, ord_.index(c.name))
            )

        # Per-session SESSION-scope state (keyed by component name).
        self._session_state: dict[str, dict] = {
            n: {} for n in self._workflow.active_nodes()
        }

        # Cache the static prompt injections (PRE_CONTEXT_BUILD +
        # SESSION_START) once at init.
        self._prompt_injection = self._collect_prompt_injection()

    # ---- prompt -------------------------------------------------------------

    def _collect_prompt_injection(self) -> str:
        """Fire PRE_CONTEXT_BUILD then SESSION_START; concatenate injections.

        PRE_CONTEXT_BUILD components receive a `proposed_system_prompt` set
        to the bare formatted SYSTEM_PROMPT. Each may INJECT_CONTEXT; the
        next handler sees the previous handler's contributions appended.
        SESSION_START fires after, with the same accumulator.
        """
        proposed = SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy,
            agent_instruction=AGENT_INSTRUCTION,
        )
        parts: list[str] = []
        for mount in (Mount.PRE_CONTEXT_BUILD, Mount.SESSION_START):
            for comp in self._by_mount.get(mount, []):
                ctx = ComponentContext(
                    mount=mount,
                    domain_policy=self.domain_policy,
                    tool_names=self._tool_names,
                    proposed_system_prompt=proposed,
                    state=self._session_state,
                )
                if comp.matcher is not None and not comp.matcher(ctx):
                    continue
                decision = comp.handler(ctx)
                validate_decision(comp.cls, comp.mount, decision.kind)
                if decision.kind is DecisionKind.INJECT_CONTEXT:
                    text = str(decision.payload)
                    parts.append(text)
                    proposed = proposed + "\n\n" + text
                    _trace(comp.name, comp.mount, decision.kind,
                           {"chars": len(text)})
                else:
                    _trace(comp.name, comp.mount, decision.kind, {})
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
        # POST_TOOL_USE: text injection for the next assistant turn.
        if isinstance(message, (ToolMessage, MultiToolMessage)):
            self._dispatch_post_tool_use(message, state)

        assistant_message, state = super().generate_next_message(message, state)

        # POST_LLM_RESPONSE: sub-LLM verifier / observer. May rewrite
        # tool_calls, block them entirely, or inject a note for next turn.
        assistant_message = self._dispatch_post_llm_response(
            assistant_message, state
        )

        # PRE_TOOL_USE: per-call rewrite / block.
        if assistant_message.tool_calls:
            assistant_message = self._dispatch_pre_tool_use(
                assistant_message, state
            )

        # Replace the recorded last message with the (possibly rewritten)
        # assistant message so the conversation history reflects what
        # actually went to the env.
        state.messages[-1] = assistant_message
        return assistant_message, state

    # ---- pre_tool_use -------------------------------------------------------

    def _dispatch_pre_tool_use(self, msg: AssistantMessage,
                               state) -> AssistantMessage:
        kept: list[ToolCall] = []
        for call in (msg.tool_calls or []):
            current_args = dict(call.arguments)
            drop = False
            for comp in self._by_mount.get(Mount.PRE_TOOL_USE, []):
                ctx = ComponentContext(
                    mount=Mount.PRE_TOOL_USE,
                    tool_call=ToolCall(
                        id=call.id, name=call.name,
                        arguments=current_args, requestor=call.requestor,
                    ),
                    domain_policy=self.domain_policy,
                    tool_names=self._tool_names,
                    history=list(state.messages),
                    state=self._session_state,
                )
                if comp.matcher is not None and not comp.matcher(ctx):
                    continue
                decision = comp.handler(ctx)
                validate_decision(comp.cls, comp.mount, decision.kind)
                _trace(comp.name, comp.mount, decision.kind,
                       {"tool": call.name})
                if decision.kind is DecisionKind.ALLOW:
                    continue
                if decision.kind is DecisionKind.REWRITE_TOOL_ARGS:
                    current_args = dict(decision.payload)
                elif decision.kind is DecisionKind.BLOCK:
                    drop = True
                    break
                elif decision.kind is DecisionKind.DEFER:
                    # v1: DEFER falls back to ALLOW; replay queue is future work.
                    continue
            if not drop:
                kept.append(ToolCall(
                    id=call.id, name=call.name,
                    arguments=current_args, requestor=call.requestor,
                ))
        return msg.model_copy(update={"tool_calls": kept or None})

    # ---- post_llm_response --------------------------------------------------

    def _dispatch_post_llm_response(self, msg: AssistantMessage,
                                    state) -> AssistantMessage:
        """Fires per AssistantMessage (with or without tool_calls).

        REWRITE_TOOL_ARGS rewrites the FIRST tool_call's arguments wholesale
        (gate by matcher if you want per-tool selectivity).
        BLOCK zeroes the entire tool_calls list.
        INJECT_CONTEXT appends a SystemMessage to state.system_messages for
        the next turn.
        """
        comps = self._by_mount.get(Mount.POST_LLM_RESPONSE, [])
        if not comps:
            return msg

        injections: list[str] = []
        current_tool_calls = list(msg.tool_calls or [])
        for comp in comps:
            ctx = ComponentContext(
                mount=Mount.POST_LLM_RESPONSE,
                assistant_message=msg,
                tool_call=(current_tool_calls[0] if current_tool_calls else None),
                domain_policy=self.domain_policy,
                tool_names=self._tool_names,
                history=list(state.messages),
                state=self._session_state,
            )
            if comp.matcher is not None and not comp.matcher(ctx):
                continue
            decision = comp.handler(ctx)
            validate_decision(comp.cls, comp.mount, decision.kind)
            _trace(comp.name, comp.mount, decision.kind, {})
            if decision.kind is DecisionKind.ALLOW:
                continue
            if decision.kind is DecisionKind.BLOCK:
                current_tool_calls = []
                break
            if decision.kind is DecisionKind.REWRITE_TOOL_ARGS and current_tool_calls:
                head = current_tool_calls[0]
                current_tool_calls[0] = ToolCall(
                    id=head.id, name=head.name,
                    arguments=dict(decision.payload),
                    requestor=head.requestor,
                )
            elif decision.kind is DecisionKind.INJECT_CONTEXT:
                injections.append(str(decision.payload))

        if injections:
            note = SystemMessage(
                role="system",
                content="<components_post_llm_response>\n" +
                        "\n\n".join(injections) +
                        "\n</components_post_llm_response>",
            )
            state.system_messages.append(note)

        if current_tool_calls != list(msg.tool_calls or []):
            return msg.model_copy(
                update={"tool_calls": current_tool_calls or None}
            )
        return msg

    # ---- post_tool_use ------------------------------------------------------

    def _dispatch_post_tool_use(self, message, state) -> None:
        ctx_msgs = (message.tool_messages
                    if isinstance(message, MultiToolMessage) else [message])
        injections: list[str] = []
        for tm in ctx_msgs:
            ctx = ComponentContext(
                mount=Mount.POST_TOOL_USE,
                incoming_message=tm,
                domain_policy=self.domain_policy,
                tool_names=self._tool_names,
                history=list(state.messages),
                state=self._session_state,
            )
            for comp in self._by_mount.get(Mount.POST_TOOL_USE, []):
                if comp.matcher is not None and not comp.matcher(ctx):
                    continue
                decision = comp.handler(ctx)
                validate_decision(comp.cls, comp.mount, decision.kind)
                _trace(comp.name, comp.mount, decision.kind,
                       {"tool": getattr(tm, "tool_name", None)})
                if decision.kind is DecisionKind.INJECT_CONTEXT:
                    injections.append(str(decision.payload))
        if injections:
            note = SystemMessage(
                role="system",
                content="<components_post_tool_use>\n" +
                        "\n\n".join(injections) +
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
