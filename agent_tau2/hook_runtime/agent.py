"""HookedLLMAgent + `build_agent` entry point.

The runtime composes a frontier hook set + the candidate hook into a single
`HookedLLMAgent`, which subclasses tau2's stock `LLMAgent` and weaves hooks
into two surfaces in v1:

  * SESSION_START — `system_prompt` is extended with the concatenated text
    of every CHANNEL / *_INJECT_CONTEXT decision at session start.
  * PRE_TOOL_USE — every `ToolCall` emitted by the LLM is run through the
    matched hooks; REWRITE_TOOL_ARGS mutates `.arguments`, BLOCK drops the
    call. (DEFER is recorded but in v1 falls back to ALLOW.)
  * POST_TOOL_USE — every incoming `ToolMessage` triggers INJECT_CONTEXT
    hooks (text appended to the next assistant turn's system context).

USER_PROMPT_SUBMIT, STOP, SESSION_END are reserved (declared in HookEvent
and validated by `policy.py`) but not yet dispatched. Adding them is local
to this file — no hook code changes.

The runtime appends one JSONL row per hook fire to
`.hook-state/<run_tag>/fired.jsonl` for durability auditing.
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

from .policy import HookPolicyError, validate_decision
from .registry import HOOKS_DIR_DEFAULT, load_hooks_from_dir
from .types import DecisionKind, Hook, HookContext, HookEvent


# --- trace sink --------------------------------------------------------------


def _trace_dir() -> Path:
    tag = os.environ.get("HOOK_RUN_TAG", "default")
    d = Path(os.environ.get("HOOK_STATE_DIR", ".hook-state")) / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def _trace(hook_name: str, event: HookEvent, decision_kind: DecisionKind,
           extra: dict) -> None:
    rec = {
        "ts": time.time(),
        "hook": hook_name,
        "event": event.value,
        "decision": decision_kind.value,
        **extra,
    }
    with (_trace_dir() / "fired.jsonl").open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --- dispatch ----------------------------------------------------------------


class HookedLLMAgent(LLMAgent):
    """LLMAgent whose `system_prompt` and `generate_next_message` are
    interposed by the active hook set."""

    def __init__(self, tools, domain_policy, llm, llm_args=None,
                 hooks: Optional[dict[HookEvent, list[Hook]]] = None):
        super().__init__(tools=tools, domain_policy=domain_policy,
                         llm=llm, llm_args=llm_args)
        self._hooks: dict[HookEvent, list[Hook]] = hooks or {e: [] for e in HookEvent}
        self._tool_names = tuple(getattr(t, "name", "") for t in tools)
        # Cache the session-start injection so each turn's system_prompt is stable.
        self._session_start_injection = self._collect_session_start_text()

    # ---- prompt -------------------------------------------------------------

    def _collect_session_start_text(self) -> str:
        ctx = HookContext(
            event=HookEvent.SESSION_START,
            domain_policy=self.domain_policy,
            tool_names=self._tool_names,
        )
        parts: list[str] = []
        for hook in self._hooks.get(HookEvent.SESSION_START, []):
            if hook.matcher is not None and not hook.matcher(ctx):
                continue
            decision = hook.handler(ctx)
            validate_decision(hook.cls, hook.event, decision.kind)
            if decision.kind is DecisionKind.INJECT_CONTEXT:
                parts.append(str(decision.payload))
                _trace(hook.name, hook.event, decision.kind,
                       {"chars": len(str(decision.payload))})
            else:
                _trace(hook.name, hook.event, decision.kind, {})
        return "\n\n".join(parts)

    @property
    def system_prompt(self) -> str:
        base = SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy,
            agent_instruction=AGENT_INSTRUCTION,
        )
        if not self._session_start_injection:
            return base
        return f"{base}\n\n<hooks_session_start>\n{self._session_start_injection}\n</hooks_session_start>"

    # ---- turn ---------------------------------------------------------------

    def generate_next_message(self, message, state):
        # POST_TOOL_USE: text injection for the next assistant turn.
        if isinstance(message, (ToolMessage, MultiToolMessage)):
            self._dispatch_post_tool_use(message, state)

        assistant_message, state = super().generate_next_message(message, state)

        # PRE_TOOL_USE: rewrite / block tool_calls before they leave the agent.
        if assistant_message.tool_calls:
            assistant_message = self._dispatch_pre_tool_use(assistant_message, state)
            # state.messages was appended by super(); replace the last entry
            # so the rewritten tool_calls become the conversation record.
            state.messages[-1] = assistant_message

        return assistant_message, state

    # ---- pre_tool_use -------------------------------------------------------

    def _dispatch_pre_tool_use(self, msg: AssistantMessage,
                               state) -> AssistantMessage:
        kept: list[ToolCall] = []
        for call in (msg.tool_calls or []):
            current_args = dict(call.arguments)
            drop = False
            for hook in self._hooks.get(HookEvent.PRE_TOOL_USE, []):
                ctx = HookContext(
                    event=HookEvent.PRE_TOOL_USE,
                    tool_call=ToolCall(
                        id=call.id, name=call.name,
                        arguments=current_args, requestor=call.requestor,
                    ),
                    domain_policy=self.domain_policy,
                    tool_names=self._tool_names,
                    history=list(state.messages),
                )
                if hook.matcher is not None and not hook.matcher(ctx):
                    continue
                decision = hook.handler(ctx)
                validate_decision(hook.cls, hook.event, decision.kind)
                _trace(hook.name, hook.event, decision.kind,
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

    # ---- post_tool_use ------------------------------------------------------

    def _dispatch_post_tool_use(self, message, state) -> None:
        ctx_msgs = (message.tool_messages
                    if isinstance(message, MultiToolMessage) else [message])
        injections: list[str] = []
        for tm in ctx_msgs:
            ctx = HookContext(
                event=HookEvent.POST_TOOL_USE,
                incoming_message=tm,
                domain_policy=self.domain_policy,
                tool_names=self._tool_names,
                history=list(state.messages),
            )
            for hook in self._hooks.get(HookEvent.POST_TOOL_USE, []):
                if hook.matcher is not None and not hook.matcher(ctx):
                    continue
                decision = hook.handler(ctx)
                validate_decision(hook.cls, hook.event, decision.kind)
                _trace(hook.name, hook.event, decision.kind,
                       {"tool": getattr(tm, "tool_name", None)})
                if decision.kind is DecisionKind.INJECT_CONTEXT:
                    injections.append(str(decision.payload))
        if injections:
            # Append a one-turn system note. The runner places it on the
            # main message stack so it reaches the next LLM call once.
            note = SystemMessage(
                role="system",
                content="<hooks_post_tool_use>\n" +
                        "\n\n".join(injections) +
                        "\n</hooks_post_tool_use>",
            )
            state.system_messages.append(note)


# --- entry point -------------------------------------------------------------


def build_agent(tools, domain_policy, **kwargs):
    """tau2 candidate entry. Identical signature to agent_tau2/v0/agent.py.

    Hook set is selected via env vars (set by meta_harness_hooks.py):
      HOOK_FILES   colon-separated list of hook file paths (overrides dir+filter)
      HOOK_DIR     directory of hook files (default: agent_tau2/hooks/)
      HOOK_NAMES   comma-separated subset of HOOK.name to activate
      HOOK_RUN_TAG namespacing tag for .hook-state/<tag>/fired.jsonl
    """
    files_env = os.environ.get("HOOK_FILES", "").strip()
    if files_env:
        from .registry import load_hooks
        hooks = load_hooks(files_env.split(":"))
    else:
        names = os.environ.get("HOOK_NAMES", "").strip()
        only = [n for n in names.split(",") if n] if names else None
        hook_dir = os.environ.get("HOOK_DIR", str(HOOKS_DIR_DEFAULT))
        hooks = load_hooks_from_dir(hook_dir, only=only)

    return HookedLLMAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
        hooks=hooks,
    )
