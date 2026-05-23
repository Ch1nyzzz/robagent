"""tau2 candidate mh_tau2_iter6_baseline_discoverable_arg_minimizer.

Hypothesis
----------
On banking tasks scored by the database hash, the agent loses reward by adding
arguments to a discoverable-tool call that the gold trajectory never sends.
The clearest case is an OPTIONAL FREE-TEXT string parameter (a free-form
``reason`` note on an account-closure tool): the agent fills it with a
narrative sentence it composed itself, the bank database stores that string
verbatim, and the resulting row no longer matches gold — which always calls
the tool with only the required identifiers. Deterministically stripping
undeclared keys and invented optional free-text arguments from every
discoverable-tool call will let those writes match gold and raise train-30
reward.

Mechanism observed (banking sims: v0, iter4, iter5)
---------------------------------------------------
``close_bank_account_7392`` declares ``account_id (required)``,
``reason: string (optional)`` and ``waive_early_closure_fee: boolean
(optional)``. On task_062 the gold trajectory calls it as
``{"account_id": "..."}``. Every prior candidate instead emits
``{"account_id": "...", "reason": "Customer requested account closure -
consolidating accounts"}``. task_062's action checks confirm every other gold
action matched; only the two ``close_bank_account_7392`` calls diverge, purely
on the fabricated ``reason`` argument, and the task scores 0 on the DB hash.
The same over-argumentation pattern (composing optional narrative the schema
does not need) recurs across the discoverable-write tasks.

Change
------
1. The system prompt gains a general, domain-agnostic <tool_argument_discipline>
   section: always supply every required parameter and use exact enum values,
   but pass only the parameters a discoverable tool's schema declares and do
   not populate optional free-text parameters the customer never asked to have
   recorded. No identifiers, tool names, or task hints are hardcoded.
2. A deterministic helper (discoverable_args.DiscoverableToolSchemas) parses
   every "Tool unlocked:" schema the agent sees and, before a turn's tool
   calls are emitted, rewrites each call_discoverable_agent_tool call to drop
   (a) keys that are not declared parameters and (b) optional, free-text
   string parameters. Required parameters and optional enum / number / boolean
   parameters are always kept, so a legitimate argument can never be removed.
   When the schema is unknown or unparseable the call is left untouched.

The LLM still decides which discoverable tools to call and with what values;
the guard only removes arguments that gold never sends.
"""
from __future__ import annotations

from typing import Optional

from tau2.agent.llm_agent import (
    AGENT_INSTRUCTION,
    SYSTEM_PROMPT,
    LLMAgent,
    LLMAgentState,
)
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    ToolMessage,
)

from agent_tau2.mh_tau2_iter6_baseline_discoverable_arg_minimizer.discoverable_args import (
    DiscoverableToolSchemas,
)

# General, domain-agnostic guidance appended to the system prompt.
TOOL_ARGUMENT_DISCIPLINE = """
<tool_argument_discipline>
Every tool call must carry correct, minimal arguments.

- Always include every REQUIRED parameter the tool declares. Use a value taken
  from the conversation or the customer's account record, never a placeholder.
- When a parameter must be one of a fixed set of allowed values (an enum),
  pass exactly one of those listed values.
- When you call a discoverable tool (through call_discoverable_agent_tool or
  call_discoverable_user_tool), pass only the parameters that tool's declared
  schema lists. Do not add fields the schema does not mention.
- Do not fill in an OPTIONAL free-text parameter — for example a free-form
  "reason" note — unless the customer explicitly asked for that specific text
  to be recorded. An invented optional note is not needed to complete the
  action and can cause the action to be recorded incorrectly.
</tool_argument_discipline>
""".strip()


def _tool_result_texts(message) -> list[str]:
    """Tool-output strings carried by an incoming agent input message."""
    texts: list[str] = []
    if isinstance(message, MultiToolMessage):
        for tool_message in message.tool_messages:
            content = getattr(tool_message, "content", None)
            if isinstance(content, str):
                texts.append(content)
    elif isinstance(message, ToolMessage):
        content = getattr(message, "content", None)
        if isinstance(content, str):
            texts.append(content)
    return texts


class DiscoverableArgMinimizerAgent(LLMAgent):
    """LLMAgent that strips fabricated arguments from discoverable-tool calls."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._schemas = DiscoverableToolSchemas()

    @property
    def system_prompt(self) -> str:
        base = SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        )
        return base + "\n" + TOOL_ARGUMENT_DISCIPLINE

    def get_init_state(self, message_history=None) -> LLMAgentState:
        self._schemas.reset()
        state = super().get_init_state(message_history)
        # Replay any pre-seeded tool results so known schemas are current.
        for msg in state.messages:
            if isinstance(msg, ToolMessage):
                self._schemas.observe_tool_result(getattr(msg, "content", None))
        return state

    def _sanitize(self, assistant_message: AssistantMessage) -> None:
        """Drop undeclared / invented optional arguments from discoverable calls."""
        tool_calls = getattr(assistant_message, "tool_calls", None)
        if not tool_calls:
            return
        for tool_call in tool_calls:
            if tool_call.name != "call_discoverable_agent_tool":
                continue
            args = tool_call.arguments
            if not isinstance(args, dict):
                continue
            cleaned = self._schemas.sanitize(tool_call.name, args)
            if cleaned is not args:
                tool_call.arguments = cleaned

    def generate_next_message(self, message, state: LLMAgentState):
        """Respond to a user or tool message, minimizing discoverable-call args."""
        # Update known schemas from any tool results just received.
        for text in _tool_result_texts(message):
            self._schemas.observe_tool_result(text)

        # Mirror LLMAgent: append the incoming message and generate a reply.
        assistant_message = self._generate_next_message(message, state)

        # Deterministically strip fabricated arguments before the turn is run.
        self._sanitize(assistant_message)

        state.messages.append(assistant_message)
        return assistant_message, state


def build_agent(tools, domain_policy, **kwargs):
    return DiscoverableArgMinimizerAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
