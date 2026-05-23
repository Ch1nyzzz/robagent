"""tau2 candidate mh_tau2_iter4_baseline_transfer_reason_lookup.

Hypothesis
----------
On tasks whose reward is the `transfer_to_human_agents` action, the agent
loses the reward by guessing the `reason` argument. That argument must be an
exact code from the bank's tiered transfer-reason-code policy, which lives in
the knowledge base; the tool's own description even says to search the KB for
it before calling. The agent instead invents a plausible-sounding reason and
gets a lower-priority / less-specific code than gold. Forcing the agent to
retrieve the transfer reason-code policy before it transfers — and to pick the
highest-tier, most-specific code — will raise train-30 reward.

Mechanism observed (banking sims, all prior candidates)
-------------------------------------------------------
Both ACTION-basis transfer tasks fail on the `reason` arg even though the DB
state matches gold:
  * task_004 — identity-verification failure during an email change. Agent sent
    reason="customer_frustrated_demands_human" (Tier 3); gold is
    "account_ownership_dispute" (Tier 1). The agent never searched the KB for
    the transfer reason policy.
  * task_014 — customer holds a mailed letter about a program the KB cannot
    confirm. Agent sent reason="kb_search_unsuccessful_customer_requests_
    transfer"; gold is "unconfirmed_external_communication" (the policy's
    dedicated code for "they have a letter/email/flyer you cannot verify").
    The agent searched the KB for the program but never for the reason policy.
This reproduces in every prior candidate run (v0, iter1, iter2, iter3): both
tasks have db_check=true and reward 0 purely from the failed action check.

Change
------
1. The system prompt gains a general <transfer_protocol>: when a tool argument
   must come from an official coded list (the transfer `reason` is the named
   example), retrieve that list from the knowledge base before calling the
   tool, then choose the value from the highest-priority tier that applies and
   the most specific code within it — never a generic disposition or catch-all
   code when a specific operational code fits. No identifiers, no task hints.
2. A deterministic guard (transfer_guard.TransferReasonGuard) tracks whether
   the transfer reason-code policy has been retrieved by any KB search this
   conversation. When the agent's turn calls transfer_to_human_agents before
   that retrieval, the turn is regenerated with a transient note telling it to
   search for the policy first. The nudge is capped per conversation and never
   blocks a transfer outright, so a legitimate transfer is never suppressed.

The LLM still chooses the final reason code; the guard only guarantees the
authoritative policy is in context when it does.
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
    SystemMessage,
    ToolMessage,
)
from tau2.utils.llm_utils import generate

from agent_tau2.mh_tau2_iter4_baseline_transfer_reason_lookup.transfer_guard import (
    TransferReasonGuard,
)

# General, domain-agnostic guidance appended to the system prompt.
TRANSFER_PROTOCOL = """
<transfer_protocol>
Some tool arguments must be an exact value from an official coded list rather
than a phrase you compose yourself. When a tool's description says a valid
value can be found in the knowledge base or the policy, retrieve that list
before you call the tool — do not guess.

The `reason` argument of transfer_to_human_agents works this way. Before you
call transfer_to_human_agents:
1. If you have not already retrieved the bank's transfer reason-code policy in
   this conversation, search the knowledge base for it first (for example with
   a query like "transfer to human agents reason codes"). Do not transfer
   until that policy is in front of you.
2. Read the policy's tiers. Always pick the reason from the HIGHEST-priority
   tier that genuinely applies, and within that tier pick the most specific
   code that matches the customer's actual underlying situation. Do not fall
   back to a generic "customer is frustrated", "could not find information",
   or catch-all code when a specific operational code fits.
3. Then call transfer_to_human_agents with that exact reason code.
</transfer_protocol>
""".strip()

# Maximum number of generations per turn (1 original + 1 steered retry).
MAX_GENERATIONS = 2


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


class TransferReasonAgent(LLMAgent):
    """LLMAgent that consults the transfer reason-code policy before transferring."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._guard = TransferReasonGuard()

    @property
    def system_prompt(self) -> str:
        base = SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        )
        return base + "\n" + TRANSFER_PROTOCOL

    def get_init_state(self, message_history=None) -> LLMAgentState:
        self._guard.reset()
        state = super().get_init_state(message_history)
        # Replay any pre-seeded tool results so the guard's view is current.
        for msg in state.messages:
            if isinstance(msg, ToolMessage):
                self._guard.observe_tool_result(getattr(msg, "content", None))
        return state

    def _generate(self, state: LLMAgentState, guidance: Optional[str]) -> AssistantMessage:
        """Run one LLM generation, optionally with a transient turn note."""
        system_messages = state.system_messages
        if guidance:
            base = state.system_messages[0].content
            system_messages = [
                SystemMessage(
                    role="system",
                    content=f"{base}\n\n<turn_note>\n{guidance}\n</turn_note>",
                )
            ]
        messages = system_messages + state.messages
        return generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            call_name="agent_response",
            **self.llm_args,
        )

    def generate_next_message(self, message, state: LLMAgentState):
        """Respond to a user or tool message, consulting the reason policy first."""
        # Mirror LLMAgent: incoming message(s) are appended to history.
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        # Update the guard from any tool results just received.
        for text in _tool_result_texts(message):
            self._guard.observe_tool_result(text)

        guidance: Optional[str] = None
        assistant_message: Optional[AssistantMessage] = None
        for _ in range(MAX_GENERATIONS):
            assistant_message = self._generate(state, guidance)
            if not self._guard.needs_lookup(assistant_message):
                break
            # Regenerate once, steering the agent to fetch the reason policy.
            self._guard.register_nudge()
            guidance = self._guard.guidance()

        state.messages.append(assistant_message)
        return assistant_message, state


def build_agent(tools, domain_policy, **kwargs):
    return TransferReasonAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
