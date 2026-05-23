"""tau2 candidate mh_tau2_iter3_baseline_exhaustive_audit.

Hypothesis
----------
On multi-transaction cash-back audit tasks the agent acts on the *wrong set*
of transactions: it eyeballs a subset of the customer's records instead of
auditing every one, so the disputes / reward corrections it submits are short
(or long) by a few transactions and the DB-hash check fails. Forcing a
complete, record-by-record audit before any resolution action will raise
train-30 reward.

Mechanism observed (v0 banking sims)
------------------------------------
The cash-back resolution set never matches gold:
  * task_018 — gold 6 disputes, agent corrected 3.
  * task_022 — gold 10 disputes, agent corrected 6.
  * task_029 — gold 6 disputes, agent submitted 4.
  * task_020 — gold 4 disputes, agent submitted 5.
  * task_027 — gold 4 disputes, agent submitted 5.
In every case the agent jumped to the resolution step after a partial scan; it
never laid out a transaction-by-transaction table covering all of the records
it had retrieved.

Change
------
1. The system prompt gains a general <thoroughness_protocol>: when asked to act
   on *every* item meeting a condition across the customer's records, the
   agent must examine each record individually, write an explicit checklist
   covering all of them, and take exactly one resolution action per flagged
   record. It also restates the floor/truncate rounding rule for numeric
   checks and the create-vs-finalise distinction. No identifiers, no rates.
2. A deterministic guard (`audit_guard.AuditCompletenessGuard`) tracks the
   universe of transaction ids surfaced by tool results and how many of them
   the agent has referenced in its own reasoning. When the agent's turn would
   submit a dispute / rewards correction while most of that universe is still
   unaccounted for, the turn is regenerated once with a note naming the
   still-unaudited transactions. The nudge fires at most once per conversation
   and never on small tasks (universe < 12), so it cannot derail a task that
   was already on track.

The LLM still makes every genuine decision; the guard only enforces that the
audit is complete before the writes happen.
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
)
from tau2.utils.llm_utils import generate

from agent_tau2.mh_tau2_iter3_baseline_exhaustive_audit.audit_guard import (
    AuditCompletenessGuard,
)

# General, domain-agnostic guidance appended to the system prompt.
THOROUGHNESS_PROTOCOL = """
<thoroughness_protocol>
When the user asks you to find, review, correct, or act on EVERY item that
meets some condition across a set of their records (for example: every
transaction with a reward error, every account that qualifies, every fee that
was mis-applied), the set of actions you take must come from a complete,
record-by-record pass — never from a sample, a quick scan, or a guess.

Before taking any resolution action on such a request:
1. Retrieve the full list of relevant records.
2. Examine every single record individually. Write an explicit, numbered
   checklist with exactly one line per record, covering ALL of them, and mark
   each line as "OK" or "needs action" with the reason and the corrected
   value.
3. Re-read the checklist and confirm its length equals the number of records
   you retrieved. If any record is missing from it, the audit is not done.
4. Take exactly one resolution action per record marked "needs action" — no
   more (do not invent extra actions) and no fewer (do not stop early).

For numeric checks (such as rewards or interest), compute each expected value
yourself from the rate and the amount, applying the rounding rule the policy
states — banking rewards points are truncated/floored, never rounded up — and
compare it against the recorded value. A record is in error only when the
value you independently compute differs from the one on file.

Distinguish creating a new request from finalising an already-resolved one: an
action that "applies", "corrects", or "finalises" a resolved case is valid
only for cases that were already resolved before this conversation began —
never for a request you yourself created during this conversation.
</thoroughness_protocol>
""".strip()

# Maximum number of generations per turn (1 original + retries).
MAX_GENERATIONS = 2


def _tool_result_texts(message) -> list[str]:
    """Tool-output strings carried by an incoming agent input message."""
    texts: list[str] = []
    if isinstance(message, MultiToolMessage):
        for tool_message in message.tool_messages:
            content = getattr(tool_message, "content", None)
            if isinstance(content, str):
                texts.append(content)
    else:
        # A single ToolMessage also has a `content` field; UserMessages too,
        # but only ToolMessages carry the bulk record dumps we care about.
        if getattr(message, "role", None) == "tool" or hasattr(message, "tool_calls"):
            content = getattr(message, "content", None)
            if isinstance(content, str) and getattr(message, "role", None) == "tool":
                texts.append(content)
    return texts


class ExhaustiveAuditAgent(LLMAgent):
    """LLMAgent that requires a complete audit before cash-back resolutions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._guard = AuditCompletenessGuard()

    @property
    def system_prompt(self) -> str:
        base = SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        )
        return base + "\n" + THOROUGHNESS_PROTOCOL

    def get_init_state(self, message_history=None) -> LLMAgentState:
        self._guard.reset()
        state = super().get_init_state(message_history)
        # Replay any pre-seeded history so the guard's universe is current.
        for msg in state.messages:
            if getattr(msg, "role", None) == "tool":
                self._guard.observe_tool_result(getattr(msg, "content", None))
            elif isinstance(msg, AssistantMessage):
                self._guard.observe_agent_text(getattr(msg, "content", None))
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
        """Respond to a user or tool message, enforcing a complete audit."""
        # Mirror LLMAgent: incoming message(s) are appended to history.
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        # Update the guard's universe from any tool results just received.
        for text in _tool_result_texts(message):
            self._guard.observe_tool_result(text)

        guidance: Optional[str] = None
        assistant_message: Optional[AssistantMessage] = None
        for _ in range(MAX_GENERATIONS):
            assistant_message = self._generate(state, guidance)
            if not self._guard.audit_incomplete(assistant_message):
                break
            # Regenerate with a note steering the agent to finish the audit.
            guidance = self._guard.guidance(assistant_message)

        # Record what this accepted turn examined, then commit it.
        self._guard.observe_agent_text(getattr(assistant_message, "content", None))
        state.messages.append(assistant_message)
        return assistant_message, state


def build_agent(tools, domain_policy, **kwargs):
    return ExhaustiveAuditAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
