"""tau2 candidate mh_tau2_iter2_baseline_write_discipline.

Hypothesis
----------
The agent loses DB-hash reward by performing *extra* write actions beyond what
the task requires. For banking_knowledge the reward is a DB-hash comparison:
the agent's set of state-modifying tool calls must exactly equal the gold set.
The agent is over-helpful — it volunteers statement credits, closes additional
accounts, and updates extra transactions that the customer never asked for.
Every such unsolicited write changes the final DB hash and zeroes the reward
even though every requested action was completed correctly.

Mechanism observed (frontier-0 banking sims)
--------------------------------------------
- task_044: agent performs an extra `apply_statement_credit_8472`
  ("retention" courtesy); gold applies no statement credit -> DB mismatch.
- task_048: agent performs an extra `close_credit_card_account_7834` on a card
  the customer did not ask to close (gold only closes one card) -> DB mismatch.
- task_026: agent calls `update_transaction_rewards_3847` five times; gold
  updates four -> one extra write -> DB mismatch.
- task_017: an extra cash-back dispute is filed beyond the gold set.
In every case `log_verification` and the explicitly-requested actions match,
yet `db_check.db_match` is False purely because of the surplus write.

Change
------
A `WriteDisciplineAgent` subclass of `LLMAgent`. The system prompt gains
general guidance: only perform write actions the customer explicitly requested
or the policy explicitly requires; never volunteer extra modifications.

Each turn is generated normally. A deterministic helper (`write_guard`)
classifies the turn's tool calls; if it contains state-modifying writes worth
reviewing, one focused verification LLM call examines the conversation, the
policy, and the proposed writes and returns any write that is clearly neither
requested nor policy-required. When something is flagged, the turn is
regenerated once with a transient note telling the agent to drop only the
unrequested modifications and keep everything the customer did ask for. The
verifier is conservative (flags only clearly-extra writes) and the regeneration
is capped at one, so a legitimately-required write is never silently dropped.
The LLM still makes every genuine decision.
"""
from __future__ import annotations

import json
import re
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
    UserMessage,
)
from tau2.utils.llm_utils import generate

from agent_tau2.mh_tau2_iter2_baseline_write_discipline import write_guard

# General, domain-agnostic guidance appended to the system prompt.
WRITE_DISCIPLINE_GUIDANCE = """
<write_discipline>
A write action is any tool call that creates, updates, or deletes data (for
example opening or closing an account, transferring or paying funds, filing a
dispute, applying a credit, updating a record).

Perform a write action ONLY when:
- the customer has explicitly asked for that specific action, or has clearly
  agreed to it, OR
- the policy explicitly requires it as a step of the customer's request.

Do NOT volunteer extra modifications. Even when well-intentioned, an
unrequested write is an error. In particular, never apply statement credits,
waive fees, open or close additional accounts, or file additional disputes on
your own initiative. Offering such a courtesy is fine; performing it without
the customer asking is not.

Before each write, confirm you can point to the customer's explicit request
for exactly that action. Do exactly what was asked — no more, no less.
</write_discipline>
""".strip()

# Verifier system prompt — a strict, conservative compliance reviewer.
VERIFIER_SYSTEM = """
You are a strict compliance reviewer for a bank customer-service agent. The
agent is about to perform one or more WRITE actions (tool calls that create,
update, or delete data).

Your job: identify any proposed write that the customer did NOT explicitly
request and that the bank policy does NOT explicitly require.

Definitions:
- SOLICITED: the customer explicitly asked for the action or clearly agreed to
  it, OR the policy requires it as a mandatory step of the customer's request
  (for example logging an identity verification). Keep these.
- UNSOLICITED: the agent is doing it proactively, as a courtesy, an upsell, or
  a retention gesture the customer never asked for — for example applying a
  statement credit, waiving a fee, opening or closing an extra account, or
  filing an extra dispute the customer did not name. Flag these.

Be conservative. When in doubt, treat the write as solicited and do NOT flag
it. Only flag writes that are clearly extra. Never flag identity-verification
logging.

Respond with ONLY a JSON object on a single line:
{"unsolicited": ["<exact description of each clearly-unsolicited write>", ...]}
If every proposed write is solicited, respond {"unsolicited": []}.
""".strip()

# Maximum number of generations per turn (1 original + 1 disciplined retry).
MAX_GENERATIONS = 2


def _parse_verdict(text: Optional[str]) -> list[str]:
    """Extract the `unsolicited` list from the verifier's JSON response.

    Fail-safe: any parsing problem yields an empty list, so a flaky verifier
    can never block a legitimate turn.
    """
    if not text:
        return []
    cleaned = text.strip()
    # Strip markdown code fences if present.
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*", "", cleaned).strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    # Isolate the first JSON object.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        obj = json.loads(cleaned[start : end + 1])
    except Exception:
        return []
    if not isinstance(obj, dict):
        return []
    items = obj.get("unsolicited")
    if not isinstance(items, list):
        return []
    return [str(x).strip() for x in items if str(x).strip()]


class WriteDisciplineAgent(LLMAgent):
    """LLMAgent that reviews and re-prompts turns that perform extra writes."""

    @property
    def system_prompt(self) -> str:
        base = SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        )
        return base + "\n" + WRITE_DISCIPLINE_GUIDANCE

    def _generate(
        self, state: LLMAgentState, guidance: Optional[str]
    ) -> AssistantMessage:
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

    def _verify_writes(
        self, state: LLMAgentState, writes: list[tuple[object, str]]
    ) -> list[str]:
        """Ask the verifier which proposed writes are clearly unsolicited."""
        transcript = write_guard.render_transcript(state.messages)
        proposed = "\n".join(
            f"{i + 1}. {desc}" for i, (_, desc) in enumerate(writes)
        )
        user_content = (
            "<conversation>\n"
            f"{transcript}\n"
            "</conversation>\n\n"
            "<bank_policy>\n"
            f"{self.domain_policy}\n"
            "</bank_policy>\n\n"
            "<proposed_write_actions>\n"
            f"{proposed}\n"
            "</proposed_write_actions>\n\n"
            "Review the proposed write actions against the conversation and the "
            "policy. Output the JSON verdict."
        )
        verifier_messages = [
            SystemMessage(role="system", content=VERIFIER_SYSTEM),
            UserMessage(role="user", content=user_content),
        ]
        try:
            response = generate(
                model=self.llm,
                tools=None,
                messages=verifier_messages,
                call_name="write_discipline_check",
                **self.llm_args,
            )
        except Exception:
            # Fail-safe: never block a turn because the verifier errored.
            return []
        return _parse_verdict(getattr(response, "content", None))

    @staticmethod
    def _build_note(flagged: list[str]) -> str:
        bullets = "\n".join(f"- {item}" for item in flagged)
        return (
            "A compliance review flagged the following write action(s) in your "
            "previous attempt as NOT requested by the customer and NOT required "
            "by the policy:\n"
            f"{bullets}\n"
            "Redo this turn. Perform ONLY the write actions the customer "
            "explicitly asked for or that the policy explicitly requires. Do "
            "not apply credits, waive fees, open or close extra accounts, or "
            "file extra disputes on your own initiative. Keep every action the "
            "customer did request. If a flagged action is in fact something "
            "the customer explicitly requested, you may keep it."
        )

    def generate_next_message(self, message, state: LLMAgentState):
        """Respond to a user or tool message, suppressing unsolicited writes."""
        # Mirror LLMAgent: incoming message(s) are appended to history.
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        guidance: Optional[str] = None
        assistant_message: Optional[AssistantMessage] = None
        for attempt in range(MAX_GENERATIONS):
            assistant_message = self._generate(state, guidance)
            writes = write_guard.extract_writes(assistant_message)
            if not writes:
                break
            if attempt == MAX_GENERATIONS - 1:
                # Already used the disciplined retry; accept what we have.
                break
            flagged = self._verify_writes(state, writes)
            if not flagged:
                break
            guidance = self._build_note(flagged)

        state.messages.append(assistant_message)
        return assistant_message, state


def build_agent(tools, domain_policy, **kwargs):
    return WriteDisciplineAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
