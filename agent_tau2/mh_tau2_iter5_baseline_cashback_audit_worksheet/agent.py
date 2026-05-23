"""tau2 candidate mh_tau2_iter5_baseline_cashback_audit_worksheet.

Hypothesis
----------
The cash-back-audit tasks fail because the agent decides the WRONG SET of
transactions to dispute / correct. The reward for these tasks is the database
hash, so the agent must act on exactly the genuinely miscalculated
transactions: disputing a correct transaction breaks the hash just as surely as
missing a wrong one. The agent eyeballs the transaction list in prose and gets
the set wrong — it applies a promotional multiplier to transactions outside the
promotion window and judges rounding-level (~1 point) differences
inconsistently. Giving the agent a clean, deterministic transaction worksheet
and a precise audit protocol — and making sure it has retrieved the earning-rate
and promotion knowledge-base documents before it acts — will let it act on the
exact gold set and raise train-30 reward.

Mechanism observed (iter4 banking simulations)
----------------------------------------------
All three are the same customer's cash-back audit, scored on the database hash,
and all three fail because the agent's dispute set != the gold set:
  * task_020 — gold disputes 4 transactions; the agent disputes 6. It applied
    the Business Silver "double cash back" promo to two transactions dated
    outside the promo window and flagged them as shortchanged.
  * task_026 — gold disputes/corrects 4 transactions; the agent disputes and
    updates 14, re-touching transactions whose recorded value was already
    correct.
  * task_027 — gold disputes 4 transactions; the agent disputes only 3 (it
    skips one genuinely wrong transaction) and adds extra rewards updates.
In every case db_check is false purely because the set of write actions does
not equal gold; the agent's own prose worksheet even marks equal ~1-point
differences inconsistently (one as wrong, one as correct).

Change
------
1. The system prompt gains a general, domain-agnostic <cash_back_audit_protocol>:
   identify every card type, retrieve each card's earning-rate policy and any
   promotion, apply a promotion only inside its eligibility window (checking each
   transaction date), compute expected = amount * rate, treat ~1-point
   differences as rounding (not errors), and act on exactly the materially
   miscalculated set. No identifiers, rates, or task hints are hardcoded.
2. A deterministic helper (cashback_guard.CashBackAuditGuard) parses the raw
   transaction records out of tool results and re-presents them as a clean
   worksheet, and tracks whether earning-rate / promotion KB documents have been
   retrieved. The first time the agent submits a cash-back dispute or rewards
   update, that turn is regenerated once with the worksheet, the protocol
   reminder, and — if the relevant KB documents were never retrieved — a note to
   look them up first. From then on the worksheet is attached to every turn as
   reference data, so each per-transaction decision is made from structured
   facts.

The worksheet is only reformatted facts, so it can never inject a wrong verdict;
the steering nudge fires once and never blocks an action, so a legitimate
dispute can never be permanently suppressed. The LLM still decides which
transactions are wrong.
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

from agent_tau2.mh_tau2_iter5_baseline_cashback_audit_worksheet.cashback_guard import (
    CashBackAuditGuard,
)

# General, domain-agnostic guidance appended to the system prompt.
CASH_BACK_AUDIT_PROTOCOL = """
<cash_back_audit_protocol>
Some requests ask you to review a customer's credit-card transactions for
incorrect cash-back / rewards and to dispute or correct the ones that are wrong.
On these tasks you must act on EXACTLY the set of transactions that are
genuinely miscalculated: disputing or adjusting a transaction that was actually
correct is just as wrong as missing one.

Work methodically before you submit any dispute or rewards adjustment:
1. Identify every card type that appears in the customer's transactions.
2. For EACH card type, retrieve from the knowledge base its earning-rate policy
   (base rate and bonus categories) AND whether any promotional offer (for
   example a temporary rate multiplier) applies to that card. Search the
   knowledge base for this; do not rely on assumption.
3. A promotion changes the rate only for transactions whose transaction_date
   falls inside the promotion's eligibility window. If a promotion is defined
   relative to account opening (for example "first 6 months"), compute that
   window from the card's open date and check each transaction's date against
   it. Never apply a promotional rate to a transaction dated outside the window.
4. For each transaction compute the expected reward:
   expected_points = transaction_amount * applicable_rate_percent
   (1% cash back on $1 = 1 point).
5. Compare the expected reward to the recorded rewards_earned. A difference of
   about 1 point is ordinary rounding and is NOT an error — treat that
   transaction as correct. Flag a transaction as miscalculated only when the
   recorded value differs from the expected value by a clearly material amount.
6. Dispute or adjust exactly the transactions you have shown to be materially
   miscalculated — no correct transactions, and no omissions.
</cash_back_audit_protocol>
""".strip()

# Maximum number of generations for the first cash-back write turn
# (1 original + 1 steered retry).
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


class CashBackAuditAgent(LLMAgent):
    """LLMAgent that audits cash-back from a deterministic transaction worksheet."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._guard = CashBackAuditGuard()

    @property
    def system_prompt(self) -> str:
        base = SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        )
        return base + "\n" + CASH_BACK_AUDIT_PROTOCOL

    def get_init_state(self, message_history=None) -> LLMAgentState:
        self._guard.reset()
        state = super().get_init_state(message_history)
        # Replay any pre-seeded tool results so the guard's view is current.
        for msg in state.messages:
            if isinstance(msg, ToolMessage):
                self._guard.observe_tool_result(getattr(msg, "content", None))
        return state

    def _generate(self, state: LLMAgentState, note: Optional[str]) -> AssistantMessage:
        """Run one LLM generation, optionally with a transient turn note."""
        system_messages = state.system_messages
        if note:
            base = state.system_messages[0].content
            system_messages = [
                SystemMessage(
                    role="system",
                    content=f"{base}\n\n<turn_note>\n{note}\n</turn_note>",
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
        """Respond to a user or tool message, auditing cash-back from a worksheet."""
        # Mirror LLMAgent: incoming message(s) are appended to history.
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        # Update the guard from any tool results just received.
        for text in _tool_result_texts(message):
            self._guard.observe_tool_result(text)

        # Once an audit is under way, attach the worksheet as reference data.
        assistant_message = self._generate(state, self._guard.context_note())

        # The first time the agent submits a cash-back write, regenerate that
        # turn once with the worksheet + protocol + (if needed) a KB-lookup note.
        if self._guard.is_first_cashback_turn(assistant_message):
            self._guard.enter_audit_mode()
            note = self._guard.first_action_note()
            for _ in range(MAX_GENERATIONS - 1):
                assistant_message = self._generate(state, note)

        state.messages.append(assistant_message)
        return assistant_message, state


def build_agent(tools, domain_policy, **kwargs):
    return CashBackAuditAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
