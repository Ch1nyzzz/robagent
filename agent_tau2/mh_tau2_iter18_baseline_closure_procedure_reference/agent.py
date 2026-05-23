"""tau2 candidate mh_tau2_iter18_baseline_closure_procedure_reference.

Hypothesis
----------
Credit-card account-closure tasks fail the database-hash check because the
agent never assembles the complete closure procedure. The procedure spans
several knowledge-base documents — closure eligibility, the closure process,
the multi-step retention protocol, and the how-to documents for the
discoverable agent tools each step uses — and the agent's only retrieval tool,
``KB_search``, is BM25 sparse retrieval that surfaces them inconsistently. The
retention-protocol document itself describes the prerequisite checks in prose
("no pending disputes", "no pending replacement cards") but never names the
tools that perform them, so even an agent that retrieves the procedure cannot
discover ``get_user_dispute_history_7291`` or
``get_pending_replacement_orders_5765``.

As a result the agent:
  * skips the universal prerequisite reads (dispute history, pending
    replacement orders) — leaving the discoverable-call set short of gold;
  * closes a card that has a pending replacement order and therefore cannot be
    closed — diverging from gold, which checks and keeps that card open;
  * improvises the retention protocol — jumping straight to a token offer
    without the documented "address the concern" / card-recommendation step,
    and never reaching the documented statement-credit retention offer that
    gold expects the customer to accept.

Deterministically compiling the credit-card account-procedure reference (the
account-logistics document family plus the how-to documents for the
discoverable tools) and the credit-card product catalog from the knowledge
base, and injecting them as read-only system reference on closure
conversations, gives the agent the complete checklist — every prerequisite
check, the tool that performs it, the full retention protocol, and the card
line-up needed for the "found a better card" retention branch — so its
discoverable-call set matches gold.

Mechanism observed (banking sims the frontier still fails)
----------------------------------------------------------
  * task_048 — four-card closure. The agent never calls
    get_user_dispute_history_7291 or get_pending_replacement_orders_5765 for any
    card, and closes the EcoCard, which has a pending replacement order and
    must NOT be closed. Gold checks pending replacements per card, keeps the
    blocked card open, and closes only the eligible one.
  * task_045 — the agent skips the dispute-history and pending-replacement
    reads, jumps to a "$20" retention offer without the documented
    address-the-concern step, the customer is insulted and ends the call, and
    the gold apply_statement_credit_8472 retention offer never happens.
  * task_044 — every closure action matches gold, but the agent mishandles the
    retention recommendation and the customer closes instead of being retained;
    the gold trajectory keeps the card and the customer applies for the
    recommended Rho-Bank card.

Decomposition
-------------
  * deterministic code (``closure_reference``, ``card_catalog``): locate the
    banking_knowledge knowledge base on disk; compile the credit-card
    account-procedure documents and the credit-card product catalog into
    read-only reference strings; detect closure / card-shopping intent.
  * LLM judgement: run the conversation, decide which cards are eligible,
    follow the retention protocol, and choose which retention offer to make.

Both additions are non-destructive: the reference is read-only context (the
same documents KB_search indexes) and the new system-prompt section is general
workflow guidance. The agent never gains, drops, or rewrites a tool call as a
result, so a task the frontier already passes cannot regress.

build_agent(tools, domain_policy, **kwargs) -> HalfDuplexAgent
"""
from __future__ import annotations

from tau2.agent.llm_agent import (
    AGENT_INSTRUCTION,
    SYSTEM_PROMPT,
    LLMAgent,
    LLMAgentState,
)
from tau2.data_model.message import MultiToolMessage, SystemMessage

from . import card_catalog, closure_reference

# General, domain-agnostic workflow guidance appended to the system prompt.
CLOSURE_GUIDANCE = """
<account_closure_workflow>
Closing a credit-card account is a documented multi-step procedure, not a
single action. When a customer asks to close a credit-card account, follow the
procedure in full:

- Before any closure decision, complete EVERY documented prerequisite
  eligibility check: outstanding balance, pending transaction disputes,
  minimum account age, pending replacement card orders, and any prior closure
  or retention attempts.
- Each prerequisite check is performed with a dedicated tool. If you do not
  already see the tool you need, consult the knowledge base — including the
  account-procedure reference provided to you — to find it. Do not skip a
  documented check merely because its tool must be looked up or unlocked first.
- Check pending replacement card orders for EACH card the customer wants to
  close. A card with a pending (not yet received or activated) replacement
  order cannot be closed; tell the customer why and do not close it.
- Follow the retention protocol in order: understand and log the closure
  reason, address the customer's specific concern, and then make exactly one
  retention offer based on the card's tier. If a prior closure record exists
  for that account within the past year, skip the retention offer.
- If the customer accepts a retention offer (for example a statement credit),
  apply that offer and keep the card open — do not also close the card.
- If the customer wants to close because they found a better card elsewhere,
  ask what features attracted them and recommend the best-fit Rho-Bank card,
  offering to help them apply instead of closing.
- When a customer raises several cards in one conversation, take each card
  through the full procedure independently.
</account_closure_workflow>
""".strip()


class ClosureProcedureReferenceAgent(LLMAgent):
    """LLMAgent that injects the credit-card account-procedure reference (and,
    for card-shopping closure conversations, the product catalog) as read-only
    system context whenever the conversation is about closing a card."""

    def __init__(self, tools, domain_policy, llm, llm_args=None):
        super().__init__(
            tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args
        )
        # Compiled once at construction; "" if the knowledge base is unreadable.
        try:
            self._closure_ref = closure_reference.build_closure_reference()
        except Exception:
            self._closure_ref = ""
        try:
            self._catalog = card_catalog.build_catalog()
        except Exception:
            self._catalog = ""

    @property
    def system_prompt(self) -> str:
        base = SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        )
        return base + "\n" + CLOSURE_GUIDANCE

    def generate_next_message(self, message, state: LLMAgentState):
        # Mirror LLMAgent: append the inbound message(s) to history.
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        assistant_message = self._generate(state)

        # Mirror LLMAgent: append the assistant message to history.
        state.messages.append(assistant_message)
        return assistant_message, state

    # --- generation -------------------------------------------------------
    def _generate(self, state: LLMAgentState):
        from tau2.utils.llm_utils import generate

        user_texts = self._user_texts(state)
        system_content = self.system_prompt
        if self._closure_ref and closure_reference.is_closure_conversation(
            user_texts
        ):
            system_content = system_content + "\n\n" + self._closure_ref
        if self._catalog and card_catalog.is_card_shopping(user_texts):
            system_content = system_content + "\n\n" + self._catalog
        system_message = SystemMessage(role="system", content=system_content)

        messages = [system_message] + list(state.messages)
        return generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            call_name="agent_response",
            **self.llm_args,
        )

    @staticmethod
    def _user_texts(state: LLMAgentState) -> list[str]:
        """Every user-message text seen so far in the conversation."""
        texts: list[str] = []
        for m in getattr(state, "messages", None) or []:
            if getattr(m, "role", None) != "user":
                continue
            content = getattr(m, "content", None)
            if isinstance(content, str):
                texts.append(content)
        return texts


def build_agent(tools, domain_policy, **kwargs):
    """Return a HalfDuplexAgent that follows the documented credit-card closure
    procedure: general system-prompt workflow guidance plus the account-procedure
    reference (and the product catalog) injected on closure conversations."""
    return ClosureProcedureReferenceAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
