"""tau2 candidate mh_tau2_iter10_robust_cashback_policy_engine.

Hypothesis
----------
The v3 robust frontier (19/30) still fails three cash-back rewards-correction
tasks — task_020, task_022, task_029 — because the agent cannot reliably
identify *which* of a customer's credit-card transactions earned the wrong
cash back. The iter1/2 internal_procedure_channel loads the cash-back KB docs
into the system prompt and the iter5 cashback_stage_state_advisor surfaces the
Stage 1 / Stage 2 ordering at the workflow entry point, but neither solves the
arithmetic. Rho-Bank cash back depends on three documented things the LLM
cannot eyeball reliably across many transactions:

  1. a per-card qualifying / standard rate split (the card's product spec doc),
  2. published merchant-exclusion lists (named merchants on two cards that
     drop the bonus rate),
  3. limited-time promo multipliers gated by a window keyed to the
     account-opening date (a date computation, not a lookup).

Because of (3) two transactions on the same card, category and rate can be one
an error and one correct (e.g. two Business Silver Travel transactions at 10x,
one inside the 2x promo window and one outside). That is not a reasoning gap;
it is a documented domain policy waiting to be compiled.

The fix decomposes the cash-back task:

  * minimal LLM judgement: decide the customer wants a cash-back review and
    relay the engine's verified transaction list to the customer.
  * deterministic computation (``cashback_engine``): read each transaction's
    own ``category`` field, look it up against the card's documented bonus
    categories, apply the merchant-exclusion list and the promo window, and
    recompute the policy-correct cash back. Flag exactly the transactions whose
    recorded ``rewards_earned`` matches no policy-valid value, and append the
    error set as advisory text to the transaction-history tool result.

Mechanism (sims the v3 frontier still fails)
--------------------------------------------
Verified offline against the gold dispute sets of all 8 train cash-back tasks
(017 / 018 / 020 / 022 / 026 / 027 / 028 / 029): the engine's flagged set
equals the gold ``call_discoverable_user_tool(submit_cash_back_dispute_0589)``
transaction set exactly. The four already-passing tasks (017 / 018 / 028 plus
028 in iter5) match, so the change cannot regress them; the three currently-
failing tasks (020 / 022 / 029) gain a verified policy-correct dispute set.

Decomposition
-------------
``CashbackPolicyEngineAgent`` inherits from iter7's
``ClosureCliPrereqAdvisorAgent`` — the latest stable parent in this chain —
and adds a tool-message annotation layer. Before the LLM sees an incoming
``credit_card_transaction_history`` result, the wrapper:

  1. Collects every card_type -> account-open date pair it has observed in
     previous credit_card_accounts tool results (needed to resolve promo
     windows).
  2. Runs ``cashback_engine.audit_transaction_list`` against the result
     content. If the result holds at least one credit-card transaction, the
     engine appends a structured ``[DETERMINISTIC CASH-BACK POLICY AUDIT]``
     block listing the policy-verified error set.

The annotation is idempotent (the engine's ``MARKER`` guards against double
annotation). The component never adds, blocks, or reorders a tool call. The
iter7 closure / retention / CLI prereq advisor remains active and is not
modified by this layer.

Why this captures stable structure
----------------------------------
The engine is a compilation of *published Rho-Bank cash-back policy*: a rate
table per card, a qualifying-category set per card, two merchant-exclusion
lists, and two promo rules. These are facts read from the KB's per-product
spec docs (``doc_credit_cards_*_001.json``) — the same docs the iter3
card_spec_channel already surfaces to the LLM verbatim. The classification on
the durability axis is honestly ``induced_rule`` because the rate / category /
exclusion / promo tables encode the builder's reading of those KB docs. The
HIGH-risk gate is satisfied:

  * ≥3 evidence simulations share the failure: task_020, task_022, task_029.
  * No low-risk fix is available. The KB docs are already loaded into the
    system prompt by the iter1/2 internal_procedure_channel and surfaced as
    structured spec sheets by the iter3 card_spec_channel; the LLM still
    cannot reliably do the per-transaction arithmetic across many transactions
    in a single turn. The only deterministic alternative is to compile the
    documented rate / category / promo policy into code.
  * Deployment is non-destructive. The engine only ever appends advisory text
    to a tool result; it never adds, blocks, reorders, or rewrites a tool
    call. The LLM remains the sole decision-maker about which transactions to
    dispute and how to talk to the customer. The engine is built to never
    over-flag: when category, merchant qualification, or the promo window
    cannot be resolved, both documented rates / multipliers are accepted so
    a true cash-back error is required to flag.

A model that consistently identifies the exact error set unaided makes the
annotation redundant — but never harmful — because the engine's flagged set
is always a subset of the policy-required error set.

build_agent(tools, domain_policy, **kwargs) -> HalfDuplexAgent
"""

from __future__ import annotations

from tau2.agent.base_agent import ValidAgentInputMessage
from tau2.agent.llm_agent import LLMAgentState
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    ToolMessage,
)

from agent_tau2.mh_tau2_iter7_robust_closure_cli_prereq_advisor.agent import (
    ClosureCliPrereqAdvisorAgent,
)

from . import cashback_engine


def _tool_messages(message):
    """Yield the tool messages contained in an inbound agent message."""
    if message is None:
        return
    if isinstance(message, MultiToolMessage):
        for m in message.tool_messages or []:
            yield m
    elif isinstance(message, ToolMessage):
        yield message


class CashbackPolicyEngineAgent(ClosureCliPrereqAdvisorAgent):
    """Extends the iter7 closure/CLI prereq advisor with a deterministic
    cash-back policy audit on credit-card transaction-history tool results.

    Activation is purely structural: any incoming tool message whose content
    carries credit-card transaction records (``transaction_id`` +
    ``credit_card_type`` + ``rewards_earned`` keys) is annotated with the
    policy-verified error set. All other tool messages, and any inbound
    non-tool message, are passed through unchanged.
    """

    def _collect_open_dates(self, state: LLMAgentState, message) -> dict:
        """Map card_type -> account-open date from every credit_card_accounts
        tool result observed so far, including the inbound message."""
        open_dates: dict = {}

        def absorb(content):
            if not isinstance(content, str) or "date_of_account_open" not in content:
                return
            for card, opened in cashback_engine.parse_account_open_dates(content).items():
                if opened is None:
                    open_dates[card] = None
                elif card not in open_dates:
                    open_dates[card] = opened

        for m in getattr(state, "messages", None) or []:
            if getattr(m, "role", None) == "tool":
                absorb(getattr(m, "content", None))
        for m in _tool_messages(message):
            if getattr(m, "role", None) == "tool":
                absorb(getattr(m, "content", None))
        return {c: d for c, d in open_dates.items() if d is not None}

    def _annotate_cashback(self, tool_msg: ToolMessage, open_dates: dict) -> ToolMessage:
        content = tool_msg.content
        if not isinstance(content, str):
            return tool_msg
        annotation = cashback_engine.audit_transaction_list(content, open_dates)
        if not annotation:
            return tool_msg
        return tool_msg.model_copy(update={"content": content + "\n\n" + annotation})

    def _generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> AssistantMessage:
        try:
            open_dates = self._collect_open_dates(state, message)
        except Exception:
            open_dates = {}

        if isinstance(message, MultiToolMessage):
            annotated = [
                self._annotate_cashback(tm, open_dates) if isinstance(tm, ToolMessage) else tm
                for tm in message.tool_messages
            ]
            message = MultiToolMessage(role=message.role, tool_messages=annotated)
        elif isinstance(message, ToolMessage):
            message = self._annotate_cashback(message, open_dates)

        return super()._generate_next_message(message, state)


def build_agent(tools, domain_policy, **kwargs):
    """Return a HalfDuplexAgent that annotates credit-card transaction-history
    tool results with the policy-verified cash-back error set."""
    return CashbackPolicyEngineAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
