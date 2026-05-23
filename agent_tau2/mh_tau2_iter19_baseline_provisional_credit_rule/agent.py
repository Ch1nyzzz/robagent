"""tau2 candidate mh_tau2_iter19_baseline_provisional_credit_rule.

Hypothesis
----------
Credit-card transaction-dispute tasks lose the database-hash reward because the
agent sets the ``eligible_for_provisional_credit`` field of
``file_credit_card_transaction_dispute_4829`` wrong. That field is a strict,
policy-derived boolean — written verbatim into a hashed table — and the agent
treats it as a courtesy or judges it loosely. The recurring concrete error is a
NON-FRAUD dispute (most visibly a ``duplicate_charge``) marked eligible even
though the customer never contacted the merchant: the "Provisional Credit
Eligibility Guidelines" article requires merchant contact for every reason
other than ``unauthorized_fraudulent_charge``.

Mechanism observed (banking sims the frontier still fails)
----------------------------------------------------------
  * task_038 — the agent files three disputes. Gold expects
    eligible_for_provisional_credit = [true (fraud), false (not-as-described),
    false (duplicate_charge, merchant not contacted)]. The iter18 frontier run
    produced [true, false, TRUE]: it marked the duplicate_charge dispute
    eligible, missing that a duplicate_charge is a non-fraud dispute and the
    customer set contacted_merchant=false. That single wrong field breaks the
    DB hash. The earlier iter16 run made the same duplicate_charge error.
  * task_053 — also files a credit-card transaction dispute; the same
    policy-derived field must be exact for the DB hash. Its dispute is
    goods_services_not_received with contacted_merchant=true, which is
    genuinely eligible, so the corrector below correctly leaves it untouched.

Fix
---
Two additions, both targeting exactly this field:

  * deterministic code (``provisional_credit``): after every turn is generated,
    inspect each ``file_credit_card_transaction_dispute`` call and flip
    ``eligible_for_provisional_credit`` from true to false when the call's OWN
    arguments prove the dispute ineligible — the dispute reason is not a
    provisional-eligible category, or it is a non-fraud reason filed with
    contacted_merchant=false. The corrector is strictly one-directional
    (true -> false only) and uses only criteria fully decidable from the call
    itself, so it can never turn a correct call into a wrong one and never
    touches any other field or tool.

  * a focused ``<credit_card_dispute_provisional_credit>`` system-prompt section
    restating the eligibility rule precisely, so the agent gets the field right
    in the first place and the corrector is only a safety net. It also notes
    the rule can be applied from the dispute details on hand, discouraging an
    unnecessary prior-dispute-history lookup.

Both additions only ever affect file_credit_card_transaction_dispute calls; no
frontier-passing train task files such a dispute, so a passing task cannot
regress. No customer names, ids, transaction ids, or per-task branching — the
corrector keys solely on the tool name and the policy rule.

build_agent(tools, domain_policy, **kwargs) -> HalfDuplexAgent
"""
from __future__ import annotations

from tau2.agent.llm_agent import (
    AGENT_INSTRUCTION,
    SYSTEM_PROMPT,
    LLMAgent,
    LLMAgentState,
)

from . import provisional_credit

# General, domain-derived guidance. This restates the knowledge base's own
# "Provisional Credit Eligibility Guidelines" article; it is not task-specific.
DISPUTE_GUIDANCE = """
<credit_card_dispute_provisional_credit>
When filing a credit-card transaction dispute with
file_credit_card_transaction_dispute_4829, the eligible_for_provisional_credit
flag is a strict policy decision, not a courtesy. It is recorded verbatim, so a
wrong value is a policy error.

A dispute is ELIGIBLE for provisional credit only when EVERY one of these
conditions holds:

1. The credit-card account has been open for at least 60 days.
2. The dispute reason is one of: 'unauthorized_fraudulent_charge',
   'duplicate_charge', or 'goods_services_not_received'. Every other reason —
   'incorrect_amount', 'goods_services_not_as_described',
   'canceled_subscription_still_charging', 'refund_never_processed' — is NOT
   eligible.
3. The transaction amount is at least $25.00 and within the card tier's
   provisional-credit maximum.
4. The customer has filed no more than 2 disputes in the past 12 months.
5. For any NON-FRAUD dispute — any reason other than
   'unauthorized_fraudulent_charge', INCLUDING 'duplicate_charge' — the
   customer must have contacted the merchant first. If contacted_merchant is
   false for a non-fraud dispute, it is NOT eligible.
6. A 'goods_services_not_received' dispute is eligible only when the purchase
   was made more than 30 days before today.

If any single condition fails, set eligible_for_provisional_credit to false.
Decide each condition from the dispute details and the customer's account
record you already have; you do not need to retrieve the customer's prior
dispute history with a separate tool just to apply this rule.
</credit_card_dispute_provisional_credit>
""".strip()


class ProvisionalCreditRuleAgent(LLMAgent):
    """LLMAgent that adds the provisional-credit policy to the system prompt and
    deterministically corrects an over-stated eligible_for_provisional_credit
    flag on credit-card transaction-dispute calls."""

    @property
    def system_prompt(self) -> str:
        base = SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        )
        return base + "\n" + DISPUTE_GUIDANCE

    def generate_next_message(self, message, state: LLMAgentState):
        # Mirror LLMAgent.generate_next_message, inserting the deterministic
        # corrector before the assistant message is committed to history.
        assistant_message = self._generate_next_message(message, state)
        self._apply_corrector(assistant_message)
        state.messages.append(assistant_message)
        return assistant_message, state

    @staticmethod
    def _apply_corrector(assistant_message) -> None:
        """Correct eligible_for_provisional_credit on any dispute call in the
        just-generated turn. Fail-safe: never raises."""
        try:
            tool_calls = getattr(assistant_message, "tool_calls", None)
            if not tool_calls:
                return
            for tool_call in tool_calls:
                provisional_credit.correct_tool_call(tool_call)
        except Exception:
            pass


def build_agent(tools, domain_policy, **kwargs):
    """Return a HalfDuplexAgent that enforces the provisional-credit eligibility
    rule on credit-card transaction-dispute writes."""
    return ProvisionalCreditRuleAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
