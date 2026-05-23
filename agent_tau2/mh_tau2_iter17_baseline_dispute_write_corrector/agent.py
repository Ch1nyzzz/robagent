"""tau2 candidate mh_tau2_iter17_baseline_dispute_write_corrector.

Hypothesis
----------
On credit-card transaction-dispute tasks the agent calls every required tool
but loses the DB-hash reward because individual ARGUMENT FIELDS of the
``file_credit_card_transaction_dispute_4829`` (and the accompanying
``order_replacement_credit_card_7291``) write call diverge from gold. Each field
of a WRITE call is stored verbatim into a hashed table, so one wrong field
zeroes the reward even when the correct tools were called. Three dispute fields
are in fact deterministic, and the agent gets them wrong by guessing or by
mis-applying policy:

* ``card_last_4_digits`` — derived by the bank from the account id with a fixed
  hash formula; the agent guesses it (task_053: filed "8231", gold "2791").
* ``eligible_for_provisional_credit`` — fixed by the documented Provisional
  Credit Eligibility policy; the agent sets it ``true`` for a non-fraud dispute
  filed without contacting the merchant (task_038 dispute #3: duplicate_charge,
  contacted_merchant=false — agent ``true``, gold ``false``).
* ``card_action`` — a card has one fate; once a fraud dispute proves the card is
  compromised, every dispute for that card must be ``cancel_and_reissue``
  (task_038 dispute #2: agent ``keep_active``, gold ``cancel_and_reissue``).

Replacing these guesses with the bank's own formulas and policy rules — and
expediting a fraud/lost/stolen replacement order (task_038: agent
``expedited_shipping=false``, gold ``true``) — makes the dispute write set match
gold and raises train-30 reward.

Mechanism observed (banking sims the frontier still fails)
----------------------------------------------------------
  * task_038 — three ``file_credit_card_transaction_dispute_4829`` calls and one
    ``order_replacement_credit_card_7291``. The right tools are all called, but
    three argument fields diverge from gold: dispute #2 ``card_action``, dispute
    #3 ``eligible_for_provisional_credit``, and the order's
    ``expedited_shipping``. All three are deterministically correctable.
  * task_053 — every gold tool is called, but the dispute's ``card_last_4_digits``
    is a number the customer recited from memory ("8231") instead of the real
    card value ("2791"), which is computable from the account id.

Decomposition
-------------
  * deterministic code (``dispute_corrector``): after each turn is generated,
    rewrite the arguments of any dispute / replacement WRITE call so the
    deterministic fields equal the bank-formula / documented-policy value. All
    rewrites are one-directional and conservative — ``eligible`` only ever
    flips true->false, ``card_action`` is only ever forced to
    ``cancel_and_reissue`` when a fraud dispute proves it, ``card_last_4_digits``
    is recomputed from the bank's published formula. When a field cannot be
    resolved the call is left exactly as the LLM produced it.
  * LLM judgement: everything else — gathering dispute details, deciding which
    transactions to dispute, conducting the conversation.

No customer names, ids, card ids, or per-task branching: the corrector keys
only on tool names, argument values, and the bank's own deterministic formulas.

build_agent(tools, domain_policy, **kwargs) -> HalfDuplexAgent
"""
from __future__ import annotations

from tau2.agent.llm_agent import (
    AGENT_INSTRUCTION,
    SYSTEM_PROMPT,
    LLMAgent,
    LLMAgentState,
)

from . import dispute_corrector

# General, domain-agnostic guidance. It restates the deterministic rules the
# corrector enforces so the LLM tends to produce them itself; the corrector is
# the guarantee, the prompt only reduces how often it must intervene.
DISPUTE_GUIDANCE = """
<dispute_write_discipline>
When you file a credit-card transaction dispute or order a replacement card,
every argument you pass is recorded verbatim. Fill these fields by rule, not by
guessing:

- card_last_4_digits: this is an exact value tied to the card account. Obtain it
  from the dedicated lookup tool's result — never type a number the customer
  recited from memory and never invent one.

- eligible_for_provisional_credit: follow the bank's Provisional Credit
  Eligibility policy. A dispute is NOT eligible for provisional credit when the
  reason is anything other than an unauthorized/fraudulent charge and the
  customer has not contacted the merchant, nor when the reason is one the policy
  does not cover. When in doubt, do not claim eligibility.

- card_action: a card has a single fate. If any charge on a card is an
  unauthorized/fraudulent charge, that card number is compromised and the card
  must be cancelled and reissued — so EVERY dispute you file for that same card
  must use card_action "cancel_and_reissue", not "keep_active".

- A replacement card ordered because of fraud, loss, or theft leaves the
  customer with no usable card; order it with expedited shipping.

Make sure all disputes filed for one card in a single conversation are
internally consistent with each other.
</dispute_write_discipline>
""".strip()


class DisputeWriteCorrectorAgent(LLMAgent):
    """LLMAgent that deterministically repairs the argument fields of
    transaction-dispute and replacement-card WRITE calls before they run."""

    @property
    def system_prompt(self) -> str:
        base = SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        )
        return base + "\n" + DISPUTE_GUIDANCE

    def generate_next_message(self, message, state: LLMAgentState):
        # state.messages holds the history (incl. the inbound message/tool
        # results) but NOT yet this turn's assistant message.
        assistant_message = self._generate_next_message(message, state)
        try:
            dispute_corrector.correct_assistant_message(
                assistant_message, state.messages
            )
        except Exception:
            # Fail-safe: never let the corrector break a turn.
            pass
        state.messages.append(assistant_message)
        return assistant_message, state


def build_agent(tools, domain_policy, **kwargs):
    """Return a HalfDuplexAgent that corrects deterministic dispute/replacement
    WRITE-call arguments after each turn is generated."""
    return DisputeWriteCorrectorAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
