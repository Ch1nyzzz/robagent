"""tau2 candidate mh_tau2_iter3_robust_card_spec_channel.

Hypothesis
----------
Credit-card recommendation tasks fail because the agent's only knowledge-base
retrieval channel (``KB_search``, BM25) cannot reliably surface the per-product
"Getting Started / Eligibility" spec sheets for the bank's card products. A
query naming one card collides on shared vocabulary with every other card's
documents and the agent never assembles a complete view of the line-up; it
then either recommends a card that does not meet the customer's requirements,
or refuses / transfers because it could not gather the necessary information.

Mechanism (banking sims the frontier still fails)
-------------------------------------------------
Tasks whose gold action is ``apply_for_credit_card`` with a card_type the
agent cannot match against customer requirements:

  * task_003 — customer needs no-foreign-fee, purchase protection, $100k+
    credit limit; gold card is "Silver Rewards Card". Agent loops on KB_search
    queries for personal-card specs that BM25 keeps mis-ranking to business
    cards and rewards-representation procedure docs.
  * task_024 — gold card is "Business Bronze Rewards Card". Agent never
    surfaces the business-bronze spec and recommends "Business Silver".
  * task_025 — gold card is "Business Platinum Rewards Card". Agent transfers
    to a human rather than recommending.
  * task_048 — closure-with-replacement workflow whose gold action set
    includes ``apply_for_credit_card``; agent never reaches that step in part
    because it could not anchor its recommendation against the product
    catalog.

Per-task analysis confirms ``required_documents`` for these tasks is dominated
by the per-product ``_001.json`` spec sheets across consumer and business
cards.

Decomposition
-------------
  * Deterministic code (``card_spec_catalog``): walk the KB documents
    directory, enumerate every credit-card product prefix
    (``doc_(business_)?credit_cards_<product>_``, excluding the procedure /
    feature groups ``credit_cards_(general)``, ``credit_card_account_logistics``,
    ``credit_card_replacements``, ``virtual_card_management``), and load each
    product's ``_001.json`` Getting Started / Eligibility spec sheet. Compile
    them into one catalog block grouped consumer vs. business.
  * LLM judgement: read the catalog, weigh the customer's stated requirements
    against the per-product specs, and decide whether/which card to recommend.

Why this captures stable structure (not training-set induction)
---------------------------------------------------------------
The selection rule is keyed to two KB-infrastructure facts that are
independent of which simulations are failing today:

  1. Document ids partition by product prefix. The set of product prefixes is
     enumerable from disk; a small explicit set of non-product prefixes
     (``(general)``, ``account_logistics``, ``replacements``, the
     ``virtual_card_management`` feature group) excludes cross-cutting
     procedure groups.
  2. The ``_001.json`` suffix is the KB's authoring convention for the first /
     Getting Started doc of each product — a deterministic property of the KB
     layout, not a property of any task.

On a freshly authored KB written under the same conventions, the rule would
select the same class of documents with no reference to any specific failure.

The catalog never overrides the LLM's output; it supplies content the BM25
channel cannot reliably surface. The LLM still chooses the card. A mis-fire
(e.g. a future "card" doc with surprising content under one of these prefixes)
only adds reference text and cannot make a wrong tool call.

build_agent(tools, domain_policy, **kwargs) -> HalfDuplexAgent
"""
from __future__ import annotations

from tau2.agent.llm_agent import (
    AGENT_INSTRUCTION,
    SYSTEM_PROMPT,
    LLMAgent,
)

from . import card_spec_catalog


class CardSpecChannelAgent(LLMAgent):
    """LLMAgent that always injects the KB's per-product credit-card spec
    sheets into the system prompt."""

    def __init__(self, tools, domain_policy, llm, llm_args=None):
        super().__init__(
            tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args
        )
        try:
            self._catalog = card_spec_catalog.build_catalog()
        except Exception:
            self._catalog = ""

    @property
    def system_prompt(self) -> str:
        base = SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        )
        if not self._catalog:
            return base
        return base + "\n\n" + self._catalog


def build_agent(tools, domain_policy, **kwargs):
    """Return a HalfDuplexAgent that augments LLMAgent's system prompt with
    the KB's per-product credit-card spec sheets."""
    return CardSpecChannelAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
