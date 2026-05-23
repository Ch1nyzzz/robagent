"""tau2 candidate mh_tau2_iter15_robust_card_spec_rate_channel.

Hypothesis
----------
Train task_003 and task_048 still fail on the v3 robust frontier (24/30)
at the credit-card recommendation step because the gold-correct product —
"Silver Rewards Card" (consumer, $0 annual fee, 4% on travel & software) —
is selected on the basis of its bonus-category cash-back rate (4% on
travel & software), which lives in ``doc_credit_cards_silver_rewards_card_002``.
That doc is NOT in the agent's reach: bm25 ``KB_search`` doesn't rank it
against natural-language recommendation queries, and the iter13-rooted v3
chain never inherited iter3's per-product spec channel (iter7 forked off
``LLMAgent`` directly; the system prompt at iter13 is identical to v0's).
In task_003 the agent recommended Gold (2.5% all-categories); in task_048
the agent recommended Business Silver (10% travel/software but $122.50
annual fee, business-eligibility required). Both failures stem from the
LLM not having the consumer Silver Rewards Card's 4% travel/software rate
in context to weigh against the alternatives. Extending iter3's per-
product channel to load BOTH ``_001.json`` (eligibility / fees / base
rate) AND ``_002.json`` (bonus rate for almost every revenue-bearing
card) for every credit-card product gives the LLM the full per-product
picture (eligibility, fees, base rate, bonus rate) at recommendation
time.

Mechanism (train sims the v3 frontier still fails)
--------------------------------------------------
  * task_003 (Catherine Wells, Rho-Bank+, mostly travel, hard reqs:
    no fxn fee + purchase protection + 100k+ limit). Gold:
    apply_for_credit_card(Silver Rewards Card). Silver has 4% on travel
    & software (doc_credit_cards_silver_rewards_card_002), $0 annual fee,
    fxn fee 0% with Rho-Bank+, purchase protection, limit range up to
    $100k. Without doc _002 the agent sees only Silver's "1% outside top
    categories" line from _001 and picks Gold (2.5% all categories) as
    the higher cash-back option. iter14 trace confirms: agent searches
    "no foreign transaction fees / purchase protection / high credit
    limit"; _002 never surfaces.
  * task_048 (Valentina Kostopoulos, multi-card closure with mid-flow
    recommendation). Gold: apply_for_credit_card(Silver Rewards Card).
    Customer is a marketing consultant whose Gold-card spend is
    travel + software (Adobe, project management). With only iter3-style
    _001 docs in reach, agent finds Business Silver (10% travel/software
    in _002 but $122.50 annual fee, business eligibility required) via
    bm25 and recommends that one. Loading consumer Silver _002 (4%
    travel/software, $0 annual fee, personal eligibility) into the
    system prompt gives the agent the cheaper, eligibility-compatible
    option.

The channel does not assert that Silver Rewards Card is the right
answer for either customer; it surfaces the spec sheets and lets the
LLM weigh eligibility, annual fee, base rate, and bonus rate.

Decomposition
-------------
  * Deterministic code (``card_spec_extended``): walk the KB documents
    directory, enumerate every credit-card product prefix under
    ``doc_(business_)?credit_cards_`` (excluding the cross-cutting
    procedure / feature groups ``credit_cards_(general)``,
    ``credit_card_account_logistics``, ``credit_card_replacements``,
    ``virtual_card_management``), and load both ``_001.json`` (anchor /
    eligibility) and ``_002.json`` (bonus rate) for every product. The
    pair is compiled into one catalog block grouped consumer vs.
    business and appended to the agent's system prompt.
  * LLM judgement: read the catalog, weigh the customer's stated
    spending pattern and eligibility against per-product specs, and
    decide whether/which card to recommend.

Why this captures stable structure (not training-set induction)
---------------------------------------------------------------
Two layers of KB-infrastructure facts, both independent of which
simulations are failing today:

  1. Document ids partition by product prefix:
     ``doc_(business_)?credit_cards_<product>_<NNN>.json``. The set of
     card-product prefixes is enumerable from disk; a small explicit
     set of non-product prefixes excludes cross-cutting groups.
  2. Sequential authoring convention: ``_001.json`` is the Getting
     Started / Eligibility spec sheet; ``_002.json`` is the first
     follow-on doc for the product and, for almost every revenue-bearing
     card, the bonus-category earning doc (Silver _002 "4% on travel
     and software"; Platinum _002 "Earning 4% Cash Back"; Bronze _002
     "Understanding Your 1% Cash Back Rewards"; Business Silver _002
     "10% Back on Travel and Software"; Business Platinum _002 "Earning
     4% Cash Back"). Both suffixes are deterministic ordering properties
     of the KB layout, not properties of any task.

On a freshly authored KB written under the same conventions, the rule
would select the same class of documents with no reference to any
specific failure.

The catalog never asserts a verdict on which card to recommend; it
supplies content the bm25 channel cannot reliably surface. A mis-fire
(e.g. a future product where ``_002`` is a non-rate doc) only adds
reference text and cannot make a wrong tool call.

Inherits iter13's full chain (status ledger SystemMessage refresh +
iter12 dispute card-last-4 audit + iter11 card-last-4 resolver + iter10
cashback policy engine + iter7 closure prereq advisor) unchanged. The
extension is purely additive to the system prompt.

build_agent(tools, domain_policy, **kwargs) -> HalfDuplexAgent
"""

from __future__ import annotations

from agent_tau2.mh_tau2_iter13_robust_cashback_status_ledger.agent import (
    CashbackStatusLedgerAgent,
)

from . import card_spec_extended


class CardSpecRateChannelAgent(CashbackStatusLedgerAgent):
    """LLMAgent subclass (via iter13 chain) that injects the per-product
    credit-card spec catalog — both Getting Started (``_001``) and the
    follow-on rate doc (``_002``) — into the system prompt.

    The catalog is built once at construction time from the KB docs
    directory. Activation is unconditional: the catalog (or the empty
    string if the KB cannot be located) is appended to the inherited
    base system prompt on every conversation. The catalog never
    overrides any LLM decision and never triggers a tool call.
    """

    def __init__(self, tools, domain_policy, llm, llm_args=None):
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )
        try:
            self._card_catalog = card_spec_extended.build_catalog()
        except Exception:
            self._card_catalog = ""

    @property
    def system_prompt(self) -> str:
        base = super().system_prompt
        if not self._card_catalog:
            return base
        return base + "\n\n" + self._card_catalog


def build_agent(tools, domain_policy, **kwargs):
    """Return a HalfDuplexAgent that augments the inherited system prompt
    with the KB's per-product credit-card spec sheets (both ``_001`` and
    ``_002``)."""
    return CardSpecRateChannelAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
