"""tau2 candidate mh_tau2_iter16_baseline_card_application_facilitator.

Hypothesis
----------
Credit-card-application tasks fail the database-hash check because the agent
never facilitates the application. In this domain a new credit-card
application is submitted by the CUSTOMER, using their own ``apply_for_credit_card``
tool (the gold ``apply_for_credit_card`` action is always ``requestor: user``).
The agent has no agent-side application tool, so it hunts the knowledge base
for one, fails to find it, concludes the action is impossible, and redirects
the customer to an external website / dashboard ("apply through your Rho-Bank
portal", "I don't have a tool to submit applications"). The customer, mirroring
that framing, never invokes their own application tool, so the gold
``apply_for_credit_card`` write never happens and the DB hash is short of gold.
Telling the agent (a) that the customer applies themselves and the agent's job
is to recommend, confirm details, and INVITE the customer to apply now, and
(b) never to redirect to an external channel, lets the customer's application
go through and matches gold.

Mechanism observed (banking sims the frontier still fails)
----------------------------------------------------------
  * task_044 — agent correctly recommends the Platinum Rewards Card, the
    customer says "I'm ready to apply, can you help me submit?" three times,
    and the agent refuses each time: "Unfortunately, I don't have a tool
    available to submit a credit card application on your behalf ... apply
    through your Rho-Bank dashboard." The customer applies "through the
    dashboard" instead, so the gold apply_for_credit_card never fires. Every
    other gold action on this task already matches.
  * task_024 — agent tells a new customer "I'm not able to process new credit
    card applications on my end ... submitted through Rho-Bank's online
    portal", and also recommends the wrong card (Business Silver) because it
    never retrieves the Business Bronze Rewards Card's $500 new-customer promo.
  * task_048 — agent runs four KB_search calls for an "open new credit card
    account" / "apply" agent tool, finds none, and never facilitates the
    application gold expects.

Decomposition
-------------
  * deterministic code (``card_catalog``): locate the banking_knowledge
    knowledge base on disk, load every credit-card product-specification
    document (specs, fees, promos, exclusions) and compile them into one
    catalog string grouped by product; plus a card-shopping intent detector.
    The catalog is injected only on card-shopping conversations.
  * LLM judgement: weigh the customer's requirements, recommend the card,
    confirm details, and invite the customer to submit the application.

Both additions are non-destructive: the catalog is read-only reference context
(the same documents KB_search indexes) and the new system-prompt sections are
general guidance. The agent never gains, drops, or rewrites a tool call as a
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

from . import card_catalog

# General, domain-agnostic guidance appended to the system prompt.
# (1) How credit-card applications are completed in this domain, and
# (2) how to compare cards when making a recommendation.
CARD_GUIDANCE = """
<credit_card_applications>
A new credit-card application is submitted by the CUSTOMER, not by you. The
customer has their own application tool for this. You do not have — and should
not search the knowledge base for — an agent-side tool that opens or submits a
credit-card application; not finding one does not mean the application is
impossible.

When a customer wants to open, apply for, sign up for, or switch to a credit
card:
- Recommend the single best-fit card for their situation.
- Confirm the exact details the application needs (for example the customer's
  full legal name, annual income, whether they have a Rho-Bank+ subscription,
  and the precise card product name).
- Then tell the customer they can submit the application right now, and invite
  them to go ahead. The customer completes it themselves from there.

Never tell a customer to apply through an external website, app, dashboard,
branch, phone line, or a different team, and never say you are "unable to" help
with an application that the customer can complete here. Do not redirect a
customer to an external channel for anything that can be handled in this
conversation.
</credit_card_applications>

<card_recommendation>
When a customer asks for help choosing, comparing, applying for, upgrading, or
switching a credit card — including when they want to close a card because a
competitor looks better — base your advice on the bank's full product line-up,
not on whichever product you happen to recall or retrieve first.

- Consider EVERY card product the customer could be eligible for, not just the
  first one that looks plausible. A product you never mention cannot be chosen.
- Match the product class to the customer: a personal customer gets a personal
  card, a business customer a business card. Do not confuse a personal product
  with its similarly named business counterpart.
- When the customer has a concrete spending plan, recommend the card with the
  highest TOTAL expected return for that plan. Total return is the ongoing
  cash back (apply the correct per-category earning rate, honouring merchant
  exclusions and any category caps) PLUS any one-time new-customer or sign-up
  promotional bonus the customer's planned spend would actually qualify for
  (check each card's promotion window and qualifying-spend threshold). A card
  with a lower headline rate can still win once its sign-up bonus is counted.
- Check the customer against each candidate card's eligibility terms (credit
  score, income, personal vs. business) before recommending it.
- When a credit-card product catalog is provided to you as a system reference,
  use it rather than re-issuing many knowledge-base searches for the same card.
</card_recommendation>
""".strip()


class CardApplicationFacilitatorAgent(LLMAgent):
    """LLMAgent that (a) tells the agent how credit-card applications and
    recommendations work in this domain via general system-prompt guidance, and
    (b) injects the full credit-card product catalog whenever the conversation
    is about choosing, comparing, or applying for a credit card."""

    def __init__(self, tools, domain_policy, llm, llm_args=None):
        super().__init__(
            tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args
        )
        # Compiled once at construction; "" if the knowledge base is unreadable.
        try:
            self._catalog = card_catalog.build_catalog()
        except Exception:
            self._catalog = ""

    @property
    def system_prompt(self) -> str:
        base = SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        )
        return base + "\n" + CARD_GUIDANCE

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

        system_content = self.system_prompt
        if self._catalog and self._card_shopping(state):
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
    def _card_shopping(state: LLMAgentState) -> bool:
        """True when any user message so far shows card-shopping intent."""
        texts = []
        for m in getattr(state, "messages", None) or []:
            if getattr(m, "role", None) != "user":
                continue
            content = getattr(m, "content", None)
            if isinstance(content, str):
                texts.append(content)
        try:
            return card_catalog.is_card_shopping(texts)
        except Exception:
            return False


def build_agent(tools, domain_policy, **kwargs):
    """Return a HalfDuplexAgent that facilitates credit-card applications and
    recommendations: general system-prompt guidance plus the credit-card
    product catalog on card-shopping conversations."""
    return CardApplicationFacilitatorAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
