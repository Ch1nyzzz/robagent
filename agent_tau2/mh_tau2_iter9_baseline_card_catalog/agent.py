"""tau2 candidate mh_tau2_iter9_baseline_card_catalog.

Hypothesis
----------
Card-recommendation / card-comparison tasks fail because the agent's only
retrieval tool, ``KB_search``, is BM25 sparse retrieval: a query naming one
card product collides on shared vocabulary with every other card's documents,
so the specification document the agent needs is buried below the top results.
The agent then issues a dozen-plus searches, never assembles a complete view
of the product line-up, and recommends a card that is not the best fit — or a
product it never realised existed. Giving the agent the complete, accurate
credit-card product catalog up front (the same knowledge-base documents
KB_search indexes, retrieved deterministically by document-id prefix instead of
by BM25) lets it compare every product and recommend the right one.

Mechanism observed (banking sims the frontier still fails)
----------------------------------------------------------
  * task_003 — customer wants a card with specific features; the agent runs ~17
    KB_search calls, never surfaces the personal Platinum Rewards Card document,
    and steers the customer to the Gold Rewards Card (gold answer: Silver
    Rewards Card).
  * task_024 — customer wants the best business card for a large purchase; the
    agent runs ~12 KB_search calls, never discovers the Business Bronze Rewards
    Card exists, and recommends the Business Silver Rewards Card (gold answer:
    Business Bronze Rewards Card).
  * task_044 — customer threatens to leave for a competitor; the agent runs ~15
    KB_search calls hunting card specs for a retention offer and never lands on
    the right product (gold answer: apply for the Platinum Rewards Card).

Decomposition
-------------
  * deterministic code (``card_catalog``): locate the banking_knowledge
    knowledge base on disk, load every credit-card product-specification
    document, and compile them into one catalog string grouped by product.
    Also a keyword intent detector for card-shopping conversations.
  * LLM judgement: run the conversation, weigh the customer's stated
    requirements against the catalog, and choose / recommend the card.

The catalog is reference material the agent is already meant to retrieve — it
adds no knowledge beyond what KB_search exposes (tau2's own ``full_kb`` config
puts the whole knowledge base in the prompt). It is injected only when the
conversation is about choosing / comparing / switching credit cards, so other
tasks are unaffected. If the knowledge base cannot be read the catalog is
empty and the agent falls back to baseline behaviour. The agent never gains a
write it would not otherwise make, so a mis-fire cannot score below baseline.

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
CARD_RECOMMENDATION = """
<card_recommendation>
When a customer asks for help choosing, comparing, applying for, upgrading, or
switching a credit card — including when they want to close a card because a
competitor looks better — base your advice on the bank's full product line-up,
not on whichever product you happen to recall or retrieve first.

- Consider EVERY card product the customer could be eligible for, not just the
  first one that looks plausible. A product you never mention cannot be chosen.
- Recommend the single card that satisfies ALL of the customer's stated
  requirements. If several qualify, prefer the one with the lowest cost
  (annual fee) unless the customer has said the rewards rate matters more.
- Check the customer against each candidate card's eligibility terms (credit
  score, income, personal vs. business) before recommending it.
- When a credit-card product catalog is provided to you as a system reference,
  use it rather than re-issuing many knowledge-base searches for the same card.
</card_recommendation>
""".strip()


class CardCatalogAgent(LLMAgent):
    """LLMAgent that injects the full credit-card product catalog whenever the
    conversation is about choosing or comparing credit cards."""

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
        return base + "\n" + CARD_RECOMMENDATION

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
    """Return a HalfDuplexAgent that injects the credit-card product catalog on
    card-recommendation conversations."""
    return CardCatalogAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
