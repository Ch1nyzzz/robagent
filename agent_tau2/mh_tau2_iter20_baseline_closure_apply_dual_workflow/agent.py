"""tau2 candidate mh_tau2_iter20_baseline_closure_apply_dual_workflow.

Hypothesis
----------
The credit-card account-closure / retention cluster (the largest block of
frontier-failing train tasks) fails the database-hash check because a closure
conversation has TWO halves the agent never completes together:

  (A) the documented closure / retention procedure — the discoverable closure
      workflow, the per-card decisions, and the retention protocol that decides
      whether a card stays open or is closed; and
  (B) when the customer wants a different card, facilitating the customer's own
      ``apply_for_credit_card`` submission.

Every prior attempt emphasised one half and dropped the other:

  * Emphasising the closure procedure (prereq reads, retention) made the agent
    finish the closure but never facilitate the new application — the gold
    ``apply_for_credit_card`` (a ``requestor: user`` action) is missing.
  * Emphasising the recommendation / application (a card catalog + "recommend
    the best card" framing) flipped the agent into pure-recommend mode: it
    facilitated the apply but skipped the discoverable closure-workflow calls.

A balanced system prompt that frames the closure / retention procedure as the
PRIMARY work and the application as an explicitly SUBORDINATE additional step —
never a replacement — lets the agent emit both halves of the gold write set.

Mechanism observed (frontier-failing banking train tasks, DB-hash basis)
-----------------------------------------------------------------------
  * task_044 — retention closure followed by the customer applying for a new
    card. The agent completes every gold closure action EXCEPT
    ``apply_for_credit_card``: finding no agent-side application tool it
    redirects the customer to an external dashboard, so the customer never
    submits and the gold apply write never lands.
  * task_048 — a multi-card closure. The agent OVER-closes cards the procedure
    keeps open (it should only log their closure reason) and also misses the
    new-card ``apply_for_credit_card`` and a ``apply_statement_credit``.
  * task_045 — a closure where the retention offer is ACCEPTED, so the gold
    write set has NO close at all (the customer keeps the card after a
    statement-credit offer); the agent closes it anyway and misses the
    ``apply_statement_credit`` retention write.

In all three the divergence is the closure WRITE set: the agent either adds a
close the procedure forbids or drops a write the procedure (or the customer's
own application) requires.

Fix
---
A single, domain-derived ``<account_closure_and_retention>`` /
``<credit_card_applications>`` pair of system-prompt sections, ordered so the
closure / retention procedure dominates and the application is subordinate:

  * Closure / retention: a closure is a documented multi-step procedure;
    complete every prerequisite eligibility check (dispute history, pending
    replacement orders) with its dedicated tool before any closure decision;
    run the retention protocol and offer the documented incentive first; if the
    customer ACCEPTS retention, apply the incentive and KEEP the card open;
    only close after the customer declines and confirms; with several cards,
    act on each individually and never close a card the customer did not ask
    about.
  * Applications: a new credit-card application is submitted by the CUSTOMER
    with their own ``apply_for_credit_card`` tool; the agent has no agent-side
    application tool and must never redirect to an external channel; when the
    customer wants a different card, recommend the best fit and invite them to
    submit now — but facilitating an application is an ADDITIONAL step that
    never replaces the closure / retention procedure.

This is a prompt-only candidate: it changes only the system-prompt text and
never gains, drops, or rewrites a tool call, so a frontier-passing task cannot
structurally regress. No customer names, ids, card ids, rates, or per-task
branching — the guidance restates the bank's own documented procedure.

build_agent(tools, domain_policy, **kwargs) -> HalfDuplexAgent
"""
from __future__ import annotations

from tau2.agent.llm_agent import (
    AGENT_INSTRUCTION,
    SYSTEM_PROMPT,
    LLMAgent,
)

# General, domain-derived guidance. This restates Rho-Bank's own documented
# account-closure / retention procedure and how customer applications are
# filed; it is not task-specific and hardcodes no identifiers.
CLOSURE_APPLY_GUIDANCE = """
<account_closure_and_retention>
A request to close a credit-card account is a documented multi-step procedure,
not a single action. Before deciding anything:
- Retrieve the full closure procedure from the knowledge base.
- Complete every prerequisite eligibility check the procedure lists. The
  customer's dispute history and any pending replacement-card orders are
  universal prerequisites — look each one up with its own dedicated tool
  before making any closure decision.
- Never close a card the procedure says must stay open (for example, a card
  with a pending replacement-card order).

Follow the retention protocol before any closure:
- Offer the documented retention incentive (such as a statement credit) before
  closing the account.
- If the customer ACCEPTS a retention offer, apply that incentive and KEEP the
  card open. Do not close a card the customer has decided to keep.
- Close a card only after the customer has declined retention and confirmed
  they still want it closed; then log the closure reason for that card.

When the customer holds several cards, act on each card individually: close
only the specific card(s) the customer asked to close and the procedure
allows, and leave every other card untouched. Do not close, or change the
status of, a card the customer never asked about.
</account_closure_and_retention>

<credit_card_applications>
A new credit-card application is submitted by the CUSTOMER using their own
apply_for_credit_card tool. You have no agent-side application tool, so do not
search the knowledge base for one, and never tell the customer to apply through
an external website, app, branch, or another team — that is not how an
application is filed here.

When a customer wants to open, switch to, or replace a card with a different
product, recommend the best-fit card for their stated needs, confirm the
details the application needs, and invite the customer to submit the
application now with their own apply_for_credit_card tool.

Facilitating an application is an ADDITIONAL step; it never replaces the
documented closure or retention procedure. If a conversation involves both
closing or keeping an existing card and opening a new one, complete the full
closure / retention procedure AND facilitate the new application — completing
one is never a substitute for the other.
</credit_card_applications>
""".strip()


class ClosureApplyDualWorkflowAgent(LLMAgent):
    """LLMAgent that adds balanced closure/retention and application guidance to
    the system prompt so a retention-closure conversation completes both the
    documented closure procedure and the customer's new-card application."""

    @property
    def system_prompt(self) -> str:
        base = SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        )
        return base + "\n" + CLOSURE_APPLY_GUIDANCE


def build_agent(tools, domain_policy, **kwargs):
    """Return a HalfDuplexAgent that completes both halves of a retention-closure
    conversation: the documented closure procedure and the customer's apply."""
    return ClosureApplyDualWorkflowAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
