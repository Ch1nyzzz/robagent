"""tau2 candidate mh_tau2_iter2_robust_internal_procedure_channel_full.

Hypothesis
----------
iter1's internal-procedure channel injects the credit-card Internal procedure
documents into the system prompt but stops at the credit-card subject area —
it never surfaces bank-account internal procedure docs, including
``doc_bank_accounts_bank_accounts_(general)_042`` ("Internal: Human Agent
Transfer Reason Codes") and the bank-account opening / closing / transfer
procedure docs. Tasks that turn on those policies (transfer_to_human_agents
with a tiered reason code; multi-step bank-account workflows that span
opening, transferring funds, and closing accounts) therefore still fail
because BM25 KB_search does not reliably surface those docs either.

Extending the same channel — same selection rule, broader doc-id prefix list
covering the rest of the banking domain's authored procedure roots — gives the
LLM reliable access to those policy texts without changing what the LLM may
decide.

Mechanism (banking_knowledge sims the iter1 frontier still fails)
----------------------------------------------------------------
  * task_004 — identity-verification failure escalation: agent calls
    transfer_to_human_agents with reason="customer_requests_human_no_specific_
    reason" (Tier 3); gold is "account_ownership_dispute" (Tier 1). The
    tiered-code policy lives in doc bank_accounts_(general)_042; iter1 does
    not load it.
  * task_055 — open a checking account and a savings account, then make a
    deposit. Agent opens the wrong savings account class. The bank-account
    opening procedures (docs 001-004) live under
    doc_bank_accounts_bank_accounts_(general)_; iter1 does not load them.
  * task_062 — open/transfer/close bank accounts. Agent's later transfers and
    closures use wrong account ids. The transferring-funds and closing
    procedures (docs 007/008/010) live under the same prefix and are not in
    iter1.
  * task_014 — customer claims a referral promotion the KB cannot confirm.
    Gold expects transfer_to_human_agents with reason
    "unconfirmed_external_communication" (Tier 2). Without the tiered-code
    policy in context the agent does not know that code exists and never
    transfers.

Decomposition
-------------
  * deterministic code (`procedure_reference`): walk the KB documents
    directory, select documents matching ``(doc-id has a procedure prefix)
    AND (title contains "Internal")``, and compile them into one reference
    block grouped by sub-category.
  * LLM judgement: read the reference, decide which procedure applies to the
    current conversation, and act accordingly.

Why this captures stable structure (not training-set induction)
---------------------------------------------------------------
The selection rule is keyed to two KB-infrastructure facts that are
independent of the failed simulations:

  1. Documents are partitioned by document-id prefix — a fixed property of
     how the KB is laid out on disk. We list every procedure-bearing prefix
     in the banking domain (bank_accounts, checking_accounts, credit_cards
     general / account_logistics / replacements, customer_support special
     codes).
  2. Internal (agent-facing) procedure documents are marked by the word
     "Internal" in their title — a deliberate KB authoring convention.

No branching rule induced from any specific failed task is encoded. The
reference block contains the documents verbatim; the LLM still picks the
applicable procedure and acts on it.

build_agent(tools, domain_policy, **kwargs) -> HalfDuplexAgent
"""
from __future__ import annotations

from tau2.agent.llm_agent import (
    AGENT_INSTRUCTION,
    SYSTEM_PROMPT,
    LLMAgent,
)

from . import procedure_reference


class InternalProcedureChannelAgent(LLMAgent):
    """LLMAgent that always injects the KB's full Internal procedure block
    into the system prompt."""

    def __init__(self, tools, domain_policy, llm, llm_args=None):
        super().__init__(
            tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args
        )
        try:
            self._reference = procedure_reference.build_reference()
        except Exception:
            self._reference = ""

    @property
    def system_prompt(self) -> str:
        base = SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        )
        if not self._reference:
            return base
        return base + "\n\n" + self._reference


def build_agent(tools, domain_policy, **kwargs):
    return InternalProcedureChannelAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
