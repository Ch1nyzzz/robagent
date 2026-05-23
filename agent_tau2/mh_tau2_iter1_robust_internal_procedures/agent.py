"""tau2 candidate mh_tau2_iter1_robust_internal_procedures.

Hypothesis
----------
Banking-knowledge tasks involving credit-card cash-back disputes, credit-card
transaction disputes, provisional credit eligibility, statement credits,
account closures, retention, CLI processing, and replacement orders fail
because the agent's only knowledge-base retrieval channel (``KB_search``) is
BM25 sparse retrieval — the internal procedure documents share vocabulary with
many similar customer-facing documents and the agent's bm25 queries fail to
surface them reliably. The agent then either skips a documented stage of a
workflow (e.g. issues ``update_transaction_rewards_*`` without first issuing
``submit_cash_back_dispute_*``) or fills argument values inconsistent with the
authoritative procedure. Deterministically pre-loading the KB's internal
credit-card procedure documents — selected by structural document-id prefix
plus the KB's "Internal" title convention — into the agent's system prompt
gives the LLM reliable access to the actual policy text, without changing what
the LLM may decide.

Mechanism (banking sims the frontier still fails)
-------------------------------------------------
  * task_018, task_028, task_029 — agent never gives
    ``submit_cash_back_dispute_*`` and goes straight to
    ``update_transaction_rewards_*``: the two-stage cash-back-correction policy
    (KB doc ``..._(general)_003`` + ``_004``) is documented but bm25 doesn't
    surface it.
  * task_031, task_037 — agent files credit-card transaction disputes with
    args inconsistent with the filing policy doc
    (``..._(general)_014``).
  * task_044 — agent unlocks extra discoverable tools for a closure task,
    inconsistent with the closure-processing procedure
    (``..._account_logistics_002``).

Decomposition
-------------
  * deterministic code (``procedure_reference``): walk the KB documents
    directory, select documents matching ``(structural doc-id prefix) AND
    (title contains "Internal")``, and compile them into one reference block
    grouped by sub-category.
  * LLM judgement: read the reference, decide which procedure applies to the
    current conversation, and act accordingly.

Why this captures stable structure (not training-set induction)
---------------------------------------------------------------
The selection rule is keyed to two KB-infrastructure facts that are
independent of the failed simulations:

  1. Documents are partitioned by document-id prefix — a fixed property of
     how the KB is laid out on disk. The three prefixes used here cover the
     credit-card procedure categories.
  2. Internal (agent-facing) procedure documents are marked by an "Internal"
     keyword in their title — a deliberate KB authoring convention to
     distinguish them from customer-facing communications.

The reference block contains the documents verbatim. The advisory does not
encode any branching rule induced from failing tasks; it is a retrieval
channel that surfaces what BM25 should ideally surface but doesn't.

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
    """LLMAgent that always injects the KB's internal credit-card procedure
    documents into the system prompt."""

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
    """Return a HalfDuplexAgent that augments LLMAgent's system prompt with the
    KB's internal credit-card procedure documents."""
    return InternalProcedureChannelAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
