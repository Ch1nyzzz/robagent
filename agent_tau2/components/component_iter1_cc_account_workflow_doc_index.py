"""Inject a navigational index of credit-card-account-workflow KB doc IDs.

Mechanism
---------
Tau2 banking gold workflows for credit card closure (task_044/045/048),
credit limit increase (task_050/051/053/054), and replacement-card-related
flows (task_036/037/038) all hinge on internal KB docs whose titles begin
with "Internal: " (e.g., "Internal: Processing Credit Card Account
Closures", "Internal: Credit Card Retention Protocol", "Internal: CLI
Payment History and Approval Criteria"). The default classic_rag_bm25 KB
pipeline ranks these "Internal:" docs poorly against customer-phrased
queries ("I want to close my card", "can I raise my limit?") because the
docs' lexical surface is procedural and does not echo the user's
phrasing. Per memory [[tau2-retention-protocol]] and
[[tau2-closure-cli-prerequisite-reads]], gold workflows on these tasks
reference content that BM25 never surfaces. Without the docs, the agent
cannot identify the prerequisite reads
(`get_user_dispute_history_7291`, `get_pending_replacement_orders_5765`)
or the workflow tools (`close_credit_card_account_7834`,
`log_credit_card_closure_reason_4521`, etc.) that the gold action sets
require.

Why CHANNEL (not the others)
----------------------------
The matcher tests no task-specific predicate; it fires on every session
to inject a static catalog of stable doc IDs + titles from the
banking_knowledge KB filesystem at
`tau2-bench-src/data/tau2/domains/banking_knowledge/documents/`. The
anchor is the KB filesystem layout (system field — doc IDs are file
basenames, titles are JSON metadata stored on disk), and the content is
KB-doc metadata that the agent cannot reliably reach through the
provided BM25 KB_search tool. CHANNEL is the class for "request needing
content the agent cannot otherwise reach (file, URL, KB doc)" with an
inject_context handler. The injection is NOT a policy reading — it
enumerates doc IDs and their on-disk titles. The LLM then uses `grep`
(by doc ID) or KB_search (by title token) to fetch the full doc and do
its own reading; this component does not encode any policy conclusion or
override the LLM's judgement.

Out-of-evidence probe
---------------------
Consider a hypothetical task asking the agent to downgrade a card to a
no-annual-fee card (logistics_008 — not in any of the train-30 evidence
sims). The matcher still fires (always-true), and the handler injects
the same index, which lists `doc_credit_cards_credit_card_account_logistics_008`
with its title ("Internal: Downgrading a Credit Card to a No-Annual-Fee
Card"). The agent can then grep that doc-id directly. Correctness
follows because the index is purely a navigational catalog — the agent
still reads the doc and decides what to do; the component has no opinion
on the downgrade workflow.

Dead-weight signal
------------------
If a future tau2 release renames the credit-card account-logistics or
replacement doc IDs (e.g., reorganises the documents directory), the
injected IDs become stale and the agent's grep returns nothing. This is
observable via durability_audit.py as zero retrievals against the listed
IDs, and via train-30 reward not improving on closure/CLI/replacement
tasks even after the inject. Also dead weight if the underlying KB
retrieval pipeline is replaced with one that ranks Internal: docs
correctly (e.g., a dense embedding pipeline) — then BM25's bias goes
away and the index becomes redundant.
"""
from __future__ import annotations

from agent_tau2.component_runtime.types import (
    Capability,
    Component,
    ComponentClass,
    ComponentContext,
    Decision,
    Mount,
    StateScope,
    Trust,
)


_DOC_INDEX = (
    "<credit_card_account_workflow_kb_index>\n"
    "## Credit Card Account Workflow — KB Doc Index\n\n"
    "The Rho-Bank KB has a small set of named internal docs that govern "
    "credit-card account workflows (closure, credit-limit-increase, "
    "downgrade, balance payoff, replacement). The default BM25 KB_search "
    "frequently ranks these docs low because their titles begin with "
    "'Internal:' and their lexical surface does not echo customer "
    "phrasing. Use `grep` on the doc-ID (or `KB_search` with a title "
    "token) when the customer's request matches one of these flows.\n\n"
    "### Credit Card Account Logistics family\n"
    "  - doc_credit_cards_credit_card_account_logistics_001 — "
    "How can I close a credit card account?\n"
    "  - doc_credit_cards_credit_card_account_logistics_002 — "
    "Internal: Processing Credit Card Account Closures\n"
    "  - doc_credit_cards_credit_card_account_logistics_003 — "
    "Internal: Credit Card Retention Protocol\n"
    "  - doc_credit_cards_credit_card_account_logistics_004 — "
    "How can I request a CLI?\n"
    "  - doc_credit_cards_credit_card_account_logistics_005 — "
    "CLI Eligibility Requirements by Card Tier\n"
    "  - doc_credit_cards_credit_card_account_logistics_006 — "
    "Internal: CLI Payment History and Approval Criteria\n"
    "  - doc_credit_cards_credit_card_account_logistics_007 — "
    "Internal: Processing CLI Approvals and Denials\n"
    "  - doc_credit_cards_credit_card_account_logistics_008 — "
    "Internal: Downgrading a Credit Card to a No-Annual-Fee Card\n"
    "  - doc_credit_cards_credit_card_account_logistics_009 — "
    "Paying Off Credit Card Balance from Checking Account (Internal)\n\n"
    "### Credit Card Replacements family\n"
    "  - doc_credit_cards_credit_card_replacements_001 — "
    "How to Order a Replacement Credit Card (Internal)\n"
    "  - doc_credit_cards_credit_card_replacements_002 — "
    "Credit Card Replacement Shipping Options and Fees\n"
    "  - doc_credit_cards_credit_card_replacements_003 — "
    "What Happens When You Order a Replacement Card\n"
    "  - doc_credit_cards_credit_card_replacements_004 — "
    "Why was your credit card replacement request rejected\n"
    "  - doc_credit_cards_credit_card_replacements_005 — "
    "Checking Pending Replacement Card Orders (Internal)\n\n"
    "When a customer's request matches one of these flows, prefer "
    "fetching the relevant Internal: doc directly (by ID via `grep`) "
    "over relying on BM25 KB_search alone — BM25 may surface only the "
    "customer-facing FAQ ('How can I…?') and miss the Internal: "
    "processing doc that names the workflow's discoverable tools and "
    "eligibility prerequisites. The Internal: docs are the source of "
    "truth for which discoverable agent-tools to unlock and call.\n"
    "</credit_card_account_workflow_kb_index>"
)


def _matches(ctx: ComponentContext) -> bool:
    return True


def _handler(ctx: ComponentContext) -> Decision:
    return Decision.inject_context(_DOC_INDEX)


COMPONENT = Component(
    name="cc_account_workflow_doc_index",
    cls=ComponentClass.CHANNEL,
    mount=Mount.SESSION_START,
    matcher=_matches,
    handler=_handler,
    state_scope=StateScope.NONE,
    capabilities=(Capability.NONE,),
    priority=150,                     # after discoverable_audit_channel (200 fires later)
    trust=Trust(
        evidence_anchor=(
            "The banking_knowledge KB stores docs as JSON files at "
            "tau2-bench-src/data/tau2/domains/banking_knowledge/documents/, "
            "each with a stable `id` (the basename) and `title` field. The "
            "doc IDs and titles enumerated by this channel are read directly "
            "from those on-disk files; the naming convention "
            "`doc_credit_cards_credit_card_account_logistics_NNN` and "
            "`doc_credit_cards_credit_card_replacements_NNN` is a fixed "
            "organisational fact of the KB filesystem layout. The "
            "`grep` tool (retrieval_mixins.py:84) matches doc-ID strings "
            "exactly, providing deterministic retrieval given the ID."
        ),
        blast_radius="workflow",
        rollback_when=(
            "Tau2 reorganises the banking_knowledge documents/ directory "
            "such that the listed doc IDs no longer exist (observable as "
            "grep-by-id returning empty on listed IDs), OR the KB retrieval "
            "pipeline is upgraded to one that ranks Internal: titled docs "
            "correctly (observable as train-30 reward on tasks 044/045/048/"
            "050/051/053/054 reaching parity with the BM25 baseline plus this "
            "index, then no longer benefiting from the index)."
        ),
        out_of_evidence_probe=(
            "On a hypothetical credit-card downgrade task (logistics_008 — "
            "not in any train-30 evidence sim), the always-true matcher "
            "fires and the handler injects the same doc index, which "
            "includes the downgrade doc's ID and title. The agent can grep "
            "the listed doc-ID directly. Correctness follows because the "
            "channel is purely a navigational catalog of on-disk facts; the "
            "LLM still reads the doc and decides the workflow. No rewrite "
            "or block; worst case is unused context."
        ),
        fallback=(
            "Matcher is always True; fallback is not applicable. If the "
            "injected text is ignored by the LLM, the component has no "
            "effect on task scores and is detectable as dead weight via "
            "durability_audit.py."
        ),
    ),
)
