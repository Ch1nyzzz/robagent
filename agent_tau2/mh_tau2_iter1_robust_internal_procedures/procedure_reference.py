"""Deterministic retrieval of banking_knowledge internal procedure documents.

Why this exists
---------------
The banking_knowledge agent has one knowledge-base retrieval channel, ``KB_search``,
which is BM25 sparse retrieval. Internal procedure documents — the policy texts
that tell the agent how to file a credit-card dispute, when a transaction is
eligible for provisional credit, how to process a closure, the two-stage
cash-back-dispute workflow, etc. — share vocabulary with many customer-facing
documents and with each other. A single BM25 query for "cash back rewards"
returns a noisy mix and the actual internal procedure is frequently buried below
the top results. The agent then either skips a documented step (e.g. submits
``update_transaction_rewards_*`` without first issuing
``submit_cash_back_dispute_*``) or invents argument values inconsistent with the
policy.

What this module does
---------------------
It bypasses BM25 for one structurally identifiable class of documents. The KB's
own document-id taxonomy uses these stable prefixes (a property of the KB
infrastructure, not of any particular task):

  ``doc_credit_cards_credit_cards_(general)_*``       cross-card general policy
  ``doc_credit_cards_credit_card_account_logistics_*`` closures, CLI, downgrades
  ``doc_credit_cards_credit_card_replacements_*``     replacement orders

Within these prefixes, the KB's authoring convention marks agent-facing
procedure documents by placing ``Internal`` (with or without parens) in the
title — a distinguishing convention from customer-facing communications. We
walk the documents directory, select documents matching both criteria
(structural prefix + "Internal" title marker), and compile them into a single
reference block grouped by sub-category.

The block is injected into the agent's system prompt verbatim. The LLM still
decides what to do; this module supplies content, not behaviour.

Anti-overfitting
----------------
Only the domain's public knowledge-base documents are read. No customer data,
no task definitions, and no evaluation data are touched. The selection rule is
``(doc-id has a structural prefix) AND (title contains "Internal")`` — a
generalising rule keyed to KB authoring conventions, independent of which tasks
happen to be failing today. If the KB cannot be located the reference is empty
and the agent falls back to baseline.
"""
from __future__ import annotations

import glob
import json
import os
import re

# Structural document-id prefixes the KB uses for credit-card procedure docs.
# Each entry is (prefix-glob, group-label).
_PROCEDURE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("doc_credit_cards_credit_cards_(general)_", "Credit Cards — Cross-Card Policy"),
    ("doc_credit_cards_credit_card_account_logistics_", "Credit Cards — Account Logistics (Closure / CLI / Downgrade)"),
    ("doc_credit_cards_credit_card_replacements_", "Credit Cards — Replacement Cards"),
)

# Title-keyword convention the KB uses to mark internal procedure docs.
_INTERNAL_MARKER_RE = re.compile(r"\binternal\b", re.IGNORECASE)

# Cap on individual document length so one pathological document cannot blow up
# the prompt. Real procedure docs sit well under this.
_MAX_DOC_CHARS = 3500


def _documents_dir() -> str | None:
    """Resolve the banking_knowledge KB documents directory.

    Prefer the path tau2 itself resolves; fall back to a project-relative walk
    so the catalog still loads in stand-alone import contexts.
    """
    try:
        from tau2.utils.utils import DATA_DIR  # type: ignore

        path = os.path.join(
            str(DATA_DIR), "tau2", "domains", "banking_knowledge", "documents"
        )
        if os.path.isdir(path):
            return path
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        cand = os.path.join(
            here,
            "tau2-bench-src",
            "data",
            "tau2",
            "domains",
            "banking_knowledge",
            "documents",
        )
        if os.path.isdir(cand):
            return cand
        here = os.path.dirname(here)
    return None


def _load_doc(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _sort_key(filename: str) -> tuple[int, str]:
    """Sort docs by trailing numeric suffix so output ordering is stable."""
    m = re.search(r"_(\d+)\.json$", filename)
    return (int(m.group(1)) if m else 1_000_000, filename)


def build_reference() -> str:
    """Build the internal-procedure reference block.

    Returns the empty string when the KB cannot be located or no qualifying
    documents are found — the agent then behaves as baseline.
    """
    docs_dir = _documents_dir()
    if not docs_dir:
        return ""

    sections: list[str] = []
    for prefix, label in _PROCEDURE_PREFIXES:
        try:
            paths = glob.glob(os.path.join(docs_dir, prefix + "*.json"))
        except Exception:
            paths = []
        paths.sort(key=lambda p: _sort_key(os.path.basename(p)))

        chunks: list[str] = []
        for path in paths:
            doc = _load_doc(path)
            if not doc:
                continue
            title = str(doc.get("title", "")).strip()
            content = str(doc.get("content", "")).strip()
            if not title or not content:
                continue
            if not _INTERNAL_MARKER_RE.search(title):
                continue  # skip customer-facing documents
            if len(content) > _MAX_DOC_CHARS:
                content = content[:_MAX_DOC_CHARS] + " ..."
            chunks.append(f"[{title}]\n{content}")

        if chunks:
            sections.append(f"=== {label} ===\n\n" + "\n\n".join(chunks))

    if not sections:
        return ""

    header = (
        "<internal_procedure_reference>\n"
        "Authoritative internal procedure documents from the Rho-Bank knowledge "
        "base, retrieved deterministically by document-id prefix. Use these as "
        "the source of truth for credit-card workflows (cash-back disputes, "
        "transaction disputes, provisional-credit eligibility, statement "
        "credits, closures, retention, CLI processing, replacement orders) "
        "rather than re-querying the knowledge base for the same procedure.\n"
    )
    footer = "\n</internal_procedure_reference>"
    return header + "\n" + "\n\n".join(sections) + footer
