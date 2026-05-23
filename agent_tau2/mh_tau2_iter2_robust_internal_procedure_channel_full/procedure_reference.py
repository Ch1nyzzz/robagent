"""Deterministic retrieval of banking_knowledge internal procedure documents.

What this module does
---------------------
It bypasses BM25 for one structurally identifiable class of documents in the
banking_knowledge KB. The KB uses two stable, infrastructure-level conventions
that are independent of any particular task or simulation:

  1. Documents are partitioned on disk by document-id prefix, grouping each
     subject area (bank accounts general, checking accounts general, credit
     cards general, credit card account logistics, credit card replacements,
     customer support special codes).
  2. Within those partitions, agent-facing procedure documents are marked by
     placing the literal word "Internal" in the title (e.g. "Internal:
     Processing Credit Card Account Closures"). This is a deliberate authoring
     convention to distinguish internal procedures from customer-facing
     communications.

We walk the documents directory, select documents matching
``(doc-id prefix is one of the procedure roots) AND (title contains "Internal")``,
and compile them into one reference block grouped by sub-category. The block is
injected into the agent's system prompt verbatim.

This module extends the iter1 channel: iter1 used only the three credit-card
prefixes; here we add the bank-accounts/checking/customer-support procedure
roots so the channel covers the rest of the banking domain's authored
procedure surface. The selection rule and behaviour are otherwise unchanged.

Anti-overfitting
----------------
Only the domain's public KB documents are read. No customer data, no task
definitions, no evaluation data. The rule
``(doc-id has a procedure prefix) AND (title contains "Internal")`` is keyed to
KB authoring conventions, not to any failing task. If the KB cannot be located
the reference is empty and the agent falls back to baseline.
"""
from __future__ import annotations

import glob
import json
import os
import re

# Structural document-id prefixes the KB uses for procedure docs. Each entry
# is (prefix-glob, group-label). The credit-card prefixes were inherited from
# iter1; the bank-accounts / checking-accounts / customer-support prefixes
# extend the channel to the rest of the banking domain's procedure surface.
_PROCEDURE_PREFIXES: tuple[tuple[str, str], ...] = (
    (
        "doc_bank_accounts_bank_accounts_(general)_",
        "Bank Accounts — General Procedures",
    ),
    (
        "doc_checking_accounts_checking_accounts_(general)_",
        "Checking Accounts — General Procedures",
    ),
    (
        "doc_credit_cards_credit_cards_(general)_",
        "Credit Cards — Cross-Card Policy",
    ),
    (
        "doc_credit_cards_credit_card_account_logistics_",
        "Credit Cards — Account Logistics (Closure / CLI / Downgrade)",
    ),
    (
        "doc_credit_cards_credit_card_replacements_",
        "Credit Cards — Replacement Cards",
    ),
    (
        "doc_customer_support_special_support_codes_",
        "Customer Support — Special Support Codes",
    ),
)

# Title-keyword convention the KB uses to mark internal procedure docs.
_INTERNAL_MARKER_RE = re.compile(r"\binternal\b", re.IGNORECASE)

# Cap on individual document length so one pathological document cannot blow
# up the prompt. Real procedure docs sit well under this.
_MAX_DOC_CHARS = 3500


def _documents_dir() -> str | None:
    try:
        from tau2.utils.utils import DATA_DIR

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
                continue
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
        "base, retrieved deterministically by document-id prefix and the KB's "
        "'Internal' title convention. Use these as the source of truth for "
        "domain workflows — bank-account opening/closing/transfer, debit-card "
        "disputes and provisional credit, human-agent transfer reason codes, "
        "credit-card cash-back disputes, credit-card transaction disputes, "
        "credit-card provisional-credit eligibility, statement credits, "
        "closures, retention, CLI processing, replacement orders — rather than "
        "re-querying the knowledge base for the same procedure.\n"
    )
    footer = "\n</internal_procedure_reference>"
    return header + "\n" + "\n\n".join(sections) + footer
