"""Deterministic credit-card account-procedure reference for banking_knowledge.

Why this exists
---------------
Account-closure tasks fail the database-hash check because the agent never
assembles the complete closure procedure. The procedure is spread across
several knowledge-base documents:

  * the closure eligibility requirements and the closure process,
  * the multi-step credit-card retention protocol, and
  * the how-to documents for the discoverable agent tools each step uses
    (checking dispute history, checking pending replacement card orders,
    logging the closure reason, applying a statement credit, etc.).

The agent's only retrieval tool, ``KB_search``, is BM25 sparse retrieval. A
closure query collides on shared vocabulary across the whole credit-card
namespace, so it surfaces one closure document but rarely the rest — and almost
never the tool how-to documents. The retention-protocol text describes the
prerequisite checks in prose ("verify there are no pending disputes", "no
pending replacement cards") but does NOT name the tools that perform them, so an
agent that only retrieves the procedure document still cannot find
``get_user_dispute_history_7291`` or ``get_pending_replacement_orders_5765``.
The agent then skips those prerequisite reads, closes a card it should not have
(one with a pending replacement order), or improvises the retention protocol
and never reaches the documented statement-credit retention offer.

What this module does
---------------------
It sidesteps BM25 for that document class. ``build_closure_reference`` loads,
straight from the domain knowledge base on disk, every credit-card
account-procedure document and compiles them into one reference string:

  * the ``credit_card_account_logistics`` document family (closure eligibility,
    closure process, the retention protocol, the CLI / downgrade procedures),
    discovered by its document-id family; and
  * every credit-card-namespace document that is structured as how-to
    documentation for a discoverable agent tool, discovered by the literal
    "use the <tool> tool" phrasing those documents share.

These are exactly the documents ``KB_search`` already indexes — handing them
over wholesale is a retrieval improvement, not extra knowledge (tau2's own
``full_kb`` configuration puts the entire knowledge base in the prompt). The
agent injects the reference only when the conversation is about closing a
credit-card account.

Anti-overfitting
----------------
Only the domain's public knowledge-base documents are read. No customer data,
no task definitions, and no evaluation data are touched. Documents are selected
by a content-driven topic filter (document-id family + tool-doc phrasing) —
nothing about any specific task, customer, card, or account is hard-coded.
"""
from __future__ import annotations

import glob
import json
import os
import re

# Account-procedure documents share this document-id family. It covers closure
# eligibility, the closure process, the retention protocol, and the adjacent
# CLI / downgrade procedures — the full credit-card account-management workflow.
_LOGISTICS_RE = re.compile(
    r"^doc_credit_cards_credit_card_account_logistics_\d+$"
)

# A how-to document for a discoverable agent tool spells out the tool by name in
# this stock phrasing ("use the <tool_name> tool" / "use <tool_name>"). The tool
# name always ends in a short numeric suffix.
_TOOLDOC_RE = re.compile(r"use (?:the )?`?[a-z][a-z_]+_\d{3,4}\b", re.IGNORECASE)

# Per-document content cap — a safety valve so one unusually long document
# cannot blow up the prompt. Real account-procedure documents sit well under it,
# so the retention protocol (the largest, and the one task outcomes depend on)
# is never truncated.
_MAX_DOC_CHARS = 6000


def _documents_dir() -> str | None:
    """Resolve the banking_knowledge knowledge-base documents directory."""
    try:
        from tau2.utils.utils import DATA_DIR  # resolved by tau2 itself

        path = os.path.join(
            str(DATA_DIR), "tau2", "domains", "banking_knowledge", "documents"
        )
        if os.path.isdir(path):
            return path
    except Exception:
        pass
    # Fallback: walk up from this file looking for the documents directory.
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        cand = os.path.join(
            here, "tau2-bench-src", "data", "tau2", "domains",
            "banking_knowledge", "documents",
        )
        if os.path.isdir(cand):
            return cand
        here = os.path.dirname(here)
    return None


def _is_account_procedure_doc(doc_id: str, content: str) -> bool:
    """True when a document belongs in the credit-card account-procedure set:
    either the account-logistics document family, or a how-to document for a
    discoverable agent tool in the credit-card namespace."""
    if _LOGISTICS_RE.match(doc_id):
        return True
    if doc_id.startswith(("doc_credit_cards_", "doc_business_credit_cards_")):
        if _TOOLDOC_RE.search(content or ""):
            return True
    return False


def build_closure_reference() -> str:
    """Build the credit-card account-procedure reference string.

    Returns an empty string if the knowledge base cannot be located or read —
    the agent then simply behaves as the baseline with no reference injected.
    """
    docs_dir = _documents_dir()
    if not docs_dir:
        return ""

    try:
        paths = glob.glob(os.path.join(docs_dir, "doc_*.json"))
    except Exception:
        return ""

    selected: list[tuple[str, str, str]] = []  # (doc_id, title, content)
    for path in sorted(paths):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
        except Exception:
            continue
        doc_id = str(doc.get("id", ""))
        content = str(doc.get("content", "")).strip()
        if not content:
            continue
        if not _is_account_procedure_doc(doc_id, content):
            continue
        title = str(doc.get("title", "")).strip()
        if len(content) > _MAX_DOC_CHARS:
            content = content[:_MAX_DOC_CHARS] + " ..."
        selected.append((doc_id, title, content))

    if not selected:
        return ""

    header = (
        "<credit_card_account_procedure_reference>\n"
        "Reference: the complete Rho-Bank knowledge-base documentation for "
        "credit-card account procedures — closure eligibility, the closure "
        "process, the retention protocol, and the how-to documents for the "
        "discoverable agent tools each step uses. Follow this procedure exactly "
        "instead of relying on repeated knowledge-base searches; it names every "
        "prerequisite check and the tool that performs it.\n"
    )
    body = []
    for doc_id, title, content in selected:
        label = title or doc_id
        body.append(f"[{label}]\n{content}")
    footer = "\n</credit_card_account_procedure_reference>"
    return header + "\n" + "\n\n".join(body) + footer


# --- conversation intent detection ---------------------------------------

# A closure conversation: a user message naming a closing action together with
# a card / account noun. Detection is sticky — once any earlier user message
# qualifies, the reference stays in for the rest of the conversation.
_CLOSE_TOKENS = (
    "close",
    "closing",
    "cancel",
    "shut down",
    "shutting down",
    "terminate",
    "get rid of",
)

_ACCOUNT_TOKENS = ("card", "account")


def is_closure_conversation(user_texts) -> bool:
    """True when a user message shows intent to close a credit-card account."""
    for text in user_texts or []:
        if not isinstance(text, str):
            continue
        low = text.lower()
        if not any(tok in low for tok in _CLOSE_TOKENS):
            continue
        if any(tok in low for tok in _ACCOUNT_TOKENS):
            return True
    return False
