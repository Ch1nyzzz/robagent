"""Deterministic retrieval of banking_knowledge credit-card product spec sheets.

Why this exists
---------------
The agent's only knowledge-base retrieval channel, ``KB_search``, is BM25 sparse
retrieval. A query that names one credit-card product (e.g. "Silver Rewards
Card cash back rate") collides on shared vocabulary with every other card's
documents — most importantly with ``Internal: Credit Card Rewards`` and the
business-card spec sheets — and the top results are often the same noisy mix
regardless of which product the agent asked about. The result, repeatedly
observed when the customer asks the agent for a credit-card recommendation, is
that the agent issues a dozen+ KB_search calls, never assembles a complete
view of the per-product specifications (annual fee, APR, foreign-transaction
fee, credit-limit range, eligibility criteria, top cash-back rate), and
recommends a card that does not satisfy the customer's stated requirements.

What this module does
---------------------
It bypasses BM25 for one structurally identifiable class of documents: the
"Getting Started / Eligibility" spec sheet for each credit-card product. The
KB's filesystem layout exposes two stable conventions:

  1. Document ids partition by product. Every credit-card product is a prefix
     ``doc_credit_cards_<product>_*.json`` or
     ``doc_business_credit_cards_<product>_*.json``. A handful of prefixes are
     reserved for cross-cutting procedure groups, not card products:
     ``credit_cards_(general)``, ``credit_card_account_logistics``,
     ``credit_card_replacements``, and the feature group
     ``virtual_card_management``. Everything else is a card product.

  2. Within a product's directory, the file with suffix ``_001.json`` is the
     authored "Getting Started / Eligibility / Apply" spec sheet — the
     authoritative per-product fact sheet. This ordering is a property of how
     the KB was authored, not of any task.

Both rules are KB-infrastructure facts independent of which simulations are
failing today. The selection here is "for each product prefix, take the
``_001.json`` doc". The compiled catalog goes into the agent's system prompt
verbatim; the LLM still decides which card matches the customer's needs and
whether to make a recommendation at all. The catalog never asserts a verdict.

Anti-overfitting
----------------
Only the domain's public knowledge-base documents are read. No customer data,
no task definitions, and no evaluation data are touched. The catalog content
is the documents the agent is already supposed to retrieve, surfaced through a
deterministic channel because BM25 cannot. If the KB cannot be located the
catalog is empty and the agent falls back to baseline behaviour.
"""
from __future__ import annotations

import glob
import json
import os
import re

# Prefixes under ``doc_(business_)?credit_cards_`` that are NOT card products
# but cross-cutting procedure groups or features. Filenames whose product
# segment matches one of these are excluded from the per-product spec catalog.
_NON_PRODUCT_SUFFIXES: frozenset[str] = frozenset(
    {
        "credit_cards_(general)",
        "credit_card_account_logistics",
        "credit_card_replacements",
        "virtual_card_management",
    }
)

# Filename pattern: doc_(business_)?credit_cards_<product>_<NNN>.json
_PRODUCT_RE = re.compile(
    r"^doc_(?P<scope>business_credit_cards|credit_cards)_(?P<product>.+?)_(?P<idx>\d{3})\.json$"
)

# Per-doc content cap so one pathological doc cannot blow up the prompt.
# Real spec sheets sit well under this.
_MAX_DOC_CHARS = 2200


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


def _discover_product_specs(docs_dir: str) -> list[tuple[str, str, str]]:
    """Return [(scope, product, path-to-001-doc)] for every credit-card
    product whose ``_001.json`` exists on disk.

    ``scope`` is "credit_cards" or "business_credit_cards" — the KB's
    top-level partition between consumer and business cards.
    """
    try:
        entries = sorted(os.listdir(docs_dir))
    except Exception:
        return []

    products: dict[tuple[str, str], str] = {}
    for fname in entries:
        m = _PRODUCT_RE.match(fname)
        if not m:
            continue
        if m.group("idx") != "001":
            continue
        product = m.group("product")
        if product in _NON_PRODUCT_SUFFIXES:
            continue
        scope = m.group("scope")
        products[(scope, product)] = os.path.join(docs_dir, fname)

    # Stable ordering: consumer cards first, then business cards, alpha within.
    def order_key(item: tuple[tuple[str, str], str]) -> tuple[int, str]:
        (scope, product), _ = item
        return (0 if scope == "credit_cards" else 1, product)

    return [
        (scope, product, path)
        for (scope, product), path in sorted(products.items(), key=order_key)
    ]


def build_catalog() -> str:
    """Build the per-product credit-card spec catalog block.

    Returns the empty string when the KB cannot be located or no qualifying
    products are found — the agent then behaves as baseline.
    """
    docs_dir = _documents_dir()
    if not docs_dir:
        return ""

    sections: dict[str, list[str]] = {"credit_cards": [], "business_credit_cards": []}
    for scope, _product, path in _discover_product_specs(docs_dir):
        doc = _load_doc(path)
        if not doc:
            continue
        title = str(doc.get("title", "")).strip()
        content = str(doc.get("content", "")).strip()
        if not title or not content:
            continue
        if len(content) > _MAX_DOC_CHARS:
            content = content[:_MAX_DOC_CHARS] + " ..."
        sections[scope].append(f"[{title}]\n{content}")

    parts: list[str] = []
    if sections["credit_cards"]:
        parts.append(
            "=== Personal / Consumer Credit Cards ===\n\n"
            + "\n\n".join(sections["credit_cards"])
        )
    if sections["business_credit_cards"]:
        parts.append(
            "=== Business Credit Cards ===\n\n"
            + "\n\n".join(sections["business_credit_cards"])
        )

    if not parts:
        return ""

    header = (
        "<card_product_catalog>\n"
        "Authoritative per-product credit-card spec sheets from the Rho-Bank "
        "knowledge base, retrieved deterministically by document-id prefix "
        "(one Getting Started / Eligibility doc per card product). Use this "
        "catalog as the source of truth for card eligibility, fees, APR, "
        "foreign-transaction fees, credit-limit ranges, base cash-back rates, "
        "and Rho-Bank+ subscription requirements when a customer is choosing, "
        "comparing, switching, or applying for a credit card. Consider every "
        "product the customer could be eligible for; a card you never mention "
        "cannot be chosen. Match the customer's stated requirements against "
        "each candidate's eligibility before recommending it."
    )
    footer = "\n</card_product_catalog>"
    return header + "\n" + "\n\n".join(parts) + footer
