"""Extended deterministic retrieval of banking_knowledge credit-card spec sheets.

Why this exists
---------------
iter3's ``card_spec_catalog`` loads each card product's ``_001.json`` Getting
Started / Eligibility doc. That doc carries eligibility, fees, APR, foreign-
transaction fee, credit-limit range, and the BASE cash-back rate (the rate that
applies outside top categories). For most products the BONUS-category rate
(e.g. Silver Rewards Card's 4% on travel & software, Platinum's 4% bonus,
Business Silver's 10% on travel & software) lives in the next sequential doc —
``_002.json`` — which the KB's bm25 ``KB_search`` channel does not reliably
surface against natural-language recommendation queries. A customer whose
spending is concentrated in a bonus category cannot be served the right card
without that doc.

What this module does
---------------------
Walks the banking_knowledge KB documents directory and loads BOTH ``_001.json``
and ``_002.json`` for every credit-card product (consumer and business),
excluding the cross-cutting procedure / feature groups under
``doc_(business_)?credit_cards_`` that are not card products
(``credit_cards_(general)``, ``credit_card_account_logistics``,
``credit_card_replacements``, ``virtual_card_management``). The pair is
compiled into one catalog block grouped Personal vs. Business; the LLM still
decides which card matches the customer's needs.

Stable structure captured
-------------------------
Two KB-infrastructure facts independent of any specific simulation:

  1. Document ids partition by product prefix:
     ``doc_(business_)?credit_cards_<product>_<NNN>.json``. The set of card-
     product prefixes is enumerable from disk; a small explicit set of non-
     product prefixes excludes the cross-cutting groups.
  2. Sequential authoring convention: ``_001.json`` is the Getting Started /
     Eligibility spec sheet; ``_002.json`` is the first follow-on doc for the
     product and, for almost every revenue-bearing card, the bonus-category
     earning doc. Both suffixes are deterministic ordering properties of the
     KB layout, not properties of any task.

On a freshly authored KB written under the same conventions, the selection
yields the same class of documents with no reference to any specific failure.
"""
from __future__ import annotations

import glob
import json
import os
import re

# Prefixes under ``doc_(business_)?credit_cards_`` that are NOT card products.
_NON_PRODUCT_SUFFIXES: frozenset[str] = frozenset(
    {
        "credit_cards_(general)",
        "credit_card_account_logistics",
        "credit_card_replacements",
        "virtual_card_management",
    }
)

# Suffixes we load per product. _001 = Getting Started / Eligibility (base
# rate, fees, APR, fxn fee, credit-limit range, Rho-Bank+ requirement). _002 =
# typically the first follow-on / bonus-rate doc.
_LOAD_SUFFIXES: tuple[str, ...] = ("001", "002")

_PRODUCT_RE = re.compile(
    r"^doc_(?P<scope>business_credit_cards|credit_cards)_(?P<product>.+?)_(?P<idx>\d{3})\.json$"
)

_MAX_DOC_CHARS = 2200


def _documents_dir() -> str | None:
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


def _discover_product_docs(
    docs_dir: str,
) -> list[tuple[str, str, list[str]]]:
    """Return [(scope, product, [doc_paths_in_load_order])] for every credit-card
    product that has at least the ``_001.json`` spec sheet on disk. Missing
    ``_002.json`` (or any other configured suffix) is silently skipped for that
    product.
    """
    try:
        entries = sorted(os.listdir(docs_dir))
    except Exception:
        return []

    # (scope, product) -> {idx: path}
    by_product: dict[tuple[str, str], dict[str, str]] = {}
    for fname in entries:
        m = _PRODUCT_RE.match(fname)
        if not m:
            continue
        product = m.group("product")
        if product in _NON_PRODUCT_SUFFIXES:
            continue
        idx = m.group("idx")
        if idx not in _LOAD_SUFFIXES:
            continue
        scope = m.group("scope")
        by_product.setdefault((scope, product), {})[idx] = os.path.join(
            docs_dir, fname
        )

    out: list[tuple[str, str, list[str]]] = []
    for (scope, product), idx_map in by_product.items():
        if "001" not in idx_map:
            # require Getting Started doc as anchor for the product
            continue
        paths = [idx_map[s] for s in _LOAD_SUFFIXES if s in idx_map]
        out.append((scope, product, paths))

    # Stable ordering: consumer cards first, then business cards, alpha within.
    out.sort(key=lambda t: (0 if t[0] == "credit_cards" else 1, t[1]))
    return out


def _render_doc(doc: dict) -> str | None:
    title = str(doc.get("title", "")).strip()
    content = str(doc.get("content", "")).strip()
    if not title or not content:
        return None
    if len(content) > _MAX_DOC_CHARS:
        content = content[:_MAX_DOC_CHARS] + " ..."
    return f"[{title}]\n{content}"


def build_catalog() -> str:
    """Build the per-product credit-card spec catalog block.

    Returns the empty string when the KB cannot be located or no qualifying
    products are found — the agent then behaves as baseline.
    """
    docs_dir = _documents_dir()
    if not docs_dir:
        return ""

    sections: dict[str, list[str]] = {
        "credit_cards": [],
        "business_credit_cards": [],
    }
    for scope, _product, paths in _discover_product_docs(docs_dir):
        rendered_for_product: list[str] = []
        for p in paths:
            doc = _load_doc(p)
            if not doc:
                continue
            r = _render_doc(doc)
            if r:
                rendered_for_product.append(r)
        if rendered_for_product:
            sections[scope].append("\n\n".join(rendered_for_product))

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
        "<card_product_catalog_rates>\n"
        "Authoritative per-product credit-card spec sheets from the Rho-Bank "
        "knowledge base, retrieved deterministically by document-id prefix. "
        "For each card product, both the Getting Started / Eligibility doc "
        "(_001 — base rate, annual fee, APR, foreign-transaction fee, "
        "credit-limit range, Rho-Bank+ subscription requirement) AND the "
        "follow-on rate / earning doc (_002 — bonus-category cash-back rate "
        "for most products) are included so the LLM can match the customer's "
        "spending pattern AND eligibility to the right product. Use this "
        "catalog as the source of truth when a customer is choosing, "
        "comparing, switching, or applying for a credit card. Consider both "
        "the base rate AND bonus-category rate when the customer describes "
        "where their spending is concentrated; also weigh annual fee, "
        "personal vs. business eligibility, and any Rho-Bank+ requirement "
        "before recommending. A card you never mention cannot be chosen."
    )
    footer = "\n</card_product_catalog_rates>"
    return header + "\n" + "\n\n".join(parts) + footer
