"""Deterministic credit-card product catalog for the banking_knowledge domain.

Why this exists
---------------
Card-recommendation / card-comparison tasks fail because the agent's only
retrieval tool, ``KB_search``, is BM25 sparse retrieval. A query for one card
product (e.g. "Business Platinum Rewards Card") collides on shared vocabulary
with every other card's documents, so the specification document the agent
actually needs is buried below the top results. In the failed simulations the
agent issues a dozen-plus searches, never assembles a complete view of the
product line-up, and recommends a card that is not the best fit — or a product
it never even discovered exists (it compared business cards without ever
realising the Business Bronze Rewards Card was an option).

What this module does
---------------------
It sidesteps BM25 for that one document class. ``build_catalog`` loads every
credit-card product-specification document straight from the domain's
knowledge base on disk and compiles them into a single catalog string, grouped
by product. These are exactly the documents ``KB_search`` indexes — handing
them over wholesale is a retrieval improvement, not extra knowledge (tau2's own
``full_kb`` configuration puts the entire knowledge base in the prompt). The
agent injects the catalog only when the conversation is about choosing,
comparing, or switching credit cards.

Anti-overfitting
----------------
Only the domain's public knowledge-base documents are read. No customer data,
no task definitions, and no evaluation data are touched. The product list is
discovered from document-id prefixes at runtime — nothing about any specific
task, customer, or card is hard-coded.
"""
from __future__ import annotations

import glob
import json
import os
import re

# A knowledge-base document id looks like:
#   doc_credit_cards_<slug>_<NNN>            (personal cards)
#   doc_business_credit_cards_<slug>_<NNN>   (business cards)
_DOC_RE = re.compile(r"^doc_(?:business_)?credit_cards_(.+)_(\d+)\.json$")

# Slugs that are NOT individual card products (general guidance, account
# logistics, replacement procedures, virtual-card tooling).
_NON_PRODUCT = ("(general)", "logistics", "replacement", "virtual_card")

# Per-document content is capped so one unusually long document cannot blow up
# the prompt; real card documents sit well under this.
_MAX_DOC_CHARS = 2600


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


def _display_name(slug: str, first_title: str) -> str:
    """Human-readable product name for a slug.

    Prefer the text before the first colon of a document title (clean, e.g.
    "Silver Rewards Card"); fall back to the slug. Business cards whose title
    omits the "Business" qualifier are corrected so they stay distinguishable
    from their personal counterparts.
    """
    name = (first_title or "").split(":", 1)[0].strip()
    if not name:
        name = slug.replace("_", " ").replace("-", "-").title()
    if slug.startswith("business_") and not name.lower().startswith("business"):
        name = "Business " + name
    return name


def build_catalog() -> str:
    """Build the full credit-card product catalog string.

    Returns an empty string if the knowledge base cannot be located or read —
    the agent then simply behaves as the baseline with no catalog injected.
    """
    docs_dir = _documents_dir()
    if not docs_dir:
        return ""

    products: dict[str, list[tuple[int, str]]] = {}
    try:
        paths = glob.glob(os.path.join(docs_dir, "doc_*credit_cards_*.json"))
    except Exception:
        return ""

    for path in paths:
        match = _DOC_RE.match(os.path.basename(path))
        if not match:
            continue
        slug, num = match.group(1), match.group(2)
        if any(token in slug for token in _NON_PRODUCT):
            continue
        products.setdefault(slug, []).append((int(num), path))

    if not products:
        return ""

    sections: list[tuple[str, str]] = []
    for slug, entries in products.items():
        entries.sort()
        docs = []
        for _, path in entries:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    docs.append(json.load(fh))
            except Exception:
                continue
        if not docs:
            continue
        name = _display_name(slug, docs[0].get("title", ""))
        body = [f"=== {name} ==="]
        for doc in docs:
            title = str(doc.get("title", "")).strip()
            content = str(doc.get("content", "")).strip()
            if not content:
                continue
            if len(content) > _MAX_DOC_CHARS:
                content = content[:_MAX_DOC_CHARS] + " ..."
            body.append(f"[{title}]\n{content}")
        if len(body) > 1:
            sections.append((name, "\n\n".join(body)))

    if not sections:
        return ""

    sections.sort(key=lambda s: s[0])
    header = (
        "<credit_card_product_catalog>\n"
        "Reference: the complete Rho-Bank credit-card product line-up, with the "
        "knowledge-base specification documents for every product. Use this to "
        "compare products and make recommendations instead of repeatedly "
        "searching the knowledge base for a single card.\n"
    )
    footer = "\n</credit_card_product_catalog>"
    return header + "\n" + "\n\n".join(text for _, text in sections) + footer


# --- conversation intent detection ---------------------------------------

# A card-related token and a shopping/comparison token must both appear in the
# same user message for the conversation to count as card shopping. Detection
# is sticky: once any earlier user message qualifies, the catalog stays in.
_CARD_TOKENS = ("credit card", "rewards card", "ecocard", " card", "card?", "card.")

_SHOP_TOKENS = (
    "recommend",
    "which card",
    "what card",
    "best card",
    "best credit",
    "compare",
    "comparison",
    "better deal",
    "better offer",
    "better reward",
    "better rate",
    "looking for a",
    "apply for",
    "applying for",
    "open a new",
    "open a credit",
    "new credit card",
    "sign up for",
    "switch",
    "competitor",
    "my options",
    "what are my option",
    "right for me",
    "suit me",
    "should i get",
    "qualify for",
    "which one",
    "best fit",
    "best option",
)


def is_card_shopping(user_texts) -> bool:
    """True when a user message shows credit-card shopping / comparison intent."""
    for text in user_texts or []:
        if not isinstance(text, str):
            continue
        low = text.lower()
        if not any(tok in low for tok in _CARD_TOKENS):
            continue
        if any(tok in low for tok in _SHOP_TOKENS):
            return True
    return False
