"""Deterministic KB_search redundancy guard for the tau2 banking agent.

KB_search in the banking_knowledge domain returns only the single
highest-ranked knowledge-base document for a query. When the agent cannot find
a detail it tends to re-issue the *same* question with different wording; each
re-wording returns the same single document and burns a conversational turn,
so the agent never assembles the reference data it needs.

This module detects that pattern deterministically. A KB_search is treated as
redundant when its query introduces no *content* token absent from the union
of every KB_search already issued this conversation. Such a query is built
entirely from words already searched, so it cannot surface new information and
is safe to block. A query that introduces a genuinely new term (a different
card name, account class, fee type, policy keyword, ...) is always allowed —
the rule never hides an un-searched topic from the agent.
"""
from __future__ import annotations

import re

# Generic English stopwords only — no domain words, so domain terms such as
# card / account names always count as distinguishing content.
_STOPWORDS = {
    "a", "an", "the", "of", "for", "to", "in", "on", "and", "or", "with",
    "is", "are", "be", "by", "at", "as", "vs", "my", "i", "do", "does",
    "what", "which", "how", "can", "any", "all", "this", "that", "it",
    "from", "about", "into", "per", "if", "you", "your", "me", "we", "have",
}


def _stem(tok: str) -> str:
    """Crude plural folding so 'rate'/'rates' count as the same term."""
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def content_tokens(query: str) -> set[str]:
    """Normalise a query to its set of distinguishing content tokens.

    Lower-cased; punctuation split out; pure-numeric tokens dropped (they are
    values the agent is probing for, not search topics); stopwords dropped;
    plurals folded.
    """
    if not query:
        return set()
    text = str(query).lower().replace("cashback", "cash back")
    out: set[str] = set()
    for tok in re.split(r"[^a-z0-9]+", text):
        if not tok or tok.isdigit():
            continue
        if tok in _STOPWORDS or len(tok) < 2:
            continue
        out.add(_stem(tok))
    return out


class KBSearchGuard:
    """Tracks KB_search queries and flags redundant re-wordings."""

    def __init__(self) -> None:
        self._seen_tokens: set[str] = set()
        self._queries: list[str] = []

    def reset(self) -> None:
        self._seen_tokens.clear()
        self._queries.clear()

    @property
    def num_searches(self) -> int:
        return len(self._queries)

    def is_redundant(self, query: str) -> bool:
        """True when `query` introduces no token not already searched."""
        if not self._queries:
            return False
        toks = content_tokens(query)
        if not toks:
            return True  # empty / all-stopword query carries no new intent
        return toks.issubset(self._seen_tokens)

    def register(self, query: str) -> None:
        """Record a KB_search query that was actually issued."""
        if query is None:
            return
        self._queries.append(str(query))
        self._seen_tokens |= content_tokens(query)

    def guidance(self, query: str) -> str:
        """Re-prompt text shown to the agent when a redundant search is blocked."""
        prior = "; ".join(f'"{q}"' for q in self._queries[-6:])
        return (
            f'You just attempted KB_search(query="{query}"). Every search term '
            "in it has already been used in an earlier KB_search this "
            f"conversation (earlier searches: {prior}). KB_search returns only "
            "the single highest-ranked document, so a query built entirely from "
            "words you have already searched returns a document you have "
            "already seen and cannot surface new information.\n"
            "Do NOT repeat this search. Instead choose ONE of:\n"
            "1. Continue the task using the documents you have already "
            "retrieved.\n"
            "2. KB_search a genuinely DIFFERENT topic, naming a specific card, "
            "account class, fee type, or policy keyword you have NOT searched "
            "yet.\n"
            "3. If every relevant topic has been searched, proceed to the next "
            "action or ask the user for any missing detail.\n"
            "Produce your next turn now."
        )
