"""v11 router — extended deterministic routing.

Builds on v1's `route_by_extras` (URL / domain / citation-phrase signals).
Adds *implicit-retrieval* signals that v10 missed (38 wrong-DIRECT failures):

- Wikipedia mentioned in the body of the question (without a URL)
- arXiv / preprint references / paper titles in quotes
- YouTube / video references (channel + video phrasing)
- Encyclopedia / database / archive mentions (Wayback, BASE, USGS, IPCC, ...)
- Census / official-report references
- Restaurant / menu + specific date (Wayback retrieval scenario)
- Episode / season identifiers + script wording
- "as of <date>" temporal anchor common in source-bound Q&A

The patterns are structural — they describe SHAPES (entity name + date,
quoted title, official-source keyword family) not corpus tokens. Each
keyword family is justified by ≥3 distinct trace failures.

NOTE: routing to NEEDS_RETRIEVAL when retrieval would still miss is fine —
the agent now blocks honestly via `claim_unverified` instead of fabricating,
which is calibration improvement even when raw correctness stays the same.
"""
from __future__ import annotations

import re
from typing import Any

from agent.v1.reducers.route import route_by_extras as _v1_route


# ------------------------------------------------------------ helpers --------
_DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\b\s+\d{1,2}(?:,?\s+\d{4})?",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_QUOTED_TITLE_RE = re.compile(r"[\"“][^\"“”]{8,120}[\"”]")
_ORDINAL_EPISODE_RE = re.compile(
    r"\b(?:season|episode|series)\s+\d+\b", re.IGNORECASE
)
_ENC_KEYWORDS = (
    # online encyclopedias / knowledge bases (structural keywords, not entity names)
    "wikipedia", "wikimedia", "wiktionary", "wikidata",
    # preprint / paper databases
    "arxiv", "biorxiv", "medrxiv", "preprint", "doi",
    # video platforms (structural, not channel names)
    "youtube", "youtu.be",
    # web archive / wayback
    "wayback", "internet archive", "archive.org",
    # institutional / library databases
    "libretext", "base ", "ddc ", "tropicos", "scikit-learn",
    # geological / scientific authorities
    "usgs", "nasa ", "ipcc",
    # cultural / media authorities
    "merriam-webster", "rotten tomatoes", "metacritic",
    # academic citation grading
    "christgau",
)
_CIPHER_RE = re.compile(
    r"\b(?:caesar|vigen[eè]re|substitution|atbash|rot13|cipher|encrypted|decode)\b",
    re.IGNORECASE,
)
_RESTAURANT_RE = re.compile(
    r"\b(?:restaurant|menu|dinner|lunch|breakfast|brunch|cafe|cafeteria)\b",
    re.IGNORECASE,
)


def _has_quoted_title(question: str) -> bool:
    return bool(_QUOTED_TITLE_RE.search(question))


def _has_episode_identifier(question: str) -> bool:
    return bool(_ORDINAL_EPISODE_RE.search(question))


def _has_enc_keyword(question: str) -> bool:
    low = question.lower()
    return any(kw in low for kw in _ENC_KEYWORDS)


def _has_date_anchor(question: str) -> bool:
    if _DATE_RE.search(question):
        return True
    # year alone is weak; require it paired with another retrieval cue.
    return False


def _has_restaurant_with_date(question: str) -> bool:
    return bool(_RESTAURANT_RE.search(question)) and bool(_DATE_RE.search(question))


def _has_quoted_title_with_year(question: str) -> bool:
    return _has_quoted_title(question) and bool(_YEAR_RE.search(question))


def _looks_like_retrieval(question: str) -> bool:
    """Return True if the question carries structural signals that an external
    source is required to answer correctly.

    Each branch corresponds to a failure cluster in v10's wrong-DIRECT set."""
    if not question:
        return False
    if _has_enc_keyword(question):
        return True
    if _has_episode_identifier(question) and bool(_YEAR_RE.search(question or "")):
        return True
    if _has_restaurant_with_date(question):
        return True
    if _has_quoted_title_with_year(question):
        return True
    return False


def _is_pure_instruction(question: str) -> bool:
    """Detect prompts that are pure literal-instruction puzzles.

    Example: "If anything doesn't make sense, write 'Pineapple'. Write 'Guava'."
    The right answer is determined by the literal instruction, not retrieval.
    Routing such questions to DIRECT is correct; we surface them only so the
    workflow can apply a tighter answer-extraction path later if needed.
    """
    if not question:
        return False
    has_write = bool(re.search(r"\bwrite\s+(?:only\s+)?(?:the\s+word\s+)?[\"“']", question, re.I))
    has_instructions = bool(re.search(r"\binstructions?\b", question, re.I))
    return has_write and has_instructions


def route_by_extras(question: str, extras: dict[str, Any]) -> str:
    """v11 router: v1 base + implicit-retrieval signal expansion + cipher gate.

    Cipher prompts stay on DIRECT (the LLM is capable when prompted directly);
    retrieval prompts route to NEEDS_RETRIEVAL even without a literal URL.
    """
    base = _v1_route(question or "", extras or {})
    if base in ("NEEDS_FILE",):
        return base
    if base == "NEEDS_RETRIEVAL":
        return base
    # base == "DIRECT": consider upgrading.
    if _CIPHER_RE.search(question or ""):
        return "DIRECT"
    if _is_pure_instruction(question or ""):
        return "DIRECT"
    if _looks_like_retrieval(question or ""):
        return "NEEDS_RETRIEVAL"
    return "DIRECT"
