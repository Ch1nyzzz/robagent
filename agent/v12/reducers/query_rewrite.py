"""Deterministic query rewriting.

When the first query returns 0 hits, generate a small set of template
variants — pure regex / token surgery, NO LLM.

Template variants applied (in order, dedup-by-string):

1. Drop quantifier modifiers ("between 2000 and 2009", "as of June 2023", ...)
2. Extract the longest capitalized phrase
3. Take only the first 3-5 capitalized tokens (entity + context)
4. Strip leading question words ("How many", "What", "Which", ...)
5. Take the last noun-phrase-looking span
"""
from __future__ import annotations

import re
from typing import Iterable


_LEAD_Q_RE = re.compile(
    r"^\s*(?:how\s+many|how\s+much|how\s+long|how\s+far|how\s+old|"
    r"what\s+(?:was|is|are|were|will|year|date|day|time|color|number|name|word|"
    r"writer|author|book|movie|album|song)|"
    r"which|where|when|who|why|name\s+the|tell\s+me|find|list|give|"
    r"under|in|on|at|according\s+to|of|the)\b\s*",
    re.IGNORECASE,
)
_MODIFIER_RE = re.compile(
    r"\b(?:between|from|since|as\s+of|in)\s+\S+\s+(?:and\s+\S+\s*)?(?:\d{4}|\(?[\d]{2,4}\)?)\b",
    re.IGNORECASE,
)
_PROPER_NOUN_RE = re.compile(
    r"\b(?:[A-Z][a-zA-Z'’\-]+(?:\s+[A-Z][a-zA-Z'’\-]+)+)\b"
)
_QUOTED_RE = re.compile(r"[\"“]([^\"“”]{4,80})[\"”]")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _variants(question: str, base_query: str) -> Iterable[str]:
    """Yield candidate alternate queries in priority order."""
    base = _norm(base_query)
    if base:
        yield base
    q = _norm(question)
    # 1) Drop modifier clauses
    no_mod = _norm(_MODIFIER_RE.sub("", q))
    if no_mod and no_mod != q:
        yield no_mod[:120]
    # 2) Quoted title is usually the highest-value entity
    qm = _QUOTED_RE.search(q)
    if qm:
        yield _norm(qm.group(1))[:120]
    # 3) Longest capitalized phrase
    spans = _PROPER_NOUN_RE.findall(q)
    spans.sort(key=len, reverse=True)
    for s in spans[:3]:
        yield _norm(s)[:120]
    # 4) Strip leading question word
    no_lead = _norm(_LEAD_Q_RE.sub("", q))
    if no_lead and no_lead != q:
        yield no_lead[:120]
    # 5) Last 8 tokens (the "noun phrase tail")
    tokens = q.split()
    if len(tokens) > 8:
        yield _norm(" ".join(tokens[-8:]))[:120]


def rewrite_query_variants(question: str, base_query: str, *, k: int = 4) -> list[str]:
    """Return up to k unique candidate queries derived from `question` / `base_query`."""
    out: list[str] = []
    seen: set[str] = set()
    for v in _variants(question or "", base_query or ""):
        v = _norm(v)
        if not v:
            continue
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
        if len(out) >= k:
            break
    return out
