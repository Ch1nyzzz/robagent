"""v9 plan_retrieval — tighter prompt, deterministic fallback.

If the LLM JSON parse fails, extract the longest proper-noun span from the
question as a deterministic backup query (much better than emitting the
whole question, which produces 0 wiki hits).
"""
from __future__ import annotations

import json
import re
from typing import Any

from agent.llm import chat, DEFAULT_MODEL


SYSTEM_PROMPT = (
    "Convert the question into a Wikipedia search query (3-6 words). "
    "Pick the SINGLE most-specific entity (person, place, work, organization) "
    "the question is about. "
    'Respond with ONLY the JSON: {"query":"<words>"}. No code fences. No prose.'
)

MAX_TOKENS = 256


_PROPER_NOUN_RE = re.compile(
    r"\b(?:[A-Z][a-zA-Z'’\-]+(?:\s+[A-Z][a-zA-Z'’\-]+)+)\b"
)
_CAPITAL_WORD_RE = re.compile(r"\b[A-Z][a-zA-Z'’\-]{2,}\b")


def deterministic_query_from_question(q: str) -> str:
    """Pick the longest capitalized multi-word phrase, else longest single capital word."""
    if not q:
        return ""
    spans = _PROPER_NOUN_RE.findall(q)
    if spans:
        spans.sort(key=len, reverse=True)
        return spans[0][:80]
    singles = _CAPITAL_WORD_RE.findall(q)
    # filter out common sentence-starters like What, Which, How
    stop = {"What", "Which", "How", "When", "Where", "Why", "Who", "The", "Is",
            "Are", "Does", "Did", "Will", "Can", "In", "On", "At", "According", "If"}
    singles = [s for s in singles if s not in stop]
    if singles:
        singles.sort(key=len, reverse=True)
        return singles[0][:60]
    return q[:80]


def _parse_query(content: str) -> str:
    if not content:
        return ""
    m = re.search(r"\{[\s\S]*?\}", content)
    if m:
        try:
            obj = json.loads(m.group(0))
            q = str(obj.get("query") or "").strip()
            if q and len(q) <= 100:
                return q
        except Exception:
            pass
    # try a "query: X" line
    m2 = re.search(r"(?i)query\s*[:=]\s*(.+)", content)
    if m2:
        return m2.group(1).strip().strip('"').strip("'")[:80]
    # fallback: first short line
    for line in content.splitlines():
        s = line.strip()
        if 3 < len(s) <= 80:
            return s
    return ""


def plan_retrieval(question: str, *, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    resp = chat(messages=messages, model=model, max_tokens=MAX_TOKENS)
    q = _parse_query(resp.get("content") or "")
    if not q or len(q) > 100:
        q = deterministic_query_from_question(question)
    resp["query"] = q
    return resp
