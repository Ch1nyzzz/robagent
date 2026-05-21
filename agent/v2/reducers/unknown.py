"""Detect when an LLM response indicates it has no parametric knowledge."""
from __future__ import annotations

import re

# Phrases the model emits when it cannot answer from parametric knowledge.
_UNKNOWN_PATTERNS = (
    re.compile(r"^\s*unknown\s*$", re.IGNORECASE),
    re.compile(r"\bi\s+(?:do\s+not|don't|cannot|can't)\s+(?:know|determine)\b", re.IGNORECASE),
    re.compile(r"\binsufficient\s+(?:information|context)\b", re.IGNORECASE),
    re.compile(r"\bunable\s+to\s+(?:answer|determine)\b", re.IGNORECASE),
    re.compile(r"\bnot\s+enough\s+information\b", re.IGNORECASE),
)


def is_unknown_response(s: str | None) -> bool:
    if not s:
        return True
    txt = s.strip()
    if not txt:
        return True
    for p in _UNKNOWN_PATTERNS:
        if p.search(txt):
            return True
    return False
