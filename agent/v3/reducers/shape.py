"""Infer expected answer shape from question; reshape LLM output accordingly.

Shapes (structural, benchmark-agnostic):
  - "numeric"   when the question asks "how many" / "what number" / "what percent"
  - "list"      when the question asks for items / names / values (plural cue)
  - "single"    default — a single short token or phrase
"""
from __future__ import annotations

import re


_NUMERIC_HINTS = re.compile(
    r"\b(?:how\s+many|how\s+much|what\s+(?:number|percent|percentage)|what\s+is\s+the\s+sum|"
    r"count|total\s+number\s+of|average|mean|median)\b",
    re.IGNORECASE,
)

_LIST_HINTS = re.compile(
    r"\b(?:list|names?|titles?|values?|items?|all\s+the|which\s+ones?|which\s+(?:are|were))\b",
    re.IGNORECASE,
)


def infer_answer_shape(question: str) -> str:
    if _NUMERIC_HINTS.search(question):
        return "numeric"
    if _LIST_HINTS.search(question):
        return "list"
    return "single"


_NUMERIC_TOKEN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def reshape_answer(answer: str, shape: str) -> str:
    """Coerce the answer towards the inferred shape, but never invent content."""
    if not answer:
        return answer
    if shape == "numeric":
        # If there's a single number anywhere in the answer, prefer it.
        m = _NUMERIC_TOKEN.search(answer)
        if m:
            return m.group(0)
    # For list / single shapes the v3 normalize already does the right thing.
    return answer
