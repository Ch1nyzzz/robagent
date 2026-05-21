"""Extract a terse final answer from an LLM response.

Most v0 LLM outputs that are non-empty are already terse one-liners
(prompt enforces "final answer only"). When the model added scaffolding,
we look for an `Answer:` / `Final answer:` / boxed pattern.
"""
from __future__ import annotations

import re


_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_ANSWER_LINE_RE = re.compile(
    r"(?im)^\s*(?:final\s+answer|answer|result)\s*[:\-]\s*(.+?)\s*$",
)
_THE_ANSWER_IS_RE = re.compile(
    r"(?im)\bthe\s+(?:final\s+)?answer\s+is\s*[:\-]?\s*(.+?)\s*(?:[.\n]|$)",
)


def extract_final_answer(content: str | None) -> str:
    if not content:
        return ""
    c = content.strip()
    m = _BOXED_RE.search(c)
    if m:
        return m.group(1).strip()
    # Last "Final answer: X" line wins (scaffolded reasoning)
    matches = list(_ANSWER_LINE_RE.finditer(c))
    if matches:
        return matches[-1].group(1).strip()
    m2 = _THE_ANSWER_IS_RE.search(c)
    if m2:
        return m2.group(1).strip()
    # Fall back to last non-empty line — the prompt asks for terse output,
    # so most of the time the whole content is the answer.
    for line in reversed(c.splitlines()):
        s = line.strip()
        if s:
            return s
    return c
