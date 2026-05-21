"""v3 normalize — extends v1 with list-separator normalization.

If a string looks like a comma- or semicolon-separated list, collapse the
separator format to ", " (single comma + space) so the scorer's elementwise
branch sees a clean shape.
"""
from __future__ import annotations

import re

from agent.v1.reducers.normalize import normalize_answer as _v1_normalize


_LIST_SEP_RE = re.compile(r"\s*[;,]\s*")


def normalize_answer(s: str | None) -> str:
    s = _v1_normalize(s)
    if not s:
        return s
    # If the string contains a comma or semicolon, reshape into clean comma-space format.
    if any(c in s for c in ",;"):
        parts = [p for p in _LIST_SEP_RE.split(s) if p]
        s = ", ".join(parts)
    return s
