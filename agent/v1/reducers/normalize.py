"""Mirror of `bench.gaia.scorer` normalization.

The scorer's `normalize_str` strips whitespace, lowercases, and removes punctuation.
For the numeric branch it strips currency / percent / comma. The agent applies a
soft pre-submit normalization so we drop trailing periods, surrounding quotes, and
explanatory prefixes that the scorer would otherwise reject.
"""
from __future__ import annotations

import re


_PREFIX_RE = re.compile(
    r"^\s*(?:final\s+answer|answer|the\s+answer\s+is|result)\s*[:\-]\s*",
    re.IGNORECASE,
)
_WRAP_QUOTES = ("\"", "'", "`")


def _strip_wrappers(s: str) -> str:
    """Iteratively strip trailing period and matched surrounding quotes.

    Loops until a fixpoint so cases like `"yellow".` reduce to `yellow`.
    """
    prev = None
    while prev != s:
        prev = s
        s = s.strip()
        # strip a single trailing period that is not part of a number
        if s.endswith(".") and not re.search(r"\d\.$", s):
            s = s[:-1].strip()
        # strip one layer of matched surrounding quotes / backticks
        if len(s) >= 2 and s[0] in _WRAP_QUOTES and s[-1] == s[0]:
            s = s[1:-1].strip()
    return s


def normalize_answer(s: str | None) -> str:
    """Pre-submit normalization aligned with the scorer.

    The scorer further normalizes internally, so we keep this conservative:
    only remove the noisy wrappers that empirically cause string-branch mismatches.
    """
    if s is None:
        return ""
    s = s.strip()
    # remove a leading "Answer:" / "Final answer:" prefix if present
    m = _PREFIX_RE.match(s)
    if m:
        s = s[m.end():]
    s = _strip_wrappers(s)
    # collapse internal whitespace runs into single spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s
