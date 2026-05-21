"""v11 normalize — closes a class of format-drift regressions observed in v10.

Specifically:
- "56, 000" → "56000" before the numeric branch of the scorer sees it
- "1st" / "2nd" / "3rd" → "1" / "2" / "3" when the surrounding shape is numeric
- Full-width digits (０-９) → ASCII digits
- Stray "approximately" / "about" prefixes are stripped
- A trailing scientific unit token (m^3, kg, %, $...) is preserved as-is,
  but only if it directly follows a number — random alphabetic suffixes are
  left alone so we don't corrupt non-numeric answers.

We DO NOT touch list-separator behavior (v3 already handles that).
"""
from __future__ import annotations

import re

from agent.v3.reducers.normalize import normalize_answer as _v3_normalize


_FULLWIDTH_MAP = {chr(0xFF10 + i): str(i) for i in range(10)}
_NUMBER_WITH_INNER_WS_RE = re.compile(r"(?<=\d)[,\s]\s*(?=\d)")
_LEADING_HEDGE_RE = re.compile(
    r"^(?:approximately|approx\.?|about|roughly|around|nearly|some)\s+",
    re.IGNORECASE,
)
_ORDINAL_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)


def _strip_fullwidth_digits(s: str) -> str:
    return s.translate(str.maketrans(_FULLWIDTH_MAP))


def _collapse_numeric_whitespace(s: str) -> str:
    """Turn "56, 000" / "1 000" into "56000" / "1000" — but only when the
    entire token (after stripping wrappers) is a number with internal
    comma/whitespace separators.
    """
    if not s:
        return s
    candidate = _NUMBER_WITH_INNER_WS_RE.sub("", s)
    # Only commit the collapse if the result is a pure numeric token
    if re.fullmatch(r"-?\d+(?:\.\d+)?", candidate):
        return candidate
    return s


def _strip_leading_hedge(s: str) -> str:
    return _LEADING_HEDGE_RE.sub("", s, count=1)


def _normalize_ordinal(s: str) -> str:
    """If the whole string is an ordinal token, strip the suffix."""
    m = _ORDINAL_RE.fullmatch(s.strip())
    if m:
        return m.group(1)
    return s


def normalize_answer(s: str | None) -> str:
    s = _v3_normalize(s)
    if not s:
        return s
    s = _strip_fullwidth_digits(s)
    s = _strip_leading_hedge(s)
    s = _normalize_ordinal(s)
    s = _collapse_numeric_whitespace(s)
    return s.strip()
