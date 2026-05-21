"""Deterministic router.

Decides one of {DIRECT, NEEDS_FILE, NEEDS_RETRIEVAL} from task extras + question.
v1 does not implement file or retrieval capabilities — non-DIRECT routes are
handled honestly by emitting a structured BLOCKED event rather than fabricating.

Routing signals are intentionally structural (URL schemes, TLD-like domain
shapes, citation phrasings) so the router stays benchmark-agnostic: no
hardcoded site names appear here.
"""
from __future__ import annotations

import re
from typing import Any


_URL_RE = re.compile(r"https?://", re.IGNORECASE)

# A naked domain like "example.com" or "foo.org" with a TLD suffix from a small
# generic structural list. We deliberately do NOT enumerate site names.
_DOMAIN_RE = re.compile(
    r"\b[a-z0-9][a-z0-9\-]{1,40}\.(?:com|org|net|gov|edu|io|co|uk|de|fr)\b",
    re.IGNORECASE,
)

# Phrases that signal "the answer lives in an external source we must open".
_CITATION_RE = re.compile(
    r"\b(?:according\s+to|as\s+(?:listed|stated|published|reported)\s+(?:on|in|by)|"
    r"in\s+the\s+(?:article|paper|study|video|recording|episode|report)|"
    r"on\s+the\s+website|on\s+the\s+page)\b",
    re.IGNORECASE,
)


def _mentions_source(question: str) -> bool:
    if _URL_RE.search(question):
        return True
    if _DOMAIN_RE.search(question):
        return True
    if _CITATION_RE.search(question):
        return True
    return False


def route_by_extras(question: str, extras: dict[str, Any]) -> str:
    file_name = (extras or {}).get("file_name") or ""
    if file_name:
        return "NEEDS_FILE"
    if _mentions_source(question):
        return "NEEDS_RETRIEVAL"
    return "DIRECT"
