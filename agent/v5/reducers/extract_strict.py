"""v5 strict extractor — same patterns as v1 but also strips a trailing
explanation block introduced by a double newline.

Many V4-Pro responses, when they exceed the prompt budget, end with
"<answer>\\n\\nExplanation: ..." — keep only the part before the first
double newline.
"""
from __future__ import annotations

import re

from agent.v1.reducers.extract import extract_final_answer as _v1_extract


_DOUBLE_NL = re.compile(r"\n\s*\n")


def extract_final_answer_strict(content: str | None) -> str:
    if not content:
        return ""
    # Trim everything past the first double-newline block.
    trimmed = _DOUBLE_NL.split(content, maxsplit=1)[0]
    return _v1_extract(trimmed)
