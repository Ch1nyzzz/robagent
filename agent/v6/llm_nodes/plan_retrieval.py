"""LLM node: extract a short Wikipedia search query from a question.

Constrained output — single line. The query becomes wikipedia_search args.
"""
from __future__ import annotations

import json
import re
from typing import Any

from agent.llm import chat, DEFAULT_MODEL


SYSTEM_PROMPT = (
    "You convert a benchmark question into ONE short Wikipedia search query "
    "(no quotes, max 8 words). Output a single JSON object: "
    '{"query": "<the search query>"}. '
    "No explanation. No code fences."
)

MAX_TOKENS = 1024


def _parse_query(content: str) -> str:
    if not content:
        return ""
    content = content.strip()
    # try fenced JSON
    m = re.search(r"\{[\s\S]*?\}", content)
    if m:
        try:
            obj = json.loads(m.group(0))
            q = obj.get("query") or ""
            return str(q).strip()
        except Exception:
            pass
    # fallback: first line
    return content.splitlines()[0][:120].strip()


def plan_retrieval(question: str, *, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    resp = chat(messages=messages, model=model, max_tokens=MAX_TOKENS)
    resp["query"] = _parse_query(resp.get("content") or "")
    return resp
