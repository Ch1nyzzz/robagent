"""v14 plan_retrieval — raise the visible-token budget from 256 to 1024.

Trace evidence from v15 full-165: 57 of 73 NEEDS_RETRIEVAL tasks hit
`finish_reason=length` on plan_retrieval at MAX_TOKENS=256. 34 of those
ended BLOCKED (planner truncated -> deterministic regex query -> wiki miss
-> parametric UNKNOWN). The reasoning model burns thinking tokens before
emitting the visible JSON; 256 visible tokens is not enough headroom on
long-form questions. v6 used 1024 with no trace-evidence of regression;
v14 reverts the budget to 1024 while keeping the v9 tighter prompt and
deterministic-fallback semantics.
"""
from __future__ import annotations

from typing import Any

from agent.llm import chat, DEFAULT_MODEL
from agent.v9.llm_nodes.plan_retrieval_v9 import (
    SYSTEM_PROMPT,
    _parse_query,
    deterministic_query_from_question,
)


MAX_TOKENS = 1024


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
