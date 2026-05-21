"""v14 answer_direct — retry once with a larger token budget when the model
exhausts its reasoning budget before emitting any visible content.

Trace evidence from v15 full-165: 11 DIRECT tasks ended BLOCKED with
`finish_reason=length` AND empty content (the reasoning model burned the
full 8192-token budget on hidden thinking and never produced an answer).
v14 keeps the primary call at 8192 (cheap), and only on the empty-length
edge case retries once at 16384 to clear the cliff. The retry path emits
the same {content, finish_reason, usage} shape so downstream finalize /
gates see no contract change.
"""
from __future__ import annotations

from typing import Any

from agent.llm import chat, DEFAULT_MODEL
from agent.v1.llm_nodes.answer_direct import SYSTEM_PROMPT


PRIMARY_MAX_TOKENS = 8192
RETRY_MAX_TOKENS = 16384


def _empty_length_truncation(resp: dict[str, Any]) -> bool:
    return (
        (resp.get("finish_reason") == "length")
        and not (resp.get("content") or "").strip()
    )


def answer_direct(question: str, *, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    resp = chat(messages=messages, model=model, max_tokens=PRIMARY_MAX_TOKENS)
    if _empty_length_truncation(resp):
        # One retry at a larger budget; tag the usage so traces show both.
        retry = chat(messages=messages, model=model, max_tokens=RETRY_MAX_TOKENS)
        retry["primary_finish_reason"] = resp.get("finish_reason")
        retry["primary_completion_tokens"] = (
            (resp.get("usage") or {}).get("completion_tokens")
        )
        return retry
    return resp
