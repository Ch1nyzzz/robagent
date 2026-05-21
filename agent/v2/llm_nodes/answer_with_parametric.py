"""LLM node for retrieval-flagged tasks — answer from parametric memory or emit UNKNOWN."""
from __future__ import annotations

from typing import Any

from agent.llm import chat, DEFAULT_MODEL


SYSTEM_PROMPT = (
    "You are solving a benchmark task that cites an external source. "
    "You do not have web access. If you confidently know the answer from your "
    "parametric memory, output the final answer only (no prefix, no explanation). "
    "If you are not confident, output the single word: UNKNOWN."
)

MAX_TOKENS = 8192


def answer_with_parametric(question: str, *, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return chat(messages=messages, model=model, max_tokens=MAX_TOKENS)
