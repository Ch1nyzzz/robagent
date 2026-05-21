"""Single LLM node: answer a DIRECT (no retrieval, no file) GAIA question.

Contract:
  input:  question (str)
  output: {content, finish_reason, usage}
  budget: max_tokens >> 2048 — V4-Pro is a reasoning model and burns most of
          its budget on hidden thinking; v0 saw 76/165 truncations at 2048.
"""
from __future__ import annotations

from typing import Any

from agent.llm import chat, DEFAULT_MODEL


SYSTEM_PROMPT = (
    "You are an assistant solving a single benchmark task. "
    "Think step by step internally, but in your final visible response output the "
    "final answer only — no explanation, no prefix, no trailing punctuation. "
    "If the expected answer is a number, output the number only. "
    "If the expected answer is a short string, output that string only. "
    "If the expected answer is a list, output it comma-separated."
)

# V4-Pro reasoning consumes thinking tokens before visible content. 2048 was too
# small (truncated 76/165). 8192 gives the model room for thinking and a clean
# terse final answer.
MAX_TOKENS = 8192


def answer_direct(question: str, *, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return chat(messages=messages, model=model, max_tokens=MAX_TOKENS)
