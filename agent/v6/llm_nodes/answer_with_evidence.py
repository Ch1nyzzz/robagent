"""LLM node: produce a (claim, source) answer given retrieved evidence.

Output must be a JSON object:
  {"claim": "<final answer>", "source_id": "<id>", "source_quote": "<short>",
   "confidence": 0.0..1.0}

The deterministic claim-graph verifier downstream rejects answers where
source_id is missing or doesn't match any source.opened event.
"""
from __future__ import annotations

import json
import re
from typing import Any

from agent.llm import chat, DEFAULT_MODEL


SYSTEM_PROMPT = (
    "You answer a benchmark question using ONLY the provided evidence snippets. "
    "Output a single JSON object (no code fences): "
    '{"claim": "<terse final answer only>", '
    '"source_id": "<exactly one of the provided S_* ids>", '
    '"source_quote": "<<=30 word quote from that source supporting the claim>", '
    '"confidence": <0.0 to 1.0>}. '
    "If the evidence does not support an answer, output "
    '{"claim": "UNKNOWN", "source_id": null, "source_quote": "", "confidence": 0.0}. '
    "The claim must be the answer only — no prefix, no explanation, no trailing punctuation. "
    "If a number is expected, output a number. If a short string is expected, output that string."
)

MAX_TOKENS = 4096


def _build_evidence_block(sources: list[dict[str, Any]]) -> str:
    parts = []
    for s in sources:
        if not s.get("ok"):
            continue
        parts.append(
            f"--- {s['source_id']} ({s.get('kind','web')}: {s.get('title') or s['url']}) ---\n"
            f"{s['content'][:6000]}"
        )
    return "\n\n".join(parts) or "(no evidence available)"


def _parse_response(content: str) -> dict[str, Any]:
    if not content:
        return {"claim": "", "source_id": None, "source_quote": "", "confidence": 0.0}
    m = re.search(r"\{[\s\S]*\}", content)
    if not m:
        return {"claim": "", "source_id": None, "source_quote": "", "confidence": 0.0}
    try:
        obj = json.loads(m.group(0))
        return {
            "claim": str(obj.get("claim") or "").strip(),
            "source_id": obj.get("source_id"),
            "source_quote": str(obj.get("source_quote") or "")[:300],
            "confidence": float(obj.get("confidence") or 0.0),
        }
    except Exception:
        return {"claim": "", "source_id": None, "source_quote": "", "confidence": 0.0}


def answer_with_evidence(
    question: str,
    sources: list[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    evidence = _build_evidence_block(sources)
    user = f"Question:\n{question}\n\nEvidence:\n{evidence}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    resp = chat(messages=messages, model=model, max_tokens=MAX_TOKENS)
    parsed = _parse_response(resp.get("content") or "")
    resp.update(parsed)
    return resp
