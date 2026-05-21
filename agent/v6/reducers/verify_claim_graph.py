"""Deterministic claim-graph verifier.

Given the LLM's claim object and the list of opened sources, return whether
the claim is source-backed. Pure function — no LLM.

Rules:
  - claim must be non-empty and not "UNKNOWN"
  - source_id must be present and match one of the opened source_ids
  - confidence must be >= min_confidence (default 0.3)
"""
from __future__ import annotations

from typing import Any


def verify_claim_graph(
    claim_obj: dict[str, Any],
    opened_source_ids: set[str],
    *,
    min_confidence: float = 0.3,
) -> dict[str, Any]:
    claim = (claim_obj.get("claim") or "").strip()
    src = claim_obj.get("source_id")
    conf = float(claim_obj.get("confidence") or 0.0)

    if not claim or claim.upper() == "UNKNOWN":
        return {"ok": False, "reason": "empty_or_unknown_claim"}
    if not src:
        return {"ok": False, "reason": "missing_source_id"}
    if src not in opened_source_ids:
        return {"ok": False, "reason": "source_id_not_in_opened_sources"}
    if conf < min_confidence:
        return {"ok": False, "reason": "low_confidence"}
    return {"ok": True, "reason": "ok"}
