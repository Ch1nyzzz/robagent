from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class AgentState:
    stage: str
    question: str
    extras: dict[str, Any] = field(default_factory=dict)
    route: str | None = None
    raw_answer: str | None = None
    answer: str | None = None
    block_reason: str | None = None
    finish_reason: str | None = None
    # v6+: claim graph
    claims: tuple = ()  # tuple of dicts {claim, source_id, source_ref, confidence}
    sources: tuple = ()  # tuple of dicts {source_id, kind, url, content_hash}

    def advance(self, stage: str, **kwargs: Any) -> "AgentState":
        return replace(self, stage=stage, **kwargs)
