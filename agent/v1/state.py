from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class AgentState:
    stage: str  # NEW | PARSED | ROUTED | ANSWERED | NORMALIZED | BLOCKED | SUBMIT
    question: str
    extras: dict[str, Any] = field(default_factory=dict)
    route: str | None = None  # DIRECT | NEEDS_FILE | NEEDS_RETRIEVAL
    raw_answer: str | None = None
    answer: str | None = None
    block_reason: str | None = None
    finish_reason: str | None = None

    def advance(self, stage: str, **kwargs: Any) -> "AgentState":
        return replace(self, stage=stage, **kwargs)
