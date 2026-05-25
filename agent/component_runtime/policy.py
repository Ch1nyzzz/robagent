"""Class × mount × decision-kind permission matrix for GAIA components.

Five classes; identical risk semantics to the tau2 component runtime:

  * MECHANISM_LAYER  — anchored to off-evidence facts (file extensions,
                        URL syntax, LLM API response fields).
  * REACTIVE_GUARD   — block / rewrite / inject on observed signals
                        (`finish_reason=length`, empty content, LLM error
                        markers like "I cannot answer").
  * CHANNEL          — inject_context only; injects content the agent
                        cannot otherwise reach (file content, URL fetch).
  * INDUCED_RULE     — ADVISORY-ONLY at PRE_PROMPT_BUILD; inject_context
                        only. LLM keeps final authority.
  * PREDICTIVE_HEURISTIC — REJECTED at load time.

Phase A of the event-runtime migration moved the validator logic into
`meta_harness.component_runtime_core.policy`. This module keeps the
GAIA-specific `ALLOWED` matrix as data and wires it into the shared
validators.
"""
from __future__ import annotations

from meta_harness.component_runtime_core.policy import (  # noqa: F401
    ComponentPolicyError,
    validate_decision as _core_validate_decision,
    validate_registration as _core_validate_registration,
    validate_trust,
)

from .types import ComponentClass, DecisionKind, Mount


_ALLOW   = {DecisionKind.ALLOW}
_INJECT  = {DecisionKind.INJECT_CONTEXT}
_BLOCK   = {DecisionKind.BLOCK}
_REWRITE = {DecisionKind.REWRITE}

ALLOWED: dict[ComponentClass, dict[Mount, set[DecisionKind]]] = {
    ComponentClass.MECHANISM_LAYER: {
        Mount.SESSION_START:      _ALLOW | _INJECT,
        Mount.PRE_PROMPT_BUILD:   _ALLOW | _INJECT | _REWRITE | _BLOCK,
        Mount.POST_LLM_RESPONSE:  _ALLOW | _REWRITE | _BLOCK | _INJECT,
        Mount.PRE_ANSWER_EMIT:    _ALLOW | _REWRITE | _BLOCK,
    },
    ComponentClass.REACTIVE_GUARD: {
        Mount.POST_LLM_RESPONSE:  _ALLOW | _REWRITE | _BLOCK | _INJECT,
        Mount.PRE_ANSWER_EMIT:    _ALLOW | _REWRITE | _BLOCK,
    },
    ComponentClass.CHANNEL: {
        Mount.SESSION_START:      _ALLOW | _INJECT,
        Mount.PRE_PROMPT_BUILD:   _ALLOW | _INJECT,
    },
    ComponentClass.INDUCED_RULE: {
        # Advisory-only: PRE_PROMPT_BUILD inject_context. LLM keeps authority.
        Mount.PRE_PROMPT_BUILD:   _ALLOW | _INJECT,
    },
    # PREDICTIVE_HEURISTIC: absent → load-time rejected.
}


def validate_registration(cls: ComponentClass, mount: Mount) -> None:
    _core_validate_registration(cls, mount, ALLOWED)


def validate_decision(cls: ComponentClass, mount: Mount, kind: DecisionKind) -> None:
    _core_validate_decision(cls, mount, kind, ALLOWED)
