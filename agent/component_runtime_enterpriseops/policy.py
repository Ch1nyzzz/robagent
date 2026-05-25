"""Class × mount × decision-kind permission matrix for EnterpriseOps components.

Mirror of `agent/component_runtime_sopbench/policy.py`. The five durability
classes preserve identical semantics:

  * MECHANISM_LAYER — anchored to off-evidence facts (LLM API fields,
                       MCP tool schemas, fixed protocol structure).
  * REACTIVE_GUARD  — fire only on observed failure signals (format
                       mismatch, MCP tool error, malformed tool args).
  * CHANNEL         — inject-context only at session start / prompt
                       build (e.g., load gym-server schema description).
  * INDUCED_RULE    — advisory-only at PRE_PROMPT_BUILD. LLM keeps
                       final authority.
  * PREDICTIVE_HEURISTIC — load-time rejected.

Phase A of the event-runtime migration moved the validator logic into
`meta_harness.component_runtime_core.policy`. This module keeps the
EnterpriseOps-specific `ALLOWED` matrix as data and wires it into the
shared validators.
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
        Mount.PRE_LLM_TURN:       _ALLOW | _REWRITE | _BLOCK,
        Mount.POST_LLM_RESPONSE:  _ALLOW | _REWRITE | _BLOCK | _INJECT,
        Mount.PRE_TOOL_USE:       _ALLOW | _REWRITE | _BLOCK,
        Mount.POST_TOOL_USE:      _ALLOW | _REWRITE,
        Mount.PRE_FINAL_EMIT:     _ALLOW | _REWRITE | _BLOCK,
    },
    ComponentClass.REACTIVE_GUARD: {
        Mount.POST_LLM_RESPONSE:  _ALLOW | _REWRITE | _BLOCK | _INJECT,
        Mount.PRE_TOOL_USE:       _ALLOW | _REWRITE | _BLOCK,
        Mount.POST_TOOL_USE:      _ALLOW | _REWRITE,
        Mount.PRE_FINAL_EMIT:     _ALLOW | _REWRITE | _BLOCK,
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
