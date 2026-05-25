"""Class × mount × decision-kind permission matrix.

Five classes; two of them gated:

  * MECHANISM_LAYER  — most permissive; anchored to off-evidence facts.
  * REACTIVE_GUARD   — block / rewrite / inject; triggers on observed failure.
  * CHANNEL          — inject_context only; injects content otherwise unreachable.
  * INDUCED_RULE     — ADVISORY-ONLY: PRE_CONTEXT_BUILD / USER_PROMPT_SUBMIT +
                       inject_context only. The LLM keeps final authority.
  * PREDICTIVE_HEURISTIC — REJECTED at load time. Matcher tests raw prompt
                       text; even advisory injection is structurally unsafe.

A component that emits a Decision its (class, mount) cell is not authorised
to make raises `ComponentPolicyError`; the runtime treats this as a hard
bug, not a fallback.

INDUCED_RULE also requires `trust.out_of_evidence_probe` to be non-empty —
see `validate_trust` below. Without a concrete OOE case, the component is
overfit by construction.

Phase A of the event-runtime migration moved the validator logic into
`meta_harness.component_runtime_core.policy`. This module keeps the
tau2-specific `ALLOWED` matrix as data and wires it into the shared
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


_ALLOW = {DecisionKind.ALLOW}
_INJECT = {DecisionKind.INJECT_CONTEXT}
_BLOCK = {DecisionKind.BLOCK}
_REWRITE = {DecisionKind.REWRITE_TOOL_ARGS}
_DEFER = {DecisionKind.DEFER}

ALLOWED: dict[ComponentClass, dict[Mount, set[DecisionKind]]] = {
    ComponentClass.MECHANISM_LAYER: {
        Mount.PRE_CONTEXT_BUILD:   _ALLOW | _INJECT,
        Mount.SESSION_START:       _ALLOW | _INJECT,
        Mount.USER_PROMPT_SUBMIT:  _ALLOW | _INJECT,
        Mount.PRE_TOOL_USE:        _ALLOW | _REWRITE | _DEFER | _BLOCK,
        Mount.POST_LLM_RESPONSE:   _ALLOW | _REWRITE | _BLOCK | _INJECT,
        Mount.POST_TOOL_USE:       _ALLOW | _INJECT,
        Mount.STOP:                _ALLOW | _BLOCK,
    },
    ComponentClass.REACTIVE_GUARD: {
        Mount.USER_PROMPT_SUBMIT:  _ALLOW | _INJECT,
        Mount.PRE_TOOL_USE:        _ALLOW | _BLOCK | _REWRITE,
        Mount.POST_LLM_RESPONSE:   _ALLOW | _REWRITE | _BLOCK | _INJECT,
        Mount.POST_TOOL_USE:       _ALLOW | _INJECT,
        Mount.STOP:                _ALLOW | _BLOCK,
    },
    ComponentClass.CHANNEL: {
        Mount.PRE_CONTEXT_BUILD:   _ALLOW | _INJECT,
        Mount.SESSION_START:       _ALLOW | _INJECT,
        Mount.USER_PROMPT_SUBMIT:  _ALLOW | _INJECT,
    },
    ComponentClass.INDUCED_RULE: {
        # Advisory-only: the LLM sees the injection and may override.
        # No PRE_TOOL_USE / POST_LLM_RESPONSE — induced rules cannot
        # mechanically rewrite or block. No SESSION_START — the rule
        # has not seen the task yet.
        Mount.PRE_CONTEXT_BUILD:   _ALLOW | _INJECT,
        Mount.USER_PROMPT_SUBMIT:  _ALLOW | _INJECT,
    },
    # ComponentClass.PREDICTIVE_HEURISTIC intentionally absent → always rejected.
}


def validate_registration(cls: ComponentClass, mount: Mount) -> None:
    _core_validate_registration(cls, mount, ALLOWED)


def validate_decision(cls: ComponentClass, mount: Mount, kind: DecisionKind) -> None:
    _core_validate_decision(cls, mount, kind, ALLOWED)
