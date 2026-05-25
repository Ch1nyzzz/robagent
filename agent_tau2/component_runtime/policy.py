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

from typing import Any

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

# Phase D additions: Tier-1 event-name strings join Mount enum keys. The
# `_normalise_key` adapter routes mount.value strings back to the enum so
# legacy validation and event validation hit the same cells.
ALLOWED: dict[ComponentClass, dict[Any, set[DecisionKind]]] = {
    ComponentClass.MECHANISM_LAYER: {
        Mount.PRE_CONTEXT_BUILD:   _ALLOW | _INJECT,
        Mount.SESSION_START:       _ALLOW | _INJECT,
        Mount.USER_PROMPT_SUBMIT:  _ALLOW | _INJECT,
        Mount.PRE_TOOL_USE:        _ALLOW | _REWRITE | _DEFER | _BLOCK,
        Mount.POST_LLM_RESPONSE:   _ALLOW | _REWRITE | _BLOCK | _INJECT,
        Mount.POST_TOOL_USE:       _ALLOW | _INJECT,
        Mount.STOP:                _ALLOW | _BLOCK,
        Mount.SESSION_END:         _ALLOW,
        # Tier-1 events (Phase D):
        "task_received":           _ALLOW | _INJECT,
        "pre_context_build":       _ALLOW | _INJECT,
        "pre_agent_construct":     _ALLOW | _INJECT,
        "pre_llm_request":         _ALLOW | _INJECT,
        "post_llm_response_raw":   _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "on_length_truncation":    _ALLOW | _BLOCK | _INJECT,
        "on_empty_response":       _ALLOW | _BLOCK | _INJECT,
        "on_no_tool_call_emitted": _ALLOW | _BLOCK | _INJECT,
        "pre_tool_arg_validation": _ALLOW | _REWRITE | _BLOCK,
        "post_tool_result_raw":    _ALLOW | _INJECT,
        "on_tool_error":           _ALLOW | _INJECT,
        "on_explicit_terminate":   _ALLOW | _BLOCK,
        "session_end":             _ALLOW,
    },
    ComponentClass.REACTIVE_GUARD: {
        Mount.USER_PROMPT_SUBMIT:  _ALLOW | _INJECT,
        Mount.PRE_TOOL_USE:        _ALLOW | _BLOCK | _REWRITE,
        Mount.POST_LLM_RESPONSE:   _ALLOW | _REWRITE | _BLOCK | _INJECT,
        Mount.POST_TOOL_USE:       _ALLOW | _INJECT,
        Mount.STOP:                _ALLOW | _BLOCK,
        "post_llm_response_raw":   _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "on_length_truncation":    _ALLOW | _BLOCK | _INJECT,
        "on_empty_response":       _ALLOW | _BLOCK | _INJECT,
        "on_no_tool_call_emitted": _ALLOW | _BLOCK | _INJECT,
        "pre_tool_arg_validation": _ALLOW | _REWRITE | _BLOCK,
        "post_tool_result_raw":    _ALLOW | _INJECT,
        "on_tool_error":           _ALLOW | _INJECT,
        "on_explicit_terminate":   _ALLOW | _BLOCK,
    },
    ComponentClass.CHANNEL: {
        Mount.PRE_CONTEXT_BUILD:   _ALLOW | _INJECT,
        Mount.SESSION_START:       _ALLOW | _INJECT,
        Mount.USER_PROMPT_SUBMIT:  _ALLOW | _INJECT,
        "task_received":           _ALLOW | _INJECT,
        "pre_context_build":       _ALLOW | _INJECT,
        "pre_agent_construct":     _ALLOW | _INJECT,
    },
    ComponentClass.INDUCED_RULE: {
        # Advisory-only: the LLM sees the injection and may override.
        Mount.PRE_CONTEXT_BUILD:   _ALLOW | _INJECT,
        Mount.USER_PROMPT_SUBMIT:  _ALLOW | _INJECT,
        "pre_context_build":       _ALLOW | _INJECT,
    },
    # ComponentClass.PREDICTIVE_HEURISTIC intentionally absent → always rejected.
}


def _normalise_key(mount_or_event: Any) -> Any:
    """Round-trip Mount.value strings back to the enum so string-key
    callers (Phase B dispatcher) hit the same matrix cell as Mount-key
    callers (legacy validation)."""
    if isinstance(mount_or_event, Mount):
        return mount_or_event
    if isinstance(mount_or_event, str):
        try:
            return Mount(mount_or_event)
        except ValueError:
            return mount_or_event  # genuine Tier-1 event string
    return mount_or_event


def validate_registration(cls: ComponentClass, mount: Mount) -> None:
    _core_validate_registration(cls, mount, ALLOWED)


def validate_decision(cls: ComponentClass, mount_or_event: Any,
                      kind: DecisionKind) -> None:
    _core_validate_decision(cls, _normalise_key(mount_or_event), kind, ALLOWED)
