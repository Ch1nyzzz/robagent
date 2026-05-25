"""Class × event × decision-kind permission matrix for tau2 components.

Five classes; two of them gated:

  * MECHANISM_LAYER  — most permissive; anchored to off-evidence facts.
  * REACTIVE_GUARD   — block / rewrite / inject; triggers on observed failure.
  * CHANNEL          — inject_context only; injects content otherwise unreachable.
  * INDUCED_RULE     — ADVISORY-ONLY: pre-context / user_prompt_submit +
                       inject_context only.
  * PREDICTIVE_HEURISTIC — REJECTED at load time.

INDUCED_RULE additionally requires `trust.out_of_evidence_probe` to be
non-empty — see `validate_trust`.
"""
from __future__ import annotations

from meta_harness.component_runtime_core.policy import (  # noqa: F401
    ComponentPolicyError,
    validate_decision as _core_validate_decision,
    validate_registration as _core_validate_registration,
    validate_trust,
)

from .types import ComponentClass, DecisionKind


_ALLOW   = {DecisionKind.ALLOW}
_INJECT  = {DecisionKind.INJECT_CONTEXT}
_BLOCK   = {DecisionKind.BLOCK}
_REWRITE = {DecisionKind.REWRITE_TOOL_ARGS}
_DEFER   = {DecisionKind.DEFER}

ALLOWED: dict[ComponentClass, dict[str, set[DecisionKind]]] = {
    ComponentClass.MECHANISM_LAYER: {
        "pre_context_build":       _ALLOW | _INJECT,
        "session_start":           _ALLOW | _INJECT,
        "task_received":           _ALLOW | _INJECT,
        "pre_agent_construct":     _ALLOW | _INJECT,
        "user_prompt_submit":      _ALLOW | _INJECT,
        "pre_llm_request":         _ALLOW | _INJECT,
        "pre_tool_use":            _ALLOW | _REWRITE | _DEFER | _BLOCK,
        "pre_tool_arg_validation": _ALLOW | _REWRITE | _BLOCK,
        "post_llm_response":       _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "post_llm_response_raw":   _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "on_length_truncation":    _ALLOW | _BLOCK | _INJECT,
        "on_empty_response":       _ALLOW | _BLOCK | _INJECT,
        "on_no_tool_call_emitted": _ALLOW | _BLOCK | _INJECT,
        "post_tool_use":           _ALLOW | _INJECT,
        "post_tool_result_raw":    _ALLOW | _INJECT,
        "on_tool_error":           _ALLOW | _INJECT,
        "on_explicit_terminate":   _ALLOW | _BLOCK,
        "stop":                    _ALLOW | _BLOCK,
        "session_end":             _ALLOW,
    },
    ComponentClass.REACTIVE_GUARD: {
        "user_prompt_submit":      _ALLOW | _INJECT,
        "pre_tool_use":            _ALLOW | _BLOCK | _REWRITE,
        "pre_tool_arg_validation": _ALLOW | _REWRITE | _BLOCK,
        "post_llm_response":       _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "post_llm_response_raw":   _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "on_length_truncation":    _ALLOW | _BLOCK | _INJECT,
        "on_empty_response":       _ALLOW | _BLOCK | _INJECT,
        "on_no_tool_call_emitted": _ALLOW | _BLOCK | _INJECT,
        "post_tool_use":           _ALLOW | _INJECT,
        "post_tool_result_raw":    _ALLOW | _INJECT,
        "on_tool_error":           _ALLOW | _INJECT,
        "on_explicit_terminate":   _ALLOW | _BLOCK,
        "stop":                    _ALLOW | _BLOCK,
    },
    ComponentClass.CHANNEL: {
        "pre_context_build":       _ALLOW | _INJECT,
        "session_start":           _ALLOW | _INJECT,
        "task_received":           _ALLOW | _INJECT,
        "pre_agent_construct":     _ALLOW | _INJECT,
        "user_prompt_submit":      _ALLOW | _INJECT,
    },
    ComponentClass.INDUCED_RULE: {
        # Advisory-only: the LLM sees the injection and may override.
        "pre_context_build":       _ALLOW | _INJECT,
        "user_prompt_submit":      _ALLOW | _INJECT,
    },
    # ComponentClass.PREDICTIVE_HEURISTIC intentionally absent → always rejected.
}


def validate_registration(cls: ComponentClass, event_name: str) -> None:
    _core_validate_registration(cls, event_name, ALLOWED)


def validate_decision(cls: ComponentClass, event_name: str,
                      kind: DecisionKind) -> None:
    _core_validate_decision(cls, event_name, kind, ALLOWED)
