"""Class × event × decision-kind permission matrix for SOP-Bench components.

Mirror of `agent/component_runtime/policy.py` adapted to the SOP-Bench FC
loop events. The five durability classes preserve the same semantics:

  * MECHANISM_LAYER — anchored to off-evidence facts.
  * REACTIVE_GUARD  — fire only on observed failure signals.
  * INDUCED_RULE    — advisory-only at pre_context_build / pre_prompt_build.
  * PREDICTIVE_HEURISTIC — load-time rejected.

The matrix is keyed by `(class, event_name)` where `event_name` is the
string the dispatcher routes on (= `Component.listens`).
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
_REWRITE = {DecisionKind.REWRITE}

ALLOWED: dict[ComponentClass, dict[str, set[DecisionKind]]] = {
    ComponentClass.MECHANISM_LAYER: {
        "session_start":           _ALLOW | _INJECT,
        "task_received":           _ALLOW | _INJECT,
        "pre_prompt_build":        _ALLOW | _INJECT | _REWRITE | _BLOCK,
        "pre_context_build":       _ALLOW | _INJECT | _REWRITE | _BLOCK,
        "pre_agent_construct":     _ALLOW | _INJECT,
        "pre_llm_turn":            _ALLOW | _REWRITE | _BLOCK,
        "pre_llm_request":         _ALLOW | _INJECT,
        "post_llm_response":       _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "post_llm_response_raw":   _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "on_length_truncation":    _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "on_empty_response":       _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "on_no_tool_call_emitted": _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "pre_tool_arg_validation": _ALLOW | _REWRITE | _BLOCK,
        "pre_tool_use":            _ALLOW | _REWRITE | _BLOCK,
        "post_tool_use":           _ALLOW | _REWRITE,
        "post_tool_result_raw":    _ALLOW | _REWRITE,
        "on_tool_error":           _ALLOW | _REWRITE | _INJECT,
        "on_explicit_terminate":   _ALLOW | _BLOCK,
        "pre_final_emit":          _ALLOW | _REWRITE | _BLOCK,
        "session_end":             _ALLOW,
    },
    ComponentClass.REACTIVE_GUARD: {
        "post_llm_response":       _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "post_llm_response_raw":   _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "on_length_truncation":    _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "on_empty_response":       _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "on_no_tool_call_emitted": _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "pre_tool_arg_validation": _ALLOW | _REWRITE | _BLOCK,
        "pre_tool_use":            _ALLOW | _REWRITE | _BLOCK,
        "post_tool_use":           _ALLOW | _REWRITE,
        "post_tool_result_raw":    _ALLOW | _REWRITE,
        "on_tool_error":           _ALLOW | _REWRITE | _INJECT,
        "on_explicit_terminate":   _ALLOW | _BLOCK,
        "pre_final_emit":          _ALLOW | _REWRITE | _BLOCK,
    },
    ComponentClass.INDUCED_RULE: {
        "pre_prompt_build":        _ALLOW | _INJECT,
        "pre_context_build":       _ALLOW | _INJECT,
    },
    # PREDICTIVE_HEURISTIC: absent → load-time rejected.
}


def validate_registration(cls: ComponentClass, event_name: str) -> None:
    _core_validate_registration(cls, event_name, ALLOWED)


def validate_decision(cls: ComponentClass, event_name: str,
                      kind: DecisionKind) -> None:
    _core_validate_decision(cls, event_name, kind, ALLOWED)
