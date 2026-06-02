"""Class × event × decision-kind permission matrix for GAIA components.

Five classes; identical risk semantics to the tau2 component runtime:

  * MECHANISM_LAYER  — anchored to off-evidence facts (file extensions,
                        URL syntax, LLM API response fields).
  * REACTIVE_GUARD   — block / rewrite / inject on observed signals
                        (`finish_reason=length`, empty content, LLM error
                        markers like "I cannot answer").
  * INDUCED_RULE     — ADVISORY-ONLY at pre_context_build / pre_prompt_build;
                        inject_context only. LLM keeps final authority.
  * PREDICTIVE_HEURISTIC — REJECTED at load time.

The matrix is keyed by `(class, event_name)` where `event_name` is the
string the dispatcher routes on (= `Component.listens`). The validator
logic lives in `ballast.component_runtime_core.policy`; this module
just supplies the GAIA-specific ALLOWED data.
"""
from __future__ import annotations

from ballast.component_runtime_core.policy import (  # noqa: F401
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

# `pre_context_build` and `pre_prompt_build` are aliases for the same
# dispatch phase (the legacy mount.value vs the cross-sibling Tier-1
# vocabulary); both keys are admitted so components can subscribe to
# either name. Same story for `post_llm_response` / `post_llm_response_raw`.
ALLOWED: dict[ComponentClass, dict[str, set[DecisionKind]]] = {
    ComponentClass.MECHANISM_LAYER: {
        "session_start":          _ALLOW | _INJECT,
        "task_received":          _ALLOW | _INJECT,
        "pre_prompt_build":       _ALLOW | _INJECT | _REWRITE | _BLOCK,
        "pre_context_build":      _ALLOW | _INJECT | _REWRITE | _BLOCK,
        "pre_agent_construct":    _ALLOW | _INJECT,
        "pre_llm_request":        _ALLOW | _INJECT,
        "post_llm_response":      _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "post_llm_response_raw":  _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "on_length_truncation":   _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "on_empty_response":      _ALLOW | _REWRITE | _BLOCK | _INJECT,
        # Per-tool events. BLOCK at pre_tool_use skips just this tool call.
        # REWRITE payload is dict (pre_tool_use) or str (post_tool_use).
        # INJECT at post_tool_use concatenates into ctx.current_tool_result.
        "pre_tool_use":           _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "post_tool_use":          _ALLOW | _REWRITE | _INJECT,
        "on_tool_error":          _ALLOW | _INJECT,
        "pre_answer_emit":        _ALLOW | _REWRITE | _BLOCK,
        "session_end":            _ALLOW,
    },
    ComponentClass.REACTIVE_GUARD: {
        # REACTIVE_GUARD fires on observed failure signals; primarily the
        # synthesised post-LLM and post-tool failure-mode events.
        "post_llm_response":      _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "post_llm_response_raw":  _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "on_length_truncation":   _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "on_empty_response":      _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "pre_tool_use":           _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "post_tool_use":          _ALLOW | _REWRITE | _INJECT,
        "on_tool_error":          _ALLOW | _INJECT,
        "pre_answer_emit":        _ALLOW | _REWRITE | _BLOCK,
    },
    ComponentClass.INDUCED_RULE: {
        # Advisory-only: pre-context inject_context. The LLM keeps final
        # authority; an induced rule cannot mechanically rewrite or block.
        "pre_prompt_build":       _ALLOW | _INJECT,
        "pre_context_build":      _ALLOW | _INJECT,
    },
    # PREDICTIVE_HEURISTIC: absent → load-time rejected by core validator.
}


def validate_registration(cls: ComponentClass, event_name: str) -> None:
    _core_validate_registration(cls, event_name, ALLOWED)


def validate_decision(cls: ComponentClass, event_name: str,
                      kind: DecisionKind) -> None:
    _core_validate_decision(cls, event_name, kind, ALLOWED)
