"""Class × event × decision-kind permission matrix (toolathlon variant).

Same four classes as tau2; matrix carries v2's relaxations:

  * `pre_tool_use`: MCP tools are wrapped as SDK FunctionTools so REWRITE_TOOL_ARGS
    and true BLOCK are admitted. DEFER requires a replay queue and is rejected.
  * `post_llm_response*` mid-turn rewrite is rejected — the SDK does not
    surface a mid-turn AssistantMessage hook (would need v3 ModelProvider
    wrapping). Subscribers to the post-Runner `post_llm_response_raw`
    event get INJECT_CONTEXT only.

`stop` / `session_end` are reserved (declared, not yet dispatched by
`agent.py`). Allowed at registration so a forward-compatible YAML can
still mention them.
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
        "pre_tool_use":            _ALLOW | _BLOCK | _REWRITE,
        "pre_tool_arg_validation": _ALLOW | _REWRITE | _BLOCK,
        "post_llm_response_raw":   _ALLOW | _INJECT,
        "on_length_truncation":    _ALLOW | _INJECT | _BLOCK,
        "on_empty_response":       _ALLOW | _INJECT | _BLOCK,
        "on_no_tool_call_emitted": _ALLOW | _INJECT | _BLOCK,
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
        "post_llm_response_raw":   _ALLOW | _INJECT,
        "on_length_truncation":    _ALLOW | _INJECT | _BLOCK,
        "on_empty_response":       _ALLOW | _INJECT | _BLOCK,
        "on_no_tool_call_emitted": _ALLOW | _INJECT | _BLOCK,
        "post_tool_use":           _ALLOW | _INJECT,
        "post_tool_result_raw":    _ALLOW | _INJECT,
        "on_tool_error":           _ALLOW | _INJECT,
        "on_explicit_terminate":   _ALLOW | _BLOCK,
        "stop":                    _ALLOW | _BLOCK,
    },
    ComponentClass.INDUCED_RULE: {
        "pre_context_build":       _ALLOW | _INJECT,
        "user_prompt_submit":      _ALLOW | _INJECT,
    },
    # PREDICTIVE_HEURISTIC: absent → load-time rejected.
}


def validate_registration(cls: ComponentClass, event_name: str) -> None:
    _core_validate_registration(cls, event_name, ALLOWED)


def validate_decision(cls: ComponentClass, event_name: str,
                      kind: DecisionKind) -> None:
    _core_validate_decision(cls, event_name, kind, ALLOWED)
