"""Class × mount × decision-kind permission matrix for SOP-Bench components.

Mirror of `agent/component_runtime/policy.py` adapted to the SOP-Bench FC
loop mounts. The five durability classes preserve the same semantics:

  * MECHANISM_LAYER — anchored to off-evidence facts (LLM API fields,
                       tool schemas, SOP procedure structure).
  * REACTIVE_GUARD  — fire only on observed failure signals (format
                       mismatch, tool error, malformed tool args).
  * CHANNEL         — inject-context only at session start / prompt
                       build (e.g., load SOP excerpt, fetch reference).
  * INDUCED_RULE    — advisory-only at PRE_PROMPT_BUILD. LLM keeps
                       final authority.
  * PREDICTIVE_HEURISTIC — load-time rejected.

Phase A of the event-runtime migration moved the validator logic into
`meta_harness.component_runtime_core.policy`. This module keeps the
SOP-Bench-specific `ALLOWED` matrix as data and wires it into the shared
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


_ALLOW   = {DecisionKind.ALLOW}
_INJECT  = {DecisionKind.INJECT_CONTEXT}
_BLOCK   = {DecisionKind.BLOCK}
_REWRITE = {DecisionKind.REWRITE}

# The matrix accepts both Mount enum keys (legacy mount-style components)
# and string keys (Tier-1 event-name subscribers introduced in Phase D).
# `_normalise_key` round-trips Mount.value strings back to the enum so a
# component with `listens="post_llm_response"` and one with
# `mount=Mount.POST_LLM_RESPONSE` hit the same matrix cell. Tier-1 event
# names that have no Mount equivalent fall through to the string key.
ALLOWED: dict[ComponentClass, dict[Any, set[DecisionKind]]] = {
    ComponentClass.MECHANISM_LAYER: {
        Mount.SESSION_START:      _ALLOW | _INJECT,
        Mount.PRE_PROMPT_BUILD:   _ALLOW | _INJECT | _REWRITE | _BLOCK,
        Mount.PRE_LLM_TURN:       _ALLOW | _REWRITE | _BLOCK,
        Mount.POST_LLM_RESPONSE:  _ALLOW | _REWRITE | _BLOCK | _INJECT,
        Mount.PRE_TOOL_USE:       _ALLOW | _REWRITE | _BLOCK,
        Mount.POST_TOOL_USE:      _ALLOW | _REWRITE,
        Mount.PRE_FINAL_EMIT:     _ALLOW | _REWRITE | _BLOCK,
        Mount.SESSION_END:        _ALLOW,
        # Tier-1 events (Phase D):
        "task_received":           _ALLOW | _INJECT,
        "pre_context_build":       _ALLOW | _INJECT | _REWRITE | _BLOCK,
        "pre_agent_construct":     _ALLOW | _INJECT,
        "pre_llm_request":         _ALLOW | _INJECT,
        "post_llm_response_raw":   _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "on_length_truncation":    _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "on_empty_response":       _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "on_no_tool_call_emitted": _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "pre_tool_arg_validation": _ALLOW | _REWRITE | _BLOCK,
        "post_tool_result_raw":    _ALLOW | _REWRITE,
        "on_tool_error":           _ALLOW | _REWRITE | _INJECT,
        "on_explicit_terminate":   _ALLOW | _BLOCK,
        "session_end":             _ALLOW,
    },
    ComponentClass.REACTIVE_GUARD: {
        Mount.POST_LLM_RESPONSE:  _ALLOW | _REWRITE | _BLOCK | _INJECT,
        Mount.PRE_TOOL_USE:       _ALLOW | _REWRITE | _BLOCK,
        Mount.POST_TOOL_USE:      _ALLOW | _REWRITE,
        Mount.PRE_FINAL_EMIT:     _ALLOW | _REWRITE | _BLOCK,
        # Tier-1 events: react on observed failure signals.
        "post_llm_response_raw":   _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "on_length_truncation":    _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "on_empty_response":       _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "on_no_tool_call_emitted": _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "pre_tool_arg_validation": _ALLOW | _REWRITE | _BLOCK,
        "post_tool_result_raw":    _ALLOW | _REWRITE,
        "on_tool_error":           _ALLOW | _REWRITE | _INJECT,
        "on_explicit_terminate":   _ALLOW | _BLOCK,
    },
    ComponentClass.CHANNEL: {
        Mount.SESSION_START:      _ALLOW | _INJECT,
        Mount.PRE_PROMPT_BUILD:   _ALLOW | _INJECT,
        # Tier-1 events: CHANNEL is inject-only at setup.
        "task_received":          _ALLOW | _INJECT,
        "pre_context_build":      _ALLOW | _INJECT,
        "pre_agent_construct":    _ALLOW | _INJECT,
    },
    ComponentClass.INDUCED_RULE: {
        # Advisory-only: PRE_PROMPT_BUILD inject_context. LLM keeps authority.
        Mount.PRE_PROMPT_BUILD:   _ALLOW | _INJECT,
        "pre_context_build":      _ALLOW | _INJECT,
    },
    # PREDICTIVE_HEURISTIC: absent → load-time rejected.
}


def _normalise_key(mount_or_event: Any) -> Any:
    """Round-trip a Mount.value string back to the enum so string-key
    callers (Phase B dispatcher passing `event_name: str`) hit the same
    matrix cell as Mount-key callers (legacy registration validation)."""
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
