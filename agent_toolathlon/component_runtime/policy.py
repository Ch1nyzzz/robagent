"""Class × mount × decision-kind permission matrix (toolathlon variant).

Same five classes as tau2; matrix carries v2's relaxations:

  * `PRE_TOOL_USE`: MCP tools are wrapped as SDK FunctionTools (the v2
    path, now mandatory — the v1 no-args fallback was removed during
    the event-runtime cleanup), so REWRITE_TOOL_ARGS and true BLOCK
    are admitted. `DEFER` requires a replay queue and is still
    rejected (v2.5).
  * `POST_LLM_RESPONSE`: SDK does not emit a mid-turn AssistantMessage
    hook carrying its `tool_calls`. The sub-LLM verifier pattern that
    POST_LLM_RESPONSE was designed for is therefore not implementable
    via lifecycle hooks; the entire mount column is rejected. v3 would
    require wrapping the ModelProvider — out of scope for v2.

`STOP` / `SESSION_END` are also reserved (declared in Mount, not yet
dispatched by `agent.py`). They are allowed at registration so a
forward-compatible YAML can still mention them.

A component that emits a Decision its (class, mount) cell is not
authorised to make raises `ComponentPolicyError`; the runtime treats
this as a hard bug, not a fallback.

INDUCED_RULE also requires `trust.out_of_evidence_probe` to be non-empty
— see `validate_trust` below.

Phase A of the event-runtime migration moved the validator logic into
`meta_harness.component_runtime_core.policy`. This module keeps the
toolathlon-specific `ALLOWED` matrix + DISABLED_MOUNTS_V1 list and
wraps the core validators.
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

# Phase D additions: Tier-1 event-name string keys for the events
# toolathlon DOES emit post-Runner (task_received / pre_context_build /
# pre_agent_construct / post_llm_response_raw / on_length_truncation /
# on_empty_response / on_explicit_terminate / session_end). The events
# that live INSIDE the SDK Runner (pre_llm_request, pre_tool_arg_validation,
# post_tool_result_raw, on_tool_error, on_no_tool_call_emitted) are NOT
# emitted by v1 — they need a ModelProvider wrap. Still declared in the
# matrix so forward-compatible component YAMLs can mention them.
ALLOWED: dict[ComponentClass, dict[Any, set[DecisionKind]]] = {
    ComponentClass.MECHANISM_LAYER: {
        Mount.PRE_CONTEXT_BUILD:   _ALLOW | _INJECT,
        Mount.SESSION_START:       _ALLOW | _INJECT,
        Mount.USER_PROMPT_SUBMIT:  _ALLOW | _INJECT,
        Mount.PRE_TOOL_USE:        _ALLOW | _BLOCK | _REWRITE,
        Mount.POST_TOOL_USE:       _ALLOW | _INJECT,
        Mount.STOP:                _ALLOW | _BLOCK,
        Mount.SESSION_END:         _ALLOW,
        # Tier-1 events emitted by task_agent.py (Phase D):
        "task_received":           _ALLOW | _INJECT,
        "pre_context_build":       _ALLOW | _INJECT,
        "pre_agent_construct":     _ALLOW | _INJECT,
        "post_llm_response_raw":   _ALLOW | _INJECT,
        "on_length_truncation":    _ALLOW | _INJECT | _BLOCK,
        "on_empty_response":       _ALLOW | _INJECT | _BLOCK,
        "on_explicit_terminate":   _ALLOW | _BLOCK,
        "session_end":             _ALLOW,
        # Tier-1 events declared but not emitted in v1 (SDK Runner gap):
        "pre_llm_request":         _ALLOW | _INJECT,
        "pre_tool_arg_validation": _ALLOW | _REWRITE | _BLOCK,
        "post_tool_result_raw":    _ALLOW | _INJECT,
        "on_tool_error":           _ALLOW | _INJECT,
        "on_no_tool_call_emitted": _ALLOW | _INJECT | _BLOCK,
    },
    ComponentClass.REACTIVE_GUARD: {
        Mount.USER_PROMPT_SUBMIT:  _ALLOW | _INJECT,
        Mount.PRE_TOOL_USE:        _ALLOW | _BLOCK | _REWRITE,
        Mount.POST_TOOL_USE:       _ALLOW | _INJECT,
        Mount.STOP:                _ALLOW | _BLOCK,
        Mount.SESSION_END:         _ALLOW,
        "post_llm_response_raw":   _ALLOW | _INJECT,
        "on_length_truncation":    _ALLOW | _INJECT | _BLOCK,
        "on_empty_response":       _ALLOW | _INJECT | _BLOCK,
        "on_explicit_terminate":   _ALLOW | _BLOCK,
        "pre_tool_arg_validation": _ALLOW | _REWRITE | _BLOCK,
        "post_tool_result_raw":    _ALLOW | _INJECT,
        "on_tool_error":           _ALLOW | _INJECT,
        "on_no_tool_call_emitted": _ALLOW | _INJECT | _BLOCK,
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

# Mounts that are entirely off-limits in v1 (SDK doesn't surface the
# required hook). Declared in Mount enum so YAML can still mention them
# for forward compatibility, but registration is rejected if used.
DISABLED_MOUNTS_V1: frozenset[Mount] = frozenset({Mount.POST_LLM_RESPONSE})


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
    """toolathlon-specific extension: reject `DISABLED_MOUNTS_V1` (v2 SDK
    gap) before delegating to the shared core validator."""
    if mount in DISABLED_MOUNTS_V1:
        raise ComponentPolicyError(
            f"mount {mount.value!r} is not dispatched in toolathlon v1 "
            f"(SDK does not surface the required hook). See policy.py."
        )
    _core_validate_registration(cls, mount, ALLOWED)


def validate_decision(cls: ComponentClass, mount_or_event: Any,
                      kind: DecisionKind) -> None:
    _core_validate_decision(cls, _normalise_key(mount_or_event), kind, ALLOWED)
