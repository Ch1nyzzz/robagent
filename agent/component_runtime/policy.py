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
# The wrapper validators below normalise the firing-time `mount_or_event`
# argument: Mount.value strings round-trip back to the Mount enum so a
# component with `listens="post_llm_response"` hits the same matrix cell
# as one declared via `mount=Mount.POST_LLM_RESPONSE`. Tier-1 event names
# without a Mount equivalent (task_received, on_length_truncation, ...)
# fall through to the string key below.
ALLOWED: dict[ComponentClass, dict[Any, set[DecisionKind]]] = {
    ComponentClass.MECHANISM_LAYER: {
        Mount.SESSION_START:      _ALLOW | _INJECT,
        Mount.PRE_PROMPT_BUILD:   _ALLOW | _INJECT | _REWRITE | _BLOCK,
        Mount.POST_LLM_RESPONSE:  _ALLOW | _REWRITE | _BLOCK | _INJECT,
        Mount.PRE_ANSWER_EMIT:    _ALLOW | _REWRITE | _BLOCK,
        Mount.SESSION_END:        _ALLOW,
        # Tier-1 events (Phase D additions):
        "task_received":          _ALLOW | _INJECT,
        "pre_context_build":      _ALLOW | _INJECT | _REWRITE | _BLOCK,
        "pre_agent_construct":    _ALLOW | _INJECT,
        "pre_llm_request":        _ALLOW | _INJECT,
        "post_llm_response_raw":  _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "on_length_truncation":   _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "on_empty_response":      _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "session_end":            _ALLOW,
    },
    ComponentClass.REACTIVE_GUARD: {
        Mount.POST_LLM_RESPONSE:  _ALLOW | _REWRITE | _BLOCK | _INJECT,
        Mount.PRE_ANSWER_EMIT:    _ALLOW | _REWRITE | _BLOCK,
        # Tier-1 events: REACTIVE_GUARD reacts strongest on post-LLM signals,
        # specifically the synthesised length / empty failure-mode events.
        "post_llm_response_raw":  _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "on_length_truncation":   _ALLOW | _REWRITE | _BLOCK | _INJECT,
        "on_empty_response":      _ALLOW | _REWRITE | _BLOCK | _INJECT,
    },
    ComponentClass.CHANNEL: {
        Mount.SESSION_START:      _ALLOW | _INJECT,
        Mount.PRE_PROMPT_BUILD:   _ALLOW | _INJECT,
        # Tier-1 events: CHANNEL is inject-only by definition.
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
    matrix cell as Mount-key callers (legacy registration validation).
    """
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
