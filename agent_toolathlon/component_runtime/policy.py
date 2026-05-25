"""Class × mount × decision-kind permission matrix (toolathlon variant).

Same five classes as tau2; matrix carries v2's relaxations:

  * `PRE_TOOL_USE`: with COMPONENT_WRAP_TOOLS=1 (cr default) MCP tools are
    wrapped as SDK FunctionTools and we see the real arguments before
    the invocation — so REWRITE_TOOL_ARGS and true BLOCK are now
    admitted. `DEFER` requires a replay queue and is still rejected (v2.5).
  * `POST_LLM_RESPONSE`: SDK does not emit a mid-turn AssistantMessage
    hook carrying its `tool_calls`. The sub-LLM verifier pattern that
    POST_LLM_RESPONSE was designed for is therefore not implementable
    via lifecycle hooks; the entire mount column is rejected. v3 would
    require wrapping the ModelProvider — out of scope for v2.

If a candidate runs with COMPONENT_WRAP_TOOLS=0 (legacy v1 path), the
PRE_TOOL_USE REWRITE_TOOL_ARGS decisions admitted here will SILENTLY NOT
FIRE — the v1 AgentHooks dispatcher has no args. Use wrap mode for any
component that depends on REWRITE_TOOL_ARGS.

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
        Mount.PRE_TOOL_USE:        _ALLOW | _BLOCK | _REWRITE,  # v2: REWRITE via tool wrapping; DEFER reserved for v2.5
        Mount.POST_TOOL_USE:       _ALLOW | _INJECT,
        Mount.STOP:                _ALLOW | _BLOCK,
        Mount.SESSION_END:         _ALLOW,
    },
    ComponentClass.REACTIVE_GUARD: {
        Mount.USER_PROMPT_SUBMIT:  _ALLOW | _INJECT,
        Mount.PRE_TOOL_USE:        _ALLOW | _BLOCK | _REWRITE,  # v2: REWRITE via tool wrapping
        Mount.POST_TOOL_USE:       _ALLOW | _INJECT,
        Mount.STOP:                _ALLOW | _BLOCK,
        Mount.SESSION_END:         _ALLOW,
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

# Mounts that are entirely off-limits in v1 (SDK doesn't surface the
# required hook). Declared in Mount enum so YAML can still mention them
# for forward compatibility, but registration is rejected if used.
DISABLED_MOUNTS_V1: frozenset[Mount] = frozenset({Mount.POST_LLM_RESPONSE})


def validate_registration(cls: ComponentClass, mount: Mount) -> None:
    """toolathlon-specific extension: reject `DISABLED_MOUNTS_V1` (v2 SDK
    gap) before delegating to the shared core validator."""
    if mount in DISABLED_MOUNTS_V1:
        raise ComponentPolicyError(
            f"mount {mount.value!r} is not dispatched in toolathlon v1 "
            f"(SDK does not surface the required hook). See policy.py."
        )
    _core_validate_registration(cls, mount, ALLOWED)


def validate_decision(cls: ComponentClass, mount: Mount, kind: DecisionKind) -> None:
    _core_validate_decision(cls, mount, kind, ALLOWED)
