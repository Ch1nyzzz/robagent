"""Class × event × decision-kind permission matrix.

Three low-risk classes only. The two HIGH-risk classes from
robust-harness-tau2 (`induced_rule` and `predictive_heuristic`) are
deleted by design — see `types.HookClass` for the rationale.

  * CHANNEL only injects content the model otherwise cannot reach.
  * REACTIVE_GUARD may block / rewrite (it triggers on observed failures).
  * DETERMINISTIC_GLUE may rewrite / defer, but not block silently
    (a deterministic always-on block would override the LLM without a
    failure signal — turn it into REACTIVE_GUARD if that is what you want).

A hook that emits a Decision its class is not authorised to make raises
`HookPolicyError`; the runtime treats this as a hard bug, not a fallback.
"""
from __future__ import annotations

from .types import DecisionKind, HookClass, HookEvent


class HookPolicyError(RuntimeError):
    """Raised when a hook's emitted decision is not permitted by its class."""


_ALLOW = {DecisionKind.ALLOW}

ALLOWED: dict[HookClass, dict[HookEvent, set[DecisionKind]]] = {
    HookClass.CHANNEL: {
        HookEvent.SESSION_START:      _ALLOW | {DecisionKind.INJECT_CONTEXT},
        HookEvent.USER_PROMPT_SUBMIT: _ALLOW | {DecisionKind.INJECT_CONTEXT},
    },
    HookClass.REACTIVE_GUARD: {
        HookEvent.PRE_TOOL_USE: _ALLOW | {DecisionKind.BLOCK,
                                          DecisionKind.REWRITE_TOOL_ARGS},
        HookEvent.POST_TOOL_USE: _ALLOW | {DecisionKind.INJECT_CONTEXT},
        HookEvent.USER_PROMPT_SUBMIT: _ALLOW | {DecisionKind.INJECT_CONTEXT},
        HookEvent.STOP: _ALLOW | {DecisionKind.BLOCK},
    },
    HookClass.DETERMINISTIC_GLUE: {
        HookEvent.PRE_TOOL_USE: _ALLOW | {DecisionKind.REWRITE_TOOL_ARGS,
                                          DecisionKind.DEFER},
        HookEvent.POST_TOOL_USE: _ALLOW | {DecisionKind.INJECT_CONTEXT},
        HookEvent.SESSION_START: _ALLOW | {DecisionKind.INJECT_CONTEXT},
    },
}


def validate_registration(cls: HookClass, event: HookEvent) -> None:
    """Reject hook registration if its class cannot attach to that event."""
    if event not in ALLOWED.get(cls, {}):
        raise HookPolicyError(
            f"class {cls.value!r} cannot attach to event {event.value!r}; "
            f"allowed events for this class: "
            f"{[e.value for e in ALLOWED.get(cls, {}).keys()]}"
        )


def validate_decision(cls: HookClass, event: HookEvent, kind: DecisionKind) -> None:
    allowed = ALLOWED.get(cls, {}).get(event, set())
    if kind not in allowed:
        raise HookPolicyError(
            f"class {cls.value!r} hook on event {event.value!r} emitted "
            f"decision {kind.value!r}; allowed: {[k.value for k in allowed]}"
        )
