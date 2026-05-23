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
"""
from __future__ import annotations

from .types import ComponentClass, DecisionKind, Mount, Trust


class ComponentPolicyError(RuntimeError):
    """Raised when a Component registration or emitted decision is not permitted."""


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
        Mount.PRE_TOOL_USE:        _ALLOW | _REWRITE | _DEFER | _BLOCK,
        Mount.POST_LLM_RESPONSE:   _ALLOW | _REWRITE | _BLOCK | _INJECT,
        Mount.POST_TOOL_USE:       _ALLOW | _INJECT,
        Mount.STOP:                _ALLOW | _BLOCK,
    },
    ComponentClass.REACTIVE_GUARD: {
        Mount.USER_PROMPT_SUBMIT:  _ALLOW | _INJECT,
        Mount.PRE_TOOL_USE:        _ALLOW | _BLOCK | _REWRITE,
        Mount.POST_LLM_RESPONSE:   _ALLOW | _REWRITE | _BLOCK | _INJECT,
        Mount.POST_TOOL_USE:       _ALLOW | _INJECT,
        Mount.STOP:                _ALLOW | _BLOCK,
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


def validate_registration(cls: ComponentClass, mount: Mount) -> None:
    """Reject Component registration if its class cannot attach to that mount."""
    if cls is ComponentClass.PREDICTIVE_HEURISTIC:
        raise ComponentPolicyError(
            "predictive_heuristic is not admissible: matcher tests raw prompt "
            "text, which is structurally unsafe even as advisory injection. "
            "Redesign so the matcher tests structure (system field / tool "
            "schema / protocol invariant / general algorithm) or do not write "
            "this component."
        )
    if mount not in ALLOWED.get(cls, {}):
        raise ComponentPolicyError(
            f"class {cls.value!r} cannot attach to mount {mount.value!r}; "
            f"allowed mounts for this class: "
            f"{[m.value for m in ALLOWED.get(cls, {}).keys()]}"
        )


def validate_decision(cls: ComponentClass, mount: Mount, kind: DecisionKind) -> None:
    allowed = ALLOWED.get(cls, {}).get(mount, set())
    if kind not in allowed:
        raise ComponentPolicyError(
            f"class {cls.value!r} component on mount {mount.value!r} emitted "
            f"decision {kind.value!r}; allowed: {[k.value for k in allowed]}"
        )


def validate_trust(cls: ComponentClass, trust: Trust) -> None:
    """Enforce required trust fields per class.

    All classes require non-empty `evidence_anchor`, `blast_radius`,
    `rollback_when`. INDUCED_RULE additionally requires
    `out_of_evidence_probe` — a concrete unseen case the matcher must not
    misfire on. Without it, the rule is a memorised map of the N evidence
    sims (see RESULTS.md §4 for the math).
    """
    if not trust.evidence_anchor.strip():
        raise ComponentPolicyError(
            f"trust.evidence_anchor is required: name the stable structure "
            f"(system field, tool schema, protocol invariant, general algorithm) "
            f"that this component anchors on, OUTSIDE your evidence sims."
        )
    if not trust.blast_radius.strip():
        raise ComponentPolicyError(
            "trust.blast_radius is required: local | workflow | global."
        )
    if not trust.rollback_when.strip():
        raise ComponentPolicyError(
            "trust.rollback_when is required: an observable condition under "
            "which the component is provably dead weight or actively harmful."
        )
    if cls is ComponentClass.INDUCED_RULE and not trust.out_of_evidence_probe.strip():
        raise ComponentPolicyError(
            "INDUCED_RULE requires trust.out_of_evidence_probe — name one "
            "concrete case NOT in your evidence sims where the matcher would "
            "fire, and state what the handler returns on it. If you cannot "
            "construct such a case, the rule is induced from finite evidence "
            "and overfits by construction. Either redesign as MECHANISM_LAYER "
            "/ REACTIVE_GUARD / CHANNEL, or do not write this component."
        )
