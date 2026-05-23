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

A component that emits a non-admitted decision raises
`ComponentPolicyError`. INDUCED_RULE additionally requires a non-empty
`trust.out_of_evidence_probe` (validate_trust).
"""
from __future__ import annotations

from .types import ComponentClass, DecisionKind, Mount, Trust


class ComponentPolicyError(RuntimeError):
    """Raised when a component registration or emitted decision is not permitted."""


_ALLOW   = {DecisionKind.ALLOW}
_INJECT  = {DecisionKind.INJECT_CONTEXT}
_BLOCK   = {DecisionKind.BLOCK}
_REWRITE = {DecisionKind.REWRITE}

ALLOWED: dict[ComponentClass, dict[Mount, set[DecisionKind]]] = {
    ComponentClass.MECHANISM_LAYER: {
        Mount.SESSION_START:      _ALLOW | _INJECT,
        Mount.PRE_PROMPT_BUILD:   _ALLOW | _INJECT | _REWRITE | _BLOCK,
        Mount.POST_LLM_RESPONSE:  _ALLOW | _REWRITE | _BLOCK | _INJECT,
        Mount.PRE_ANSWER_EMIT:    _ALLOW | _REWRITE | _BLOCK,
    },
    ComponentClass.REACTIVE_GUARD: {
        Mount.POST_LLM_RESPONSE:  _ALLOW | _REWRITE | _BLOCK | _INJECT,
        Mount.PRE_ANSWER_EMIT:    _ALLOW | _REWRITE | _BLOCK,
    },
    ComponentClass.CHANNEL: {
        Mount.SESSION_START:      _ALLOW | _INJECT,
        Mount.PRE_PROMPT_BUILD:   _ALLOW | _INJECT,
    },
    ComponentClass.INDUCED_RULE: {
        # Advisory-only: PRE_PROMPT_BUILD inject_context. LLM keeps authority.
        Mount.PRE_PROMPT_BUILD:   _ALLOW | _INJECT,
    },
    # PREDICTIVE_HEURISTIC: absent → load-time rejected.
}


def validate_registration(cls: ComponentClass, mount: Mount) -> None:
    if cls is ComponentClass.PREDICTIVE_HEURISTIC:
        raise ComponentPolicyError(
            "predictive_heuristic is not admissible: matcher tests raw prompt "
            "text, structurally unsafe. Redesign or do not write."
        )
    if mount not in ALLOWED.get(cls, {}):
        raise ComponentPolicyError(
            f"class {cls.value!r} cannot attach to mount {mount.value!r}; "
            f"allowed mounts: {[m.value for m in ALLOWED.get(cls, {}).keys()]}"
        )


def validate_decision(cls: ComponentClass, mount: Mount, kind: DecisionKind) -> None:
    allowed = ALLOWED.get(cls, {}).get(mount, set())
    if kind not in allowed:
        raise ComponentPolicyError(
            f"class {cls.value!r} component on mount {mount.value!r} emitted "
            f"decision {kind.value!r}; allowed: {[k.value for k in allowed]}"
        )


def validate_trust(cls: ComponentClass, trust: Trust) -> None:
    if not trust.evidence_anchor.strip():
        raise ComponentPolicyError(
            "trust.evidence_anchor required: name the stable structure "
            "(system field / LLM API field / general algorithm) outside evidence sims."
        )
    if not trust.blast_radius.strip():
        raise ComponentPolicyError("trust.blast_radius required.")
    if not trust.rollback_when.strip():
        raise ComponentPolicyError("trust.rollback_when required.")
    if cls is ComponentClass.INDUCED_RULE and not trust.out_of_evidence_probe.strip():
        raise ComponentPolicyError(
            "INDUCED_RULE requires trust.out_of_evidence_probe — a concrete OOE "
            "case the matcher fires on, and the handler's return on it. Without "
            "it, the rule is overfit by construction."
        )
