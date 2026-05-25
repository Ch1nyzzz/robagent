"""Class × mount × decision-kind permission matrix for EnterpriseOps components.

Mirror of `agent/component_runtime_sopbench/policy.py`. The five durability
classes preserve identical semantics:

  * MECHANISM_LAYER — anchored to off-evidence facts (LLM API fields,
                       MCP tool schemas, fixed protocol structure).
  * REACTIVE_GUARD  — fire only on observed failure signals (format
                       mismatch, MCP tool error, malformed tool args).
  * CHANNEL         — inject-context only at session start / prompt
                       build (e.g., load gym-server schema description).
  * INDUCED_RULE    — advisory-only at PRE_PROMPT_BUILD. LLM keeps
                       final authority.
  * PREDICTIVE_HEURISTIC — load-time rejected.

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
        Mount.PRE_LLM_TURN:       _ALLOW | _REWRITE | _BLOCK,
        Mount.POST_LLM_RESPONSE:  _ALLOW | _REWRITE | _BLOCK | _INJECT,
        Mount.PRE_TOOL_USE:       _ALLOW | _REWRITE | _BLOCK,
        Mount.POST_TOOL_USE:      _ALLOW | _REWRITE,
        Mount.PRE_FINAL_EMIT:     _ALLOW | _REWRITE | _BLOCK,
    },
    ComponentClass.REACTIVE_GUARD: {
        Mount.POST_LLM_RESPONSE:  _ALLOW | _REWRITE | _BLOCK | _INJECT,
        Mount.PRE_TOOL_USE:       _ALLOW | _REWRITE | _BLOCK,
        Mount.POST_TOOL_USE:      _ALLOW | _REWRITE,
        Mount.PRE_FINAL_EMIT:     _ALLOW | _REWRITE | _BLOCK,
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
            "(LLM API field / MCP tool schema / fixed protocol) outside evidence sims."
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
