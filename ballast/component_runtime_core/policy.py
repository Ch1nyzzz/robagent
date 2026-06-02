"""Generic policy validators (Phase A of event-runtime migration).

Per-sibling `ALLOWED` matrices stay in each sibling's `policy.py` because
the (Mount, DecisionKind) keyspace genuinely differs (e.g. tau2 admits
REWRITE_TOOL_ARGS / DEFER which gaia does not). These validators take
the sibling's matrix as a data argument so the rule itself can be
shared:

  * validate_registration(cls, mount, allowed_map)
        load-time check that (cls, mount) is admissible
  * validate_decision(cls, mount_or_event, kind, allowed_map)
        fire-time check that the handler returned an admitted decision
  * validate_trust(cls, trust)
        load-time check that the Trust block is non-empty + that
        INDUCED_RULE includes an out_of_evidence_probe

PREDICTIVE_HEURISTIC is rejected at load time across every sibling — its
matcher tests raw text and is structurally unsafe.
"""
from __future__ import annotations

from typing import Mapping, Any

from .shared_types import ComponentClass, Trust


class ComponentPolicyError(RuntimeError):
    """Raised when a component registration or emitted decision is not permitted."""


def validate_registration(
    cls: ComponentClass,
    mount: Any,
    allowed: Mapping[ComponentClass, Mapping[Any, set[Any]]],
) -> None:
    """Load-time guard. `mount` is the per-sibling Mount enum value (or
    a Tier-1 event-name string after Phase B); `allowed` is the
    per-sibling ALLOWED matrix."""
    if cls is ComponentClass.PREDICTIVE_HEURISTIC:
        raise ComponentPolicyError(
            "predictive_heuristic is not admissible: matcher tests raw "
            "prompt text, structurally unsafe. Redesign or do not write."
        )
    if mount not in allowed.get(cls, {}):
        admitted = [getattr(m, "value", m) for m in allowed.get(cls, {}).keys()]
        raise ComponentPolicyError(
            f"class {cls.value!r} cannot attach to mount {getattr(mount, 'value', mount)!r}; "
            f"allowed mounts: {admitted}"
        )


def validate_decision(
    cls: ComponentClass,
    mount: Any,
    kind: Any,
    allowed: Mapping[ComponentClass, Mapping[Any, set[Any]]],
) -> None:
    """Fire-time guard. Raises if the handler returned a Decision whose
    kind is not in the (cls, mount)-cell of the per-sibling matrix."""
    cell = allowed.get(cls, {}).get(mount, set())
    if kind not in cell:
        raise ComponentPolicyError(
            f"class {cls.value!r} component on mount "
            f"{getattr(mount, 'value', mount)!r} emitted decision "
            f"{getattr(kind, 'value', kind)!r}; "
            f"allowed: {[getattr(k, 'value', k) for k in cell]}"
        )


def validate_trust(cls: ComponentClass, trust: Trust) -> None:
    """Load-time guard. Required fields must be non-empty. INDUCED_RULE
    additionally requires `out_of_evidence_probe` — the rule is
    overfit-by-construction without it."""
    if not trust.evidence_anchor.strip():
        raise ComponentPolicyError(
            "trust.evidence_anchor required: name the stable structure "
            "(system field / tool schema / SDK property / general algorithm) "
            "outside evidence sims."
        )
    if not trust.blast_radius.strip():
        raise ComponentPolicyError("trust.blast_radius required.")
    if not trust.rollback_when.strip():
        raise ComponentPolicyError("trust.rollback_when required.")
    if cls is ComponentClass.INDUCED_RULE and not trust.out_of_evidence_probe.strip():
        raise ComponentPolicyError(
            "INDUCED_RULE requires trust.out_of_evidence_probe — a concrete "
            "OOE case the matcher fires on, and the handler's return on it. "
            "Without it, the rule is overfit by construction."
        )
