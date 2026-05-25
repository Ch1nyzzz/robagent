"""Truly cross-sibling enums + dataclasses.

These three types are 5/5 identical across the gaia / tau2 / toolathlon /
sopbench / enterpriseops runtimes (only docstring drift). Verified by:

    grep -A 8 '^class Trust' agent/component_runtime/types.py agent_tau2/component_runtime/types.py agent_toolathlon/component_runtime/types.py agent/component_runtime_sopbench/types.py agent/component_runtime_enterpriseops/types.py
    grep -A 5 '^class StateScope' ...
    grep -A 10 '^class ComponentClass' ...

Mount / DecisionKind / Capability / Decision / ComponentContext / Component
stay per-sibling because they LEGITIMATELY differ (different lifecycle
vocabularies, different decision kinds, different mount-specific ctx
fields, different capability sets). See the package docstring for the
full split.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ComponentClass(str, Enum):
    """Durability class admissible into a Component.

    The matcher's predicate decides the class — you do not assert it:

      * MECHANISM_LAYER: tests a system field, a tool-declared schema, a
        protocol invariant, or a general algorithm. Off-evidence facts.
      * REACTIVE_GUARD: tests an observed failure event (tool.failed,
        malformed call, empty/looping turn, finish_reason=length).
        Self-disables when the model stops failing.
      * CHANNEL: injects content the agent cannot otherwise reach
        (file contents, URL fetch result, advisory grammar).
        Inject-only; never overrides the LLM.
      * INDUCED_RULE: a policy/instruction reading compiled from N
        evidence rows. Admitted only as ADVISORY-ONLY inject_context.
        Requires non-empty trust.out_of_evidence_probe.
      * PREDICTIVE_HEURISTIC: tests raw prompt/response text via
        regex or keyword guess. REJECTED at load time — structurally
        unsafe (overfits to seen text).
    """
    MECHANISM_LAYER       = "mechanism_layer"
    REACTIVE_GUARD        = "reactive_guard"
    CHANNEL               = "channel"
    INDUCED_RULE          = "induced_rule"
    PREDICTIVE_HEURISTIC  = "predictive_heuristic"


class StateScope(str, Enum):
    """Lifetime of any per-Component scratchpad.

    NONE          : handler is a pure function of ctx.
    SESSION       : ctx.state[component_name] persists for the task only.
    CROSS_SESSION : persisted to .component-state/<tag>/<name>.json
                    across task invocations. Reserved; not enforced in v1.
    """
    NONE          = "none"
    SESSION       = "session"
    CROSS_SESSION = "cross_session"


@dataclass(frozen=True)
class Trust:
    """Component verification block.

    `evidence_anchor`, `blast_radius`, `rollback_when` are required for
    every Component. `out_of_evidence_probe` is additionally required for
    INDUCED_RULE — `policy.validate_trust` enforces this at load time.
    """
    evidence_anchor: str
    blast_radius: str               # local | workflow | global
    rollback_when: str
    out_of_evidence_probe: str = ""  # REQUIRED for INDUCED_RULE
    fallback: str = ""               # OPTIONAL
