"""Shared component-runtime core (Phase A of event-based runtime migration).

Holds the pieces that are GENUINELY identical across all 5 sibling
component runtimes (gaia / tau2 / toolathlon / sopbench / enterpriseops):

  - shared_types.py  Trust, StateScope, ComponentClass
  - workflow.py      Workflow, Edge, Patch, PatchOp, FrontierSnapshot,
                     apply_patch (YAML schema + patch ops the proposer
                     emits; data-only, no sibling-specific behaviour)
  - registry.py      _load_one(path, pkg_prefix) — generic component
                     module loader; siblings wrap with their own
                     COMPONENTS_DIR_DEFAULT + Mount group-by
  - policy.py        ComponentPolicyError + generic validators that
                     take the per-sibling ALLOWED matrix as data

Stays per-sibling (do NOT lift):

  - Mount enum            (different lifecycle vocabularies)
  - DecisionKind enum     (gaia/sopbench/enterpriseops vs tau2/toolathlon
                           differ on REWRITE vs REWRITE_TOOL_ARGS + DEFER)
  - Capability enum       (gaia lacks TOOL_CALL)
  - Decision dataclass    (sibling-specific factory methods)
  - ComponentContext      (mount-specific fields differ per sibling)
  - Component dataclass   (siblings hold their own Mount type;
                           Phase B will add a shared `listens: str` field
                           via per-sibling dataclass definitions)

Phase B will add `dispatcher.py` and `event_context.py` here for the
event-based dispatch surface.
"""

from .shared_types import ComponentClass, StateScope, Trust
from .workflow import (
    Edge,
    FrontierSnapshot,
    Patch,
    PatchOp,
    Workflow,
    apply_patch,
)
from .policy import (
    ComponentPolicyError,
    validate_decision,
    validate_registration,
    validate_trust,
)
from .registry import load_module_from_path
from .dispatcher import Dispatcher
from .event_context import EventContext
from . import events  # Tier-1 event-name string constants

__all__ = [
    "ComponentClass",
    "ComponentPolicyError",
    "Dispatcher",
    "Edge",
    "EventContext",
    "FrontierSnapshot",
    "Patch",
    "PatchOp",
    "StateScope",
    "Trust",
    "Workflow",
    "apply_patch",
    "events",
    "load_module_from_path",
    "validate_decision",
    "validate_registration",
    "validate_trust",
]
