"""Shared component-runtime core (Phase A of event-based runtime migration).

Holds the pieces that are GENUINELY identical across all 5 sibling
component runtimes (gaia / tau2 / toolathlon / sopbench / enterpriseops):

  - shared_types.py  Trust, ComponentClass
  - workflow.py      Workflow, Edge, Patch, PatchOp, FrontierSnapshot,
                     apply_patch (YAML schema + patch ops the proposer
                     emits; data-only, no sibling-specific behaviour)
  - registry.py      _load_one(path, pkg_prefix) — generic component
                     module loader; siblings wrap with their own
                     COMPONENTS_DIR_DEFAULT + Mount group-by
  - policy.py        ComponentPolicyError + generic validators that
                     take the per-sibling ALLOWED matrix as data

Stays per-sibling (do NOT lift):

  - DecisionKind enum     (gaia/sopbench/enterpriseops vs tau2/toolathlon
                           differ on REWRITE vs REWRITE_TOOL_ARGS + DEFER)
  - Decision dataclass    (sibling-specific factory methods)
  - ComponentContext      (mount-specific fields differ per sibling)
  - Component dataclass   (siblings hold their own per-bench ctx type)
"""

from .shared_types import ComponentClass, Trust
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
    "Trust",
    "Workflow",
    "apply_patch",
    "events",
    "load_module_from_path",
    "validate_decision",
    "validate_registration",
    "validate_trust",
]
