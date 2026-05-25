"""Component runtime for the SOP-Bench function-calling agent.

Sibling of `agent/component_runtime/` (GAIA, single-shot prompt→answer) and
`agent_tau2/component_runtime/` (tau2, multi-turn dialogue with tool use).

SOP-Bench specifics:
  * Agent shape: multi-turn function-calling loop reading an SOP document +
    structured task input + tool catalog, emitting an `<xxx>VALUE</xxx>`
    final XML.
  * Mounts extend GAIA's prompt-build / response-postprocess set with
    per-turn (PRE_LLM_TURN) and per-tool-call (PRE/POST_TOOL_USE) points
    plus PRE_FINAL_EMIT for output normalization.
"""
from .types import (
    Capability,
    Component,
    ComponentClass,
    ComponentContext,
    Ctx,
    Decision,
    DecisionKind,
    Handler,
    Matcher,
    StateScope,
    Trust,
)
from .policy import ComponentPolicyError, validate_decision, validate_registration, validate_trust
from .registry import (
    COMPONENTS_DIR_DEFAULT,
    load_components,
    load_components_from_dir,
)
from .workflow import (
    Edge,
    FrontierSnapshot,
    Patch,
    PatchOp,
    Workflow,
    apply_patch,
)
from .base import (
    Dispatcher,
    build_dispatcher,
    clear_dispatcher_cache,
    resolve_workflow_path,
)

__all__ = [
    "Capability",
    "Component",
    "ComponentClass",
    "ComponentContext",
    "ComponentPolicyError",
    "Ctx",
    "Decision",
    "DecisionKind",
    "Dispatcher",
    "Edge",
    "FrontierSnapshot",
    "Handler",
    "Matcher",
    "Patch",
    "PatchOp",
    "StateScope",
    "Trust",
    "Workflow",
    "apply_patch",
    "build_dispatcher",
    "clear_dispatcher_cache",
    "load_components",
    "load_components_from_dir",
    "resolve_workflow_path",
    "validate_decision",
    "validate_registration",
    "validate_trust",
    "COMPONENTS_DIR_DEFAULT",
]
