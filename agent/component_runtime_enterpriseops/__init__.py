"""Component runtime for the ServiceNow EnterpriseOps-Gym function-calling agent.

Sibling of `agent/component_runtime_sopbench/` (Amazon SOP-Bench),
`agent/component_runtime/` (GAIA, single-shot prompt→answer) and
`agent_tau2/component_runtime/` (tau2, multi-turn dialogue with tool use).

EnterpriseOps-Gym specifics:
  * Agent shape: multi-turn function-calling loop over MCP servers, evaluated
    by SQL verifiers run against the final database state (not output text).
  * Mounts identical to SOP-Bench (PRE_LLM_TURN / PRE_TOOL_USE / etc.) — the
    upstream `orchestrators/react.py` loop maps cleanly. PRE_FINAL_EMIT is
    kept for symmetry but is less load-bearing here because the judge looks
    at DB rows rather than at agent text output.
"""
from .types import (
    Component,
    ComponentClass,
    ComponentContext,
    Ctx,
    Decision,
    DecisionKind,
    Handler,
    Matcher,
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
