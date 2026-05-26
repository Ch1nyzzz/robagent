"""Component runtime for the EnterpriseOps-Gym function-calling agent.

Owned by the agent version it lives under (e.g. `agent/enterpriseops/v0/runtime/`).
A fresh `cp -r v0 v_<domain>_<N>` carries this runtime forward unchanged, so
relative imports inside sibling component files (`from ..runtime import ...`)
keep resolving regardless of which v_N they live in.

EnterpriseOps-Gym specifics:
  * Agent shape: multi-turn function-calling loop over MCP servers, evaluated
    by SQL verifiers run against the final database state (not output text).
  * The component-set for each run is the .py files inside the sibling
    `components_<domain>/` directory — no workflow.yaml, no Patch enum, no
    frontier snapshot file. Add a file = active; delete a file = disabled;
    overwrite a file = replaced.
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
from .registry import load_components, load_components_from_dir
from .base import Dispatcher, build_dispatcher, clear_dispatcher_cache

__all__ = [
    "Component",
    "ComponentClass",
    "ComponentContext",
    "ComponentPolicyError",
    "Ctx",
    "Decision",
    "DecisionKind",
    "Dispatcher",
    "Handler",
    "Matcher",
    "Trust",
    "build_dispatcher",
    "clear_dispatcher_cache",
    "load_components",
    "load_components_from_dir",
    "validate_decision",
    "validate_registration",
    "validate_trust",
]
