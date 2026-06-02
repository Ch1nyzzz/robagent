"""GAIA workflow / patch / frontier-snapshot — now a shim over the shared core.

Phase A of the event-runtime migration extracted the byte-identical
sibling workflow.py implementations into
`ballast.component_runtime_core.workflow`. This shim keeps the
existing import surface (`from agent.component_runtime.workflow import
Workflow, Patch, ...`) intact so callers across `agent/`,
`ballast/`, and the proposer prompts don't need to change.
"""
from ballast.component_runtime_core.workflow import (  # noqa: F401
    Edge,
    FrontierSnapshot,
    Patch,
    PatchOp,
    Workflow,
    apply_patch,
)
