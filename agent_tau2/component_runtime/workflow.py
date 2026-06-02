"""tau2 workflow — shim over the shared core (Phase A migration).

Kept the existing import surface
(`from agent_tau2.component_runtime.workflow import Workflow, ...`)
intact while the actual implementation now lives in
`ballast.component_runtime_core.workflow`. Previously this file was
byte-identical to gaia / toolathlon / sopbench / enterpriseops.
"""
from ballast.component_runtime_core.workflow import (  # noqa: F401
    Edge,
    FrontierSnapshot,
    Patch,
    PatchOp,
    Workflow,
    apply_patch,
)
