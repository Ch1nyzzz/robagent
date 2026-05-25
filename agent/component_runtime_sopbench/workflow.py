"""SOP-Bench workflow — shim over the shared core (Phase A migration).

Kept the existing import surface (`from agent.component_runtime_sopbench.workflow
import Workflow, Patch, ...`) intact while the actual implementation now
lives in `meta_harness.component_runtime_core.workflow`. Previously this
file was byte-identical to gaia / tau2 / toolathlon / enterpriseops.
"""
from meta_harness.component_runtime_core.workflow import (  # noqa: F401
    Edge,
    FrontierSnapshot,
    Patch,
    PatchOp,
    Workflow,
    apply_patch,
)
