"""Inject a strict output spec for warehouse_package_inspection.

The SOP's section 6 ("Output") lists deliverables ("Resolution Status
Update", etc.) but never specifies the XML format the grader expects.
The agent's default system prompt only says "output it EXACTLY in the
XML format that the SOP's Output section specifies" — which leaves the
model to improvise, and v0 traces show it does so wildly (sometimes
nests resolution_status under <resolution_details>, sometimes emits a
plain string, sometimes leaves the status at "pending" because the
SOP's only explicit transitions are Pending→Processing).

This component anchors on two off-evidence facts:

  1) metadata.json::output_columns == ["resolution_status"] — the
     benchmark's declared single-field output.
  2) WarehousePackageInspectionManager.updateResolutionStatus.valid_statuses
     == ["Pending", "Processing", "Resolved", "Returned to Vendor"] —
     the tool's hard-coded value space (and the only structurally
     enforced enum in this domain; the toolspec JSON itself omits the
     enum on the schema).

It injects a brief output spec at SESSION_START. No reasoning is
encoded — just the format and the enum.
"""
from __future__ import annotations

from agent.component_runtime_sopbench import (
    Capability,
    Component,
    ComponentClass,
    ComponentContext,
    Decision,
    Mount,
    StateScope,
    Trust,
)


_OUTPUT_SPEC = """

OUTPUT FORMAT (warehouse_package_inspection):
The benchmark's declared output column is `resolution_status`. Your
final answer MUST end with a single top-level XML tag:

    <resolution_status>VALUE</resolution_status>

VALUE must be exactly one of (case-sensitive, exactly as written):
    Processing
    Resolved
    Returned to Vendor

These are the canonical values the `updateResolutionStatus` tool
accepts and returns. Prefer the value that tool returns over any
free-form synthesis you might do. Do NOT emit "Pending" as the final
answer — per SOP 5.3.2, Pending is only the *initial* state, not a
resolution.

You may emit the rest of the Problem Classification Report in any
structure you like, but the FINAL `<resolution_status>VALUE</resolution_status>`
tag must be present at the top level (not nested inside other tags)
and must contain ONLY the value text (no nested XML)."""


def _matches(ctx: ComponentContext) -> bool:
    return ctx.benchmark == "warehouse_package_inspection"


def _handler(ctx: ComponentContext) -> Decision:
    return Decision.inject_context(_OUTPUT_SPEC)


COMPONENT = Component(
    name="sopbench_warehouse_package_inspection_resolution_status_output_spec",
    cls=ComponentClass.MECHANISM_LAYER,
    mount=Mount.SESSION_START,
    matcher=_matches,
    handler=_handler,
    state_scope=StateScope.NONE,
    capabilities=(Capability.NONE,),
    priority=100,
    trust=Trust(
        evidence_anchor=(
            "metadata.json::output_columns == ['resolution_status'] AND "
            "third_party/SOP-Bench/.../warehouse_package_inspection/tools.py::"
            "updateResolutionStatus.valid_statuses == "
            "['Pending','Processing','Resolved','Returned to Vendor']"
        ),
        blast_radius="local",
        rollback_when=(
            "metadata.json::output_columns changes, or updateResolutionStatus "
            "drops/renames any of the four enum values, or the SOP starts "
            "specifying its own XML format for the output section."
        ),
        fallback=(
            "agent default behavior — model improvises XML structure from "
            "SOP section 6's prose."
        ),
    ),
)
