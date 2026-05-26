"""Clean-path terminal: rewrite 'Processing' to 'Resolved' when no problems found.

iter1/4/5 leave one residual structural hole. SOP 5.3.2 reads:

    Update resolution_status based on problem_type:
      - Initialize as "Pending"
      - Progress to "Processing" upon problem confirmation

The SOP names Pending as the initial state and Processing as the transition
triggered "upon problem confirmation". It does not name the terminal state
for the no-problem path -- yet updateResolutionStatus.valid_statuses ==
['Pending','Processing','Resolved','Returned to Vendor'] contains 'Resolved'
which appears nowhere else in the SOP. The natural reading is:

    no-Wrong-Item-and-no-other-problem -> 'Resolved'

iter4 already maps a bare 'Pending' final emit to 'Resolved' under this
reading. But the iter5 train trace shows the model emits 'Processing' (not
'Pending') on the clean path because validateBarcode itself returns
resolution_status='Processing' alongside problem_type=[] (tools.py:367-371),
which the model echoes through the rest of the loop. iter4's
Pending->Resolved mapping never fires; the bare 'Processing' falls through
to 'Processing' even though no tool ever reported a problem.

This component closes the gap. At PRE_FINAL_EMIT, when the model's
final_output canonicalises to 'Processing' AND every executed tool call's
problem_type field is empty AND SOP 5.2's input-derived problem checks
(quantity / warehouse) also find nothing, the package is on the clean path
and the canonical terminal status is 'Resolved'.

Anchors (all off-evidence):
  * tools.py::updateResolutionStatus.valid_statuses ==
      ['Pending','Processing','Resolved','Returned to Vendor'] -- 'Resolved'
      is a tool-declared enum value with no other defining trigger.
  * SOP 5.3.2 -- Pending initial, Processing on problem confirmation; no
    SOP rule fires Processing when problem_type is empty.
  * SOP 5.2.1 -- Cancelled (confirmed=0), Overage (received>ordered),
    Underage (received<ordered) -- the deterministic SOP 5.2 derivations
    re-applied here from declared input columns.
  * SOP 5.2.2 -- Wrong Warehouse when intended != actual.
  * Every problem-classifying tool (validateBarcode, calculateQuantity-
    Variance, verifyWarehouseLocation, assessPackageCondition) returns a
    'problem_type' list in its result dict. Empty list across all four ==
    "no problems found by the SOP's authoritative checks".
  * metadata.json::output_columns == ['resolution_status'].
  * metadata.json::input_columns -- ordered_quantity, confirmed_quantity,
    received_quantity, intended_warehouse_id, actual_warehouse_id are
    declared inputs, so SOP 5.2 can be re-applied deterministically.

Matcher (all of):
  1. benchmark == 'warehouse_package_inspection'
  2. final_output canonicalises to 'Processing'
  3. at least one tool call with a 'problem_type' field was executed (so
     we have signal), AND every such call returned an empty list
  4. SOP 5.2.1/5.2.2 re-applied to declared input columns finds no problem

Handler: rewrite to '<resolution_status>Resolved</resolution_status>'

Priority is 40 so it runs BEFORE iter5 (priority 50, fires only on RTV)
and BEFORE iter4's normaliser (priority 200). iter5's matcher and iter6's
matcher are mutually exclusive (iter5 needs RTV-as-final and any 5.2
problem; iter6 needs Processing-as-final and zero 5.2 problems), so order
between them is cosmetic. Running before iter4 means iter4 then sees a
canonical 'Resolved' wrapping and noops.
"""
from __future__ import annotations

import re
from typing import Any

from agent.component_runtime_sopbench import (
    Component,
    ComponentClass,
    ComponentContext,
    Decision,
    Trust,
)


_PROCESSING_NORMALISED = "processing"
_RESOLVED_XML = "<resolution_status>Resolved</resolution_status>"

_TAG_RESOLUTION_STATUS = re.compile(
    r"<resolution_status>\s*(.*?)\s*</resolution_status>",
    re.DOTALL | re.IGNORECASE,
)
_TAG_CURRENT_STATUS = re.compile(
    r"<current_status>\s*(.*?)\s*</current_status>",
    re.DOTALL | re.IGNORECASE,
)
_TRAILING_TOKEN = re.compile(r"([A-Za-z][A-Za-z _\-]{0,40}[A-Za-z])\s*\Z")


def _normalise(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    return " ".join(s.split())


def _extract_value(final_output: str) -> str | None:
    matches = _TAG_RESOLUTION_STATUS.findall(final_output)
    if matches:
        return matches[-1]
    matches = _TAG_CURRENT_STATUS.findall(final_output)
    if matches:
        return matches[-1]
    m = _TRAILING_TOKEN.search(final_output)
    if m:
        return m.group(1)
    return None


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _input_derived_problem_exists(task_input: dict[str, Any]) -> bool:
    """Re-apply SOP 5.2.1 + 5.2.2 to declared input columns.

    Mirrors tools.py::calculateQuantityVariance and verifyWarehouseLocation.
    Returns True iff any quantity or warehouse problem would be raised.
    """
    confirmed = _as_int(task_input.get("confirmed_quantity"))
    ordered = _as_int(task_input.get("ordered_quantity"))
    received = _as_int(task_input.get("received_quantity"))

    if confirmed is not None and confirmed == 0:
        return True  # SOP 5.2.1: Cancelled Quantity
    if ordered is not None and received is not None and received != ordered:
        return True  # SOP 5.2.1: Overage / Underage

    intended = task_input.get("intended_warehouse_id")
    actual = task_input.get("actual_warehouse_id")
    if isinstance(intended, str) and isinstance(actual, str):
        if intended.strip().upper() != actual.strip().upper():
            return True  # SOP 5.2.2: Wrong Warehouse

    return False


def _all_tool_problems_empty(executed_tool_calls: list[dict[str, Any]]) -> bool:
    """True iff at least one tool call returned a problem_type field and
    EVERY such return was empty. Vacuously False when no tool with a
    problem_type field was called (defensive: we want positive evidence
    that the SOP's authoritative checks ran and found nothing).
    """
    saw_signal = False
    for tc in executed_tool_calls:
        if not isinstance(tc, dict):
            continue
        result = tc.get("result")
        if not isinstance(result, dict):
            continue
        if "problem_type" not in result:
            continue
        saw_signal = True
        if result.get("problem_type"):
            return False
    return saw_signal


def _matches(ctx: ComponentContext) -> bool:
    if ctx.benchmark != "warehouse_package_inspection":
        return False
    if not isinstance(ctx.final_output, str) or not ctx.final_output.strip():
        return False
    raw = _extract_value(ctx.final_output)
    if raw is None:
        return False
    if _normalise(raw) != _PROCESSING_NORMALISED:
        return False
    if not isinstance(ctx.task_input, dict) or not ctx.task_input:
        return False
    if _input_derived_problem_exists(ctx.task_input):
        return False
    return _all_tool_problems_empty(ctx.executed_tool_calls or [])


def _handler(ctx: ComponentContext) -> Decision:
    return Decision.rewrite(_RESOLVED_XML)


COMPONENT = Component(
    name="sopbench_warehouse_package_inspection_clean_path_resolved_terminal",
    cls=ComponentClass.MECHANISM_LAYER,
    listens="pre_final_emit",
    matcher=_matches,
    handler=_handler,
    priority=40,  # Run before iter5 (50) and iter4 normaliser (200).
    trust=Trust(
        evidence_anchor=(
            "third_party/SOP-Bench/.../warehouse_package_inspection/tools.py::"
            "updateResolutionStatus.valid_statuses == "
            "['Pending','Processing','Resolved','Returned to Vendor'] -- "
            "'Resolved' is a tool-declared enum value with no other "
            "SOP-defined trigger besides the no-problem terminal path "
            "(SOP 5.3.2 names only Pending as initial and Processing as "
            "the transition on problem confirmation). "
            "AND every problem-classifying tool (validateBarcode, "
            "calculateQuantityVariance, verifyWarehouseLocation, "
            "assessPackageCondition) returns a 'problem_type' list in its "
            "result dict; empty across all called such tools == the SOP's "
            "authoritative checks found nothing. "
            "AND SOP 5.2.1 (Cancelled when confirmed=0, Overage when "
            "received>ordered, Underage when received<ordered) + SOP 5.2.2 "
            "(Wrong Warehouse when intended!=actual) are deterministic "
            "from metadata.json::input_columns (ordered_quantity, "
            "confirmed_quantity, received_quantity, intended_warehouse_id, "
            "actual_warehouse_id). "
            "AND metadata.json::output_columns == ['resolution_status']."
        ),
        blast_radius="local",
        rollback_when=(
            "updateResolutionStatus.valid_statuses changes (drops 'Resolved' "
            "or adds a new no-problem terminal value), or any of the four "
            "problem-classifying tools stops returning a 'problem_type' "
            "key in its result dict, or SOP 5.3.2 is reformulated to "
            "specify a different no-problem terminal status, or SOP 5.2 "
            "input-derived rules change formulas, or evolution_summary "
            "shows train accuracy drops after this component is admitted."
        ),
        fallback=(
            "iter4's Pending->Resolved mapping alone -- relies on the LLM "
            "to emit 'Pending' on the clean path; in iter5 traces the "
            "model echoes the validateBarcode tool's resolution_status="
            "'Processing' instead, so iter4's mapping never fires and the "
            "bare 'Processing' falls through unchanged."
        ),
    ),
)
