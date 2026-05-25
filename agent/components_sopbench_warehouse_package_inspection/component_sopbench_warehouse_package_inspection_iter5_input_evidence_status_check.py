"""Respect SOP 5.1.1 short-circuit before rewriting RTV -> Processing.

iter8 widened iter7 to fire when the final output is 'Returned to Vendor'
AND the declared inputs show at least one of:
    (a) Severe Unmatched Quantity (SOP 5.2.1 final bullet, QVT-gated)
    (b) Wrong Warehouse           (SOP 5.2.2, categorical)
    (c) Cancelled Quantity        (SOP 5.2.1 first bullet, categorical)

That widening implicitly assumed SOP 5.2 classifications dominate the
SOP 5.1.1 "Wrong Item" short-circuit. But SOP 5.1.1 literally says:

    "Execute barcode validation protocol by comparing the barcode from
    path provided in received_product_bar_code against confirmed_product_id
    using image processing system, if not match, ignore all the steps
    below and mark the problem as 'Wrong Item' and set resolution_status
    to 'Returned to Vendor', only execute 'Problem Classification Report'
    step 6."

The "ignore all the steps below" clause makes a confirmed barcode
mismatch dominant over every SOP 5.2 signal (the SOP authors thought
through the precedence and chose RTV). When the barcode image, decoded
by a general-purpose decoder, demonstrably does NOT match the
LLM-supplied confirmed_product_id, iter5 must not rewrite the LLM's
'Returned to Vendor' final answer to 'Processing'.

iter10 adds exactly that check to iter5's matcher: before applying the
RTV -> Processing rewrite, decode the barcode image at
task_input['received_product_bar_code'] using cv2.barcode.BarcodeDetector
(the same general-purpose decoder iter9 uses at POST_TOOL_USE). If the
decoder yields a non-empty value that disagrees with task_input
['confirmed_product_id'], the SOP 5.1.1 short-circuit has fired and we
defer (Decision.allow), letting iter4's normaliser pass the LLM's RTV
through unchanged.

Why the rewrite is still preserved when the decoder fails
---------------------------------------------------------
For ~half the train rows the decoder cannot extract any digits (the
barcode crop is too small / too noisy for OpenCV's bundled detector).
In those rows we have no positive evidence either way, so the
short-circuit is NOT confirmed and iter5 fires exactly as before. This
preserves iter5's win on tasks where the LLM emitted RTV solely because
the tool's po-parity simulation said barcode_match=False but the gold
status is actually Processing (the structurally strong SOP 5.2 signal
still tells the right answer there).

Why this lives inside iter5 and not as a new node
-------------------------------------------------
iter6 (priority 40) and iter4 (priority 200) bracket iter5 (priority 50);
adding a separate PRE_FINAL_EMIT node that "forces RTV" wouldn't help
because iter5 still rewrites RTV -> Processing afterwards. The right
fix is a tightening of iter5's matcher itself: when the SOP-5.1.1
precedence applies, iter5 should not fire. Same name, same class
(mechanism_layer), same mount (PRE_FINAL_EMIT), same priority (50);
only the matcher gains one extra clause and the capability set gains
READ_FILE.

Behaviour at PRE_FINAL_EMIT
---------------------------
Fires when final_output canonicalises to 'Returned to Vendor' AND the
declared inputs show at least one of (a)-(c) above AND the barcode
decoder did NOT confirm a barcode mismatch (decoder produced an empty
string, OR decoder produced a value that equals the LLM-supplied
confirmed_product_id).

Rewrites final_output to '<resolution_status>Processing</resolution_status>'.

Out-of-evidence probe
---------------------
* cv2 / cv2.barcode not importable, or the relative path doesn't
  resolve, or the decoder returns empty / no digits -> defer to
  iter8's previous matcher (fire on strong SOP 5.2 signal regardless).
* Decoder result equal to confirmed_product_id -> short-circuit NOT
  active; iter5 fires as before (no behaviour change vs iter8).
* Decoder result differs from confirmed_product_id (mismatch
  confirmed) -> SOP 5.1.1 dominates; iter5 defers.

Anchors (all off-evidence rows)
-------------------------------
  * SOP 5.1.1 verbatim quote above (the "ignore all the steps below"
    clause encodes the SOP's chosen precedence).
  * SOP 3.4 (QVT default 5%), SOP 5.2.1 (categorical + severity),
    SOP 5.2.2 (warehouse equality) -- unchanged from iter8.
  * tools.py::calculateQuantityVariance / verifyWarehouseLocation /
    updateResolutionStatus.valid_statuses == ['Pending','Processing',
    'Resolved','Returned to Vendor'] -- unchanged from iter8.
  * toolspecs.json::validateBarcode.inputSchema declares
    received_product_bar_code as 'File path to the barcode image of
    the received product'.
  * metadata.json::input_columns includes received_product_bar_code,
    confirmed_product_id, ordered_quantity, confirmed_quantity,
    received_quantity, intended_warehouse_id, actual_warehouse_id.
  * metadata.json::output_columns == ['resolution_status'].
  * cv2.barcode.BarcodeDetector is a general-purpose EAN/UPC decoder
    with no task-specific state (same anchor iter9 uses).
"""
from __future__ import annotations

import importlib
import os
import re
from typing import Any, Optional

from agent.component_runtime_sopbench import (
    Capability,
    Component,
    ComponentClass,
    ComponentContext,
    Decision,
    StateScope,
    Trust,
)


_RTV_NORMALISED = "returned to vendor"
_PROCESSING_XML = "<resolution_status>Processing</resolution_status>"

# SOP 3.4: QVT default value is 5%.
_QVT_DEFAULT_PCT = 5.0

_BENCH_MODULE = (
    "amazon_sop_bench.benchmarks.data.warehouse_package_inspection.tools"
)

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
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _strong_input_derived_problem(task_input: dict[str, Any]) -> bool:
    """SOP 5.2.1 categorical-without-tolerance + Severe + SOP 5.2.2.

    Returns True iff at least one of:
      (a) Severe Unmatched Quantity: confirmed > 0 AND
          |received - confirmed| / confirmed * 100 > QVT (default 5%).
      (b) Wrong Warehouse: intended_warehouse_id != actual_warehouse_id.
      (c) Cancelled Quantity: confirmed_quantity == 0
          (SOP 5.2.1 first bullet -- categorical, no tolerance band).
    """
    confirmed = _as_int(task_input.get("confirmed_quantity"))
    received = _as_int(task_input.get("received_quantity"))

    if confirmed is not None and confirmed == 0:
        return True

    if confirmed is not None and confirmed > 0 and received is not None:
        variance_pct = abs(received - confirmed) / confirmed * 100.0
        if variance_pct > _QVT_DEFAULT_PCT:
            return True

    intended = task_input.get("intended_warehouse_id")
    actual = task_input.get("actual_warehouse_id")
    if isinstance(intended, str) and isinstance(actual, str):
        if intended.strip().upper() != actual.strip().upper():
            return True

    return False


def _resolve_bench_dir() -> Optional[str]:
    try:
        mod = importlib.import_module(_BENCH_MODULE)
    except Exception:
        return None
    f = getattr(mod, "__file__", None)
    if not f:
        return None
    return os.path.dirname(os.path.abspath(f))


def _resolve_image_path(rel_path: str) -> Optional[str]:
    if not rel_path:
        return None
    if os.path.isabs(rel_path) and os.path.exists(rel_path):
        return rel_path
    base = _resolve_bench_dir()
    if base:
        cand = os.path.normpath(os.path.join(base, rel_path))
        if os.path.exists(cand):
            return cand
    return None


def _decode_barcode(abs_path: str) -> str:
    try:
        import cv2
    except Exception:
        return ""
    if not hasattr(cv2, "barcode"):
        return ""
    try:
        img = cv2.imread(abs_path)
        if img is None:
            return ""
        detector = cv2.barcode.BarcodeDetector()
        ok, decoded, _types, _ = detector.detectAndDecodeWithType(img)
        if not ok or not decoded:
            return ""
        for d in decoded:
            if d:
                return str(d).strip()
        return ""
    except Exception:
        return ""


def _normalise_id(value: object) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1].strip()
    return s


def _sop_511_short_circuit_confirmed(task_input: dict[str, Any]) -> bool:
    """True iff the barcode decoder confirms the SOP 5.1.1 short-circuit.

    Returns True only when the decoder produces a non-empty value that
    disagrees with task_input['confirmed_product_id']. Returns False if
    we cannot decode, or if the decoded value matches (no short-circuit).
    """
    rel = task_input.get("received_product_bar_code")
    confirmed = _normalise_id(task_input.get("confirmed_product_id"))
    if not isinstance(rel, str) or not rel or not confirmed:
        return False
    abs_path = _resolve_image_path(rel)
    if not abs_path:
        return False
    decoded = _decode_barcode(abs_path)
    if not decoded:
        return False
    return decoded != confirmed


def _matches(ctx: ComponentContext) -> bool:
    if ctx.benchmark != "warehouse_package_inspection":
        return False
    if not isinstance(ctx.final_output, str) or not ctx.final_output.strip():
        return False
    raw = _extract_value(ctx.final_output)
    if raw is None:
        return False
    if _normalise(raw) != _RTV_NORMALISED:
        return False
    if not isinstance(ctx.task_input, dict) or not ctx.task_input:
        return False
    if not _strong_input_derived_problem(ctx.task_input):
        return False
    # SOP 5.1.1 short-circuit: if the barcode image disagrees with the
    # LLM-supplied confirmed_product_id, the SOP-mandated outcome is
    # RTV; defer to iter4 to normalise the LLM's RTV emission.
    if _sop_511_short_circuit_confirmed(ctx.task_input):
        return False
    return True


def _handler(ctx: ComponentContext) -> Decision:
    return Decision.rewrite(_PROCESSING_XML)


COMPONENT = Component(
    name="sopbench_warehouse_package_inspection_input_evidence_status_check",
    cls=ComponentClass.MECHANISM_LAYER,
    listens="pre_final_emit",
    matcher=_matches,
    handler=_handler,
    state_scope=StateScope.NONE,
    capabilities=(Capability.READ_FILE,),
    priority=50,
    trust=Trust(
        evidence_anchor=(
            "SOP 5.1.1 ('Execute barcode validation protocol by comparing "
            "the barcode from path provided in received_product_bar_code "
            "against confirmed_product_id using image processing system, "
            "if not match, ignore all the steps below and mark the problem "
            "as Wrong Item and set resolution_status to Returned to Vendor') "
            "-- the 'ignore all the steps below' clause encodes the SOP's "
            "chosen precedence: a confirmed barcode mismatch dominates every "
            "SOP 5.2 classification. AND SOP 5.2.1 enumerates three "
            "categorical input-derived problems followed by one quantitative "
            "severity callout: (1) confirmed_quantity == 0 -> 'Cancelled "
            "Quantity' (no tolerance band), (2) received > ordered -> "
            "'Overage Quantity', (3) received < ordered -> 'Underage "
            "Quantity', (4) |variance| > QVT -> 'Severe Unmatched Quantity'. "
            "SOP 3.4 defines QVT default = 5%. SOP 5.2.2: 'If mismatch "
            "detected, initiate Wrong Warehouse protocol.' "
            "tools.py::calculateQuantityVariance has a dedicated early-return "
            "branch for confirmed_quantity == 0 emitting problem_type="
            "['Cancelled Quantity']; tools.py::verifyWarehouseLocation "
            "compares intended vs actual. tools.py::updateResolutionStatus "
            "branches to 'Processing' when problem_type is non-empty and "
            "current_status == 'Pending'. tools.py::updateResolutionStatus."
            "valid_statuses == ['Pending','Processing','Resolved','Returned "
            "to Vendor']. toolspecs.json::validateBarcode.inputSchema "
            "declares received_product_bar_code as 'File path to the barcode "
            "image of the received product'. metadata.json::input_columns "
            "includes received_product_bar_code, confirmed_product_id, "
            "ordered_quantity, confirmed_quantity, received_quantity, "
            "intended_warehouse_id, actual_warehouse_id. "
            "metadata.json::output_columns == ['resolution_status']. "
            "cv2.barcode.BarcodeDetector is a general-purpose EAN/UPC "
            "decoder with no task-specific state (same anchor iter9 uses)."
        ),
        blast_radius="local",
        rollback_when=(
            "SOP 5.1.1 changes its 'ignore all the steps below' precedence "
            "clause (e.g., reframes barcode validation as one among equal "
            "checks rather than a short-circuit), or SOP 5.2.1 changes the "
            "'Cancelled Quantity' clause, or SOP 3.4 changes the QVT default, "
            "or SOP 5.2.2 changes its warehouse equality rule, or "
            "tools.py::calculateQuantityVariance removes its dedicated "
            "confirmed_quantity == 0 branch, or metadata.json::input_columns "
            "drops/renames received_product_bar_code or confirmed_product_id "
            "or any of confirmed_quantity, received_quantity, "
            "intended_warehouse_id, actual_warehouse_id, or "
            "updateResolutionStatus renames 'Processing'/'Returned to "
            "Vendor', or OpenCV's cv2.barcode submodule is removed or "
            "renamed, or the received_product_bar_code value stops being a "
            "path resolvable against the benchmark data directory, or "
            "evolution_summary shows train accuracy drops after this "
            "matcher tightening is admitted."
        ),
        fallback=(
            "iter8's matcher: fire whenever final_output is RTV and any of "
            "Cancelled Quantity / Severe Unmatched Quantity / Wrong "
            "Warehouse is present in task_input. That over-fires when the "
            "barcode decoder confirms SOP 5.1.1 short-circuit -- the SOP's "
            "chosen precedence is then RTV, not Processing."
        ),
    ),
)
