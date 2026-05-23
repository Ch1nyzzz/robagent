"""Deterministic audit-completeness guard.

Used by ``mh_tau2_iter3_baseline_exhaustive_audit``. It tracks, purely from the
conversation text:

  * the *universe* — every customer record (transaction) surfaced by a tool
    result, and
  * the *examined* set — every record the agent has actually referenced in its
    own reasoning.

When the agent is about to take a cash-back resolution action (submit a
dispute, correct rewards) without having referenced most of that universe, the
audit is judged incomplete so the turn can be regenerated with a corrective
note. All logic here is deterministic string processing — no LLM calls and no
task-specific identifiers (the transaction ids it handles are the customer's
own data, read back out of tool results).
"""
from __future__ import annotations

import json
import re

# Transaction identifiers in the banking domain look like ``txn_2037a5f15196``.
_TXN_RE = re.compile(r"\btxn_[0-9a-f]{8,}\b")

# Substrings marking a tool (or discoverable tool) as a cash-back resolution
# write. Kept generic: any dispute-submission or rewards-correction tool.
_RESOLUTION_MARKERS = ("cash_back_dispute", "cashback_dispute", "transaction_rewards")

# Discoverable-tool wrapper calls whose *target* tool we must inspect.
_WRAPPER_TOOLS = {
    "give_discoverable_user_tool",
    "unlock_discoverable_agent_tool",
    "call_discoverable_user_tool",
    "call_discoverable_agent_tool",
}

# Below this many transactions a task is not a bulk audit; never nudge.
_MIN_UNIVERSE = 12
# Nudge when the agent has referenced fewer than this fraction of the universe.
_COVERAGE_FLOOR = 0.6
# Cap on how many missing ids to spell out in the corrective note.
_MAX_LISTED = 24


def _txn_ids(text) -> set:
    """Every transaction id appearing in ``text``."""
    if not isinstance(text, str) or not text:
        return set()
    return set(_TXN_RE.findall(text))


def _target_tool_name(arguments) -> str:
    """The discoverable tool a wrapper call points at, or ``""``."""
    data = arguments
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            return ""
    if not isinstance(data, dict):
        return ""
    for key in ("discoverable_tool_name", "agent_tool_name", "tool_name"):
        value = data.get(key)
        if isinstance(value, str):
            return value
    return ""


def _is_resolution_call(tool_call) -> bool:
    """True if this tool call submits a dispute or corrects rewards."""
    name = (getattr(tool_call, "name", "") or "").lower()
    if any(marker in name for marker in _RESOLUTION_MARKERS):
        return True
    if name in _WRAPPER_TOOLS:
        target = _target_tool_name(getattr(tool_call, "arguments", None)).lower()
        if any(marker in target for marker in _RESOLUTION_MARKERS):
            return True
    return False


class AuditCompletenessGuard:
    """Tracks audit coverage and decides when a resolution turn is premature."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._universe: set = set()   # txn ids seen in tool results
        self._examined: set = set()   # txn ids referenced in agent text
        self._nudged: bool = False    # at most one nudge per conversation

    # -- observation -------------------------------------------------------
    def observe_tool_result(self, text) -> None:
        """Record the records a tool result surfaced."""
        self._universe |= _txn_ids(text)

    def observe_agent_text(self, text) -> None:
        """Record the records the agent referenced in its own message."""
        self._examined |= _txn_ids(text)

    # -- decision ----------------------------------------------------------
    @staticmethod
    def has_resolution_call(message) -> bool:
        for tool_call in getattr(message, "tool_calls", None) or []:
            if _is_resolution_call(tool_call):
                return True
        return False

    def audit_incomplete(self, pending_message) -> bool:
        """True when a resolution turn is not backed by a complete audit."""
        if self._nudged:
            return False
        if not self.has_resolution_call(pending_message):
            return False
        if len(self._universe) < _MIN_UNIVERSE:
            return False
        # Count this turn's own text too: the agent may audit, then act.
        examined = self._examined | _txn_ids(getattr(pending_message, "content", ""))
        covered = examined & self._universe
        return len(covered) < _COVERAGE_FLOOR * len(self._universe)

    def guidance(self, pending_message) -> str:
        """Corrective note for an incomplete audit. Marks the nudge as spent."""
        self._nudged = True
        examined = self._examined | _txn_ids(getattr(pending_message, "content", ""))
        missing = sorted(self._universe - examined)
        total = len(self._universe)
        accounted = total - len(missing)
        note = (
            f"You retrieved {total} transactions but your analysis so far has "
            f"individually accounted for only {accounted} of them. Do not "
            "submit any dispute or correction yet. First send a plain message "
            "containing a complete, numbered audit table with one row per "
            f"transaction — all {total} of them — showing the amount, the "
            "category, the reward you compute (floored), the recorded reward, "
            "and whether it is an error. Only once every transaction appears "
            "in that table should you act, taking exactly one resolution "
            "action per error you found — no more and no fewer."
        )
        if missing and len(missing) <= _MAX_LISTED:
            note += " Transactions still missing from your analysis: " + ", ".join(missing) + "."
        return note
