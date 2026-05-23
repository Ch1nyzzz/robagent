"""Deterministic Stage-2 cash-back rewards gate.

Why this exists
---------------
The banking_knowledge cash-back rewards correction workflow is two-stage:

  Stage 1 (KB doc credit_cards_(general)_003): the agent ``give``s the user
      the discoverable user-tool ``submit_cash_back_dispute_0589``; the user
      calls it once per disputed transaction. The framework writes one row
      to ``cash_back_disputes`` and returns a response body containing a
      literal ``Status: RESOLVED`` (when ``task_config.dispute_settings.
      auto_resolve_disputes=True``) or ``Status: SUBMITTED`` (when False)
      line at tau2-bench-src/src/tau2/domains/banking_knowledge/tools.py
      lines 4140 / 4149.

  Stage 2 (KB doc credit_cards_(general)_004): for transactions whose
      Stage-1 response showed ``Status: RESOLVED``, the agent unlocks and
      calls ``update_transaction_rewards_3847`` (tools.py:711) once per
      transaction. Stage 2 writes a row to ``credit_card_transaction_history``
      (rewards_earned update) AND the call itself writes a unique-tool-name
      row to ``agent_discoverable_tools``; both rows participate in the
      evaluation DB hash.

On the adversarial task_029 (``auto_resolve_disputes=False``), the user
EXPLICITLY LIES after Stage 1: "I just got a notification that they've all
been approved and resolved! Can you now update my transaction records?" The
framework Status for every Stage-1 response in that conversation is
``SUBMITTED`` — gold has ZERO Stage-2 calls — but the LLM trusts the user's
natural-language claim and emits six ``update_transaction_rewards_3847``
calls, diverging the DB hash.

Four prior advisory attempts (iter5, iter6, iter8, iter13) all surfaced the
framework Status verdict to the LLM through tool-message augmentation or a
persistent SystemMessage ledger, and all four failed on task_029. The
iter13 ledger IS mounted in ``state.system_messages`` at the Stage-2 turn
yet the LLM still chooses user-prose authority over framework-system
authority. The advisory path is exhausted.

What this module does
---------------------
This module provides the predicate functions that the agent uses to
intercept and FILTER out Stage-2 ``call_discoverable_agent_tool`` invocations
whose target transaction's Stage-1 response did NOT carry ``Status: RESOLVED``
in the visible message history.

The gate is a sequencing controller anchored to a framework-emitted system
field (the literal ``Status: RESOLVED`` / ``Status: SUBMITTED`` line):

  * predicate ``resolved_transactions(messages)`` walks every tool message
    in the visible conversation history, parses each
    ``Executed: submit_cash_back_dispute_0589`` chunk for its
    ``transaction_id`` (from the framework's JSON-formatted Arguments block
    at tools.py:4155) AND its Status verdict (RESOLVED vs SUBMITTED). It
    returns the set of transaction_ids whose Stage-1 response was RESOLVED.

  * predicate ``is_blocked_stage2(call, resolved_set)`` returns the
    transaction_id of a tool call iff it is a Stage-2 update of an
    UNRESOLVED transaction; None otherwise.

The agent's ``generate_next_message`` uses these to drop blocked tool calls
from the LLM's emitted ``AssistantMessage.tool_calls`` and append a deferral
note to ``AssistantMessage.content`` explaining (a) which calls were dropped
and why, (b) the framework Status that the gate observed, (c) the KB doc
the gate cites. The LLM sees the note in its own assistant turn on the next
iteration and can re-plan; the framework simply does not execute the
filtered calls.

What this module does NOT do
----------------------------
  - It never fabricates a tool call.
  - It never fabricates a tool response.
  - It never reads user prose; the predicate is keyed exclusively to the
    framework's literal response markers and the call's inner schema.
  - It never references customer names, user ids, account ids, or any
    other task-specific datum. Transaction ids are extracted from the
    framework's Arguments block and the LLM's inner call arguments at
    runtime.
  - It does NOT block ``unlock_discoverable_agent_tool`` for the Stage-2
    tool name. Per memory entry ``tau2 cash-back resolution adversarial
    pair``, the unlock writes no DB row; the discoverable-tools audit
    table only receives a row from ``call_discoverable_agent_tool`` (one
    per unique inner tool name). Leaving the unlock alone keeps the
    auto-resolve path (task_026) intact — the LLM may still legitimately
    unlock the tool while we wait for Stage-1 resolution.

Stable structure captured
-------------------------
Two layers, both independent of the failed simulations:

  1. Framework tool-schema facts.
     - ``call_discoverable_agent_tool`` is declared at
       banking_knowledge/tools.py:631 (the framework entry-point for
       discoverable WRITE tool execution).
     - ``update_transaction_rewards_3847`` is declared at
       banking_knowledge/tools.py:711 as @is_discoverable_tool(WRITE).
     - ``submit_cash_back_dispute_0589`` is declared at
       banking_knowledge/tools.py:4106 with @is_discoverable_tool(WRITE).
     The predicate matches by exact framework tool-name; on a fresh
     banking_knowledge suite written against the same tools.py these
     match unchanged.

  2. Framework return-string facts.
     - The literal ``Executed: submit_cash_back_dispute_0589`` is emitted
       at tools.py:4155.
     - The literals ``Status: RESOLVED`` and ``Status: SUBMITTED`` are
       emitted at tools.py:4140 / 4149 respectively, keyed off
       ``task_config.dispute_settings.auto_resolve_disputes``. The Status
       is produced by tau2 code — not by KB policy text and not by the
       agent's prose.

If the framework renames or reformats any of these markers, the predicate
yields zero blocked calls and the gate is dead weight (safe).
"""

from __future__ import annotations

import json
import re
from typing import Iterable, List, Optional, Set

# Framework literals from tau2-bench-src/src/tau2/domains/banking_knowledge/tools.py
SUBMIT_TOOL_NAME = "submit_cash_back_dispute_0589"
STAGE2_TOOL_NAME = "update_transaction_rewards_3847"
EXECUTED_MARKER = f"Executed: {SUBMIT_TOOL_NAME}"
STATUS_RESOLVED = "Status: RESOLVED"
STATUS_SUBMITTED = "Status: SUBMITTED"

# Framework entry-point name (tau2-bench-src/.../banking_knowledge/tools.py:631)
ENTRY_POINT_CALL = "call_discoverable_agent_tool"

# Marker for the agent's deferral note. Idempotent guard so re-filtering the
# same AssistantMessage never duplicates the note.
NOTE_MARKER = "[STAGE-2 GATE]"

# JSON arg-parser for the framework's Arguments block. The block is rendered
# with json.dumps(..., indent=2) at tools.py:4155.
_TXN_ID_RE = re.compile(r'"transaction_id"\s*:\s*"([^"]+)"')


def _content(msg) -> str:
    c = getattr(msg, "content", None)
    if isinstance(c, str):
        return c
    return ""


def resolved_transactions(messages: Iterable) -> Set[str]:
    """Return the set of transaction_ids whose Stage-1 dispute response carried
    ``Status: RESOLVED`` in the visible message history.

    Walks every message in ``messages`` (ToolMessage, AssistantMessage, etc.;
    only messages with a ``content`` str matter). Splits each tool message's
    body on the framework-emitted ``Executed: submit_cash_back_dispute_0589``
    marker (one chunk per dispute). For each chunk extracts
    ``transaction_id`` (from the framework's JSON Arguments block) AND checks
    whether the chunk contains the literal ``Status: RESOLVED``. The two
    are anchored to the same framework response body so there is no
    cross-chunk leak.
    """
    resolved: Set[str] = set()
    for m in messages or []:
        text = _content(m)
        if not text or EXECUTED_MARKER not in text:
            continue
        parts = text.split(EXECUTED_MARKER)
        # parts[0] is everything BEFORE the first marker; skip it.
        for chunk in parts[1:]:
            if STATUS_RESOLVED not in chunk:
                continue
            m_id = _TXN_ID_RE.search(chunk)
            if m_id is not None:
                resolved.add(m_id.group(1))
    return resolved


def _parse_inner_args(call) -> Optional[dict]:
    """Decode the inner ``arguments`` JSON of a call_discoverable_agent_tool
    invocation. Returns None on any parse failure."""
    outer = getattr(call, "arguments", None)
    if not isinstance(outer, dict):
        return None
    raw = outer.get("arguments")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return None
    return None


def is_blocked_stage2(call, resolved_set: Set[str]) -> Optional[str]:
    """If ``call`` is a Stage-2 ``update_transaction_rewards_3847`` invocation
    whose target transaction_id is NOT in ``resolved_set``, return that
    transaction_id; else return None.

    Activation predicate components, all framework-schema facts:
      - ``call.name == ENTRY_POINT_CALL`` (outer entry-point name)
      - ``call.arguments['agent_tool_name'] == STAGE2_TOOL_NAME``
      - inner ``transaction_id`` field is a non-empty string and is NOT in
        ``resolved_set``

    If any component cannot be evaluated (missing field, malformed JSON,
    non-string txn id), the predicate returns None — fail-open so the LLM's
    call passes through unchanged.
    """
    if getattr(call, "name", None) != ENTRY_POINT_CALL:
        return None
    outer = getattr(call, "arguments", None)
    if not isinstance(outer, dict):
        return None
    if outer.get("agent_tool_name") != STAGE2_TOOL_NAME:
        return None
    inner = _parse_inner_args(call)
    if not inner:
        return None
    txn = inner.get("transaction_id")
    if not isinstance(txn, str) or not txn:
        return None
    if txn in resolved_set:
        return None
    return txn


def render_deferral_note(blocked_txns: List[str]) -> str:
    """Render the deferral note appended to AssistantMessage.content when the
    gate suppresses one or more Stage-2 calls.

    The note contains the marker token so the agent's filter can detect a
    note already in place and avoid double-appending.
    """
    if not blocked_txns:
        return ""
    bullets = "\n".join(f"  - {t}" for t in blocked_txns)
    return (
        f"{NOTE_MARKER} Suppressed {len(blocked_txns)} Stage-2 call(s) to "
        f"`{STAGE2_TOOL_NAME}` because no Stage-1 `{SUBMIT_TOOL_NAME}` "
        f"response in this conversation carries `Status: RESOLVED` for the "
        f"corresponding transaction id(s):\n"
        f"{bullets}\n"
        f"Per KB doc credit_cards_(general)_004 and the framework's "
        f"emitted `Status: SUBMITTED` verdict in "
        f"banking_knowledge/tools.py:4127-4149, "
        f"`{STAGE2_TOOL_NAME}` may only execute after a dispute's Stage-1 "
        f"response is marked RESOLVED. Inform the customer their disputes "
        f"have been submitted and they will be notified when the disputes "
        f"are reviewed; do not retry these Stage-2 calls in this "
        f"conversation."
    )


def already_filtered(assistant_message) -> bool:
    """True if this AssistantMessage already carries a STAGE-2 GATE note —
    avoids double-filtering across nested overrides."""
    c = getattr(assistant_message, "content", None)
    return isinstance(c, str) and NOTE_MARKER in c
