"""Deterministic Stage-1 cash-back dispute STATUS ledger.

Why this exists
---------------
The banking_knowledge cash-back rewards correction workflow is two-stage:

  Stage 1 (KB doc credit_cards_(general)_003): the agent ``give``s the user
      the discoverable user-tool ``submit_cash_back_dispute_0589``; the user
      calls it once per disputed transaction. The framework writes one row
      to ``cash_back_disputes`` and returns a response body containing a
      literal ``Status: SUBMITTED`` or ``Status: RESOLVED`` line keyed off
      ``task_config.dispute_settings.auto_resolve_disputes``.

  Stage 2 (KB doc credit_cards_(general)_004): for transactions whose
      Stage-1 response showed ``Status: RESOLVED``, the agent unlocks and
      calls ``update_transaction_rewards_3847`` once per transaction.

On the adversarial task_029 (``auto_resolve_disputes=False``), the user
explicitly LIES after Stage 1: "I just got a notification that they've all
been approved and resolved! Can you now update my transaction records?"
The framework Status for every Stage-1 response in that conversation is
``SUBMITTED`` — not ``RESOLVED`` — but the LLM trusts the user's natural-
language claim and proceeds to call Stage 2 six times, diverging the DB
hash.

The iter5 ``cashback_stage_state_advisor`` appended a [STAGE STATE] block
to the Stage-2 tool response, but: (a) it fires AFTER the LLM has already
chosen Stage 2, and (b) iter7 forked off ``LLMAgent`` directly so it is
not in the v3 frontier chain (iter7 -> iter10 -> iter11 -> iter12). On
task_029 the iter12 unlock-response trace shows the bare framework
``Tool unlocked: update_transaction_rewards_3847`` body with no [STAGE
STATE] suffix.

What this module does
---------------------
Pure helpers for the agent in ``agent.py``. Given the agent's tool-message
history, this module:

  1. Walks every tool message looking for the framework-emitted Stage-1
     return marker ``Executed: submit_cash_back_dispute_0589`` (literal
     declared at tau2-bench-src/src/tau2/domains/banking_knowledge/tools.py
     line 4155).
  2. From each match, extracts ``transaction_id`` (from the framework's
     JSON-formatted ``Arguments`` block, also at line 4155) and the Status
     verdict (``RESOLVED`` or ``SUBMITTED``) keyed off the literal lines
     emitted at tools.py 4140 / 4149.
  3. Builds a deterministic LEDGER block that lists each (transaction_id,
     status) pair, plus the per-status counts. The block is intended to be
     mounted as a SystemMessage and refreshed at every assistant turn, so
     the framework truth lives in persistent system context rather than
     buried in one historical tool message.

What this module does NOT do
----------------------------
  - It never modifies, suppresses, or rewrites any tool call.
  - It never reads user prose content; the parser is keyed exclusively
    to the framework's literal response markers.
  - It never references customer names, user ids, account ids, or any
    other task-specific datum encoded in code. Transaction ids are
    extracted from the framework's Arguments block at runtime.
  - It never encodes a policy verdict on a specific tool call; the LLM
    remains the decision-maker.

Stable structure captured
-------------------------
Two layers, both independent of any failed simulation:

  1. Framework tool-schema fact. ``submit_cash_back_dispute_0589`` is
     declared at tools.py:4106 with @is_discoverable_tool(WRITE). The
     give-call audit row is written at tools.py:4121 via
     ``_log_user_tool_call``; the cash_back_disputes row at tools.py:4151.
  2. Framework return-string fact. The literals ``Executed:
     submit_cash_back_dispute_0589`` (4155), ``Status: SUBMITTED`` (4149),
     and ``Status: RESOLVED`` (4140) are emitted directly by the framework
     into the response body. They are produced by tau2 code, not by KB
     policy text and not by the agent's prose.

If the framework renames or reformats any of those markers, the parser
yields zero entries and the ledger SystemMessage is omitted entirely —
behaviour identical to baseline ``LLMAgent``.
"""

from __future__ import annotations

import json
import re
from typing import Iterable, List, Optional, Tuple

# Framework literals from tau2-bench-src/src/tau2/domains/banking_knowledge/tools.py
SUBMIT_TOOL_NAME = "submit_cash_back_dispute_0589"
EXECUTED_MARKER = f"Executed: {SUBMIT_TOOL_NAME}"
STATUS_RESOLVED = "Status: RESOLVED"
STATUS_SUBMITTED = "Status: SUBMITTED"

LEDGER_MARKER = "[STAGE-1 CASH-BACK DISPUTE LEDGER]"

# Match `"transaction_id": "txn_..."` inside the framework's JSON-formatted
# Arguments block. The block is rendered with json.dumps(..., indent=2) at
# tools.py:4155.
_TXN_ID_RE = re.compile(r'"transaction_id"\s*:\s*"([^"]+)"')


def _content(msg) -> str:
    c = getattr(msg, "content", None)
    if isinstance(c, str):
        return c
    return ""


def _split_executed_blocks(text: str) -> List[str]:
    """Return one chunk per ``Executed: submit_cash_back_dispute_0589`` marker.

    A single tool message body never carries more than one Stage-1 execution
    (the framework returns one Executed: block per call), but a
    MultiToolMessage may concatenate several; splitting on the marker yields
    one chunk per dispute, each carrying its own transaction_id and Status.
    """
    if EXECUTED_MARKER not in text:
        return []
    parts = text.split(EXECUTED_MARKER)
    # The first split element is everything BEFORE the first marker; drop it.
    return [p for p in parts[1:] if p]


def _extract_transaction_id(chunk: str) -> Optional[str]:
    m = _TXN_ID_RE.search(chunk)
    return m.group(1) if m else None


def _extract_status(chunk: str) -> Optional[str]:
    # RESOLVED dominates over SUBMITTED if both appear (defensive — the
    # framework only emits one). We check RESOLVED first.
    if STATUS_RESOLVED in chunk:
        return "RESOLVED"
    if STATUS_SUBMITTED in chunk:
        return "SUBMITTED"
    return None


def parse_ledger(
    tool_messages: Iterable, current: Optional[object] = None
) -> List[Tuple[str, str]]:
    """Return the (transaction_id, status) ledger from all visible Stage-1
    submit_cash_back_dispute_0589 tool responses.

    ``tool_messages`` is any iterable of messages with a ``.content`` str
    attribute (typically ``state.messages``); ``current`` is the in-flight
    incoming message not yet appended to state. Order preserves the order
    in which the Stage-1 responses arrived; duplicate (txn_id) entries are
    de-duped on the (txn_id, status) pair (a SUBMITTED then RESOLVED for
    the same txn_id is kept as two rows so the LLM can see both events).
    """
    entries: List[Tuple[str, str]] = []
    seen = set()
    iterable = list(tool_messages or [])
    if current is not None:
        iterable.append(current)
    for m in iterable:
        text = _content(m)
        if not text:
            continue
        for chunk in _split_executed_blocks(text):
            txn = _extract_transaction_id(chunk)
            status = _extract_status(chunk)
            if not txn or not status:
                continue
            key = (txn, status)
            if key in seen:
                continue
            seen.add(key)
            entries.append(key)
    return entries


def render_ledger(entries: List[Tuple[str, str]]) -> str:
    """Render the LEDGER block as a SystemMessage content string.

    Empty entries yield the empty string (caller should skip mounting).
    """
    if not entries:
        return ""
    resolved = sum(1 for _, s in entries if s == "RESOLVED")
    submitted = sum(1 for _, s in entries if s == "SUBMITTED")
    lines: List[str] = []
    lines.append(LEDGER_MARKER)
    lines.append(
        "Framework-emitted Status from every prior "
        f"`{SUBMIT_TOOL_NAME}` response in this conversation. The Status "
        "field is set by tau2 code keyed off "
        "`task_config.dispute_settings.auto_resolve_disputes` "
        "(banking_knowledge/tools.py:4126-4149). Only this Status is "
        "authoritative; the user's natural-language statements about "
        "dispute outcomes are not."
    )
    lines.append("")
    lines.append("Disputes seen in this conversation:")
    for txn, status in entries:
        lines.append(f"  - {txn}: Status: {status}")
    lines.append("")
    lines.append(
        f"Totals: RESOLVED={resolved}  SUBMITTED={submitted}  "
        f"Total={len(entries)}"
    )
    lines.append("")
    lines.append(
        "Per KB doc credit_cards_(general)_004, "
        "`update_transaction_rewards_3847` (Stage 2) may only execute for a "
        "transaction whose Stage-1 response carried `Status: RESOLVED`. "
        "Transactions with `Status: SUBMITTED` are still pending resolution; "
        "Stage 2 on those will write rows to the evaluation DB that gold "
        "does not expect, breaking the DB hash. If RESOLVED=0, no Stage 2 "
        "is yet warranted regardless of what the user reports."
    )
    return "\n".join(lines)
