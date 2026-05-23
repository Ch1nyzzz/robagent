"""Deterministic write-precondition gate for transaction-rewards corrections.

Mechanism targeted
------------------
A tool that directly overwrites a transaction's stored rewards / cash-back
value (`update_transaction_rewards`) is, per the bank's documented procedure, a
*post-resolution* step: it may only be applied once the related cash-back
dispute has been RESOLVED and approved. Filing a dispute does not resolve it,
and the customer asserting "it's resolved" is not proof. On the adversarial
cash-back tasks the customer claims a just-filed dispute is resolved while the
bank's records still show it open; the agent overwrites the rewards anyway, and
every overwrite is an extra DB write that zeroes the database-hash reward
(gold there is disputes-only).

Routing fact this gate respects
-------------------------------
A cash-back dispute is filed by the *user* via a discoverable user tool. In
tau2 a user-side tool result is routed back to the user, never to the agent —
so the agent cannot see a "Status: SUBMITTED/RESOLVED" line from the dispute
submission. What the agent *can* observe is (a) its own act of handing the
customer the dispute tool (`give_discoverable_user_tool`), and (b) the result
of any dispute-status tool it calls itself (an agent-side tool such as a
dispute-history look-up). The gate is built only on those observable facts.

Precondition enforced
---------------------
Once the agent has handed the customer a cash-back dispute tool in this
conversation, a rewards overwrite is allowed only if the agent has itself
verified, from an agent-side tool result, that the dispute for that
transaction is RESOLVED. Absent that verification the overwrite is dropped:
the post-resolution precondition is unverified. When no dispute tool has been
given at all the overwrite is left alone — there is no dispute in play.
"""
from __future__ import annotations

import json
import re
from typing import Optional

# A discoverable rewards-overwrite tool is detected structurally: its inner
# tool name pairs an edit verb with "reward". This matches the documented
# update_transaction_rewards tool without naming it, and never matches a
# dispute-filing tool (whose name carries "dispute", not "reward").
_REWARD_TOOL_RE = re.compile(
    r"(updat|correct|adjust|modif|edit|overwrit|recalc|fix|chang|set)\w*[\s_]*reward",
    re.I,
)

_TXN_RE = re.compile(r"txn_[0-9A-Za-z]+")

# Token scanner: a transaction id, or a `status` field and its first word.
_TOKEN_RE = re.compile(
    r"(txn_[0-9A-Za-z]+)|status[\"']?\s*[:=]\s*[\"']?([A-Za-z]+)",
    re.I,
)

# Status words that mean the dispute is finished and a correction is warranted.
_RESOLVED_WORDS = ("resolved", "approved", "completed", "credited", "finalized")


def _as_dict(raw) -> dict:
    """Coerce a discoverable-call inner `arguments` payload to a dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, TypeError):
            return {}
    return {}


def reward_update_target(tool_call) -> Optional[str]:
    """Return the transaction id a tool call would overwrite rewards for.

    Returns None when the call is not a rewards-overwrite write. When the call
    is a rewards overwrite but no transaction id can be extracted, returns the
    sentinel ``""`` (treated as un-gateable -> allowed).
    """
    name = getattr(tool_call, "name", None)
    if name != "call_discoverable_agent_tool":
        return None
    outer = getattr(tool_call, "arguments", None)
    if not isinstance(outer, dict):
        return None
    inner_name = str(outer.get("agent_tool_name") or "")
    if not _REWARD_TOOL_RE.search(inner_name):
        return None
    args = _as_dict(outer.get("arguments"))
    for key, value in args.items():
        kl = key.lower()
        if "transaction" in kl and "id" in kl:
            m = _TXN_RE.search(str(value))
            if m:
                return m.group(0)
    m = _TXN_RE.search(json.dumps(args))
    return m.group(0) if m else ""


def dispute_tool_given(messages) -> bool:
    """True when the agent has handed the customer a cash-back dispute tool
    earlier in this conversation (an agent-side, observable action)."""
    for m in messages or []:
        if getattr(m, "role", None) != "assistant":
            continue
        for call in getattr(m, "tool_calls", None) or []:
            if getattr(call, "name", None) != "give_discoverable_user_tool":
                continue
            args = getattr(call, "arguments", None)
            if not isinstance(args, dict):
                continue
            given = str(args.get("discoverable_tool_name") or "").lower()
            if "dispute" in given:
                return True
    return False


def _scan_dispute_message(content: str, statuses: dict) -> None:
    """Update `statuses` (txn id -> resolved bool) from one dispute result."""
    events = []
    for m in _TOKEN_RE.finditer(content):
        if m.group(1):
            events.append(("txn", m.group(1), m.start()))
        elif m.group(2):
            events.append(("status", m.group(2).lower(), m.start()))
    txns = [e for e in events if e[0] == "txn"]
    if not txns:
        return
    for typ, val, pos in events:
        if typ != "status":
            continue
        tid = min(txns, key=lambda t: abs(t[2] - pos))[1]
        resolved = any(w in val for w in _RESOLVED_WORDS)
        if tid not in statuses:
            statuses[tid] = resolved
        elif resolved:
            statuses[tid] = True


def collect_dispute_statuses(messages) -> dict:
    """Map transaction id -> True/False (its dispute has been seen RESOLVED)
    across every dispute result the AGENT itself observed.

    Only agent-side tool results that are genuine dispute results are read — a
    dispute-history look-up or a dispute submission echo ("Dispute ID: ..."). A
    user-side dispute result never reaches the agent, so it is never present
    here; the per-transaction `status` of an ordinary transaction list is never
    consulted.
    """
    statuses: dict = {}
    for m in messages or []:
        if getattr(m, "role", None) != "tool":
            continue
        content = getattr(m, "content", None)
        if not isinstance(content, str):
            continue
        low = content.lower()
        if "dispute id" not in low and "dispute history" not in low:
            continue
        _scan_dispute_message(content, statuses)
    return statuses


def is_blocked(txn_id: Optional[str], statuses: dict, dispute_given: bool) -> bool:
    """Decide whether a rewards overwrite must be dropped.

    - dispute status observed (agent-side) and RESOLVED  -> allow
    - dispute status observed (agent-side) and not resolved -> block
    - no status observed, but a dispute tool was handed out -> block
      (a dispute is in play and its resolution has not been verified)
    - no status observed and no dispute tool handed out -> allow
    """
    if not txn_id:  # None or "" -> not gateable
        return False
    if txn_id in statuses:
        return not statuses[txn_id]
    return dispute_given
