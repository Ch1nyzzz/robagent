"""Deterministic helper for the cash-back-audit-worksheet candidate.

Several banking tasks ask the agent to review a customer's credit-card
transactions for incorrect cash-back / rewards and to dispute or correct the
ones that are wrong. The reward for these tasks is the database hash, so the
agent must act on EXACTLY the set of genuinely miscalculated transactions:
disputing a transaction that was actually correct breaks the hash just as
surely as missing a wrong one.

In the observed failures the agent eyeballs the transaction list in prose and
gets the set wrong — it applies a promotional multiplier to transactions that
fall outside the promotion window, judges rounding-level (~1 point) differences
inconsistently, and so disputes too many or too few transactions.

This module contributes two deterministic things and decides nothing about
correctness itself:

1. It parses the raw transaction records out of tool results and re-presents
   them as a clean, fixed-width worksheet (id, card, category, amount, date,
   recorded reward). The agent then reasons from structured facts instead of
   re-reading prose, and every transaction date is in front of it.
2. It tracks whether any earning-rate / promotional knowledge-base document has
   been retrieved, so the agent module can steer the agent to look those up
   before it submits a dispute or rewards adjustment.

The worksheet is only reformatted facts, so injecting it can never introduce a
wrong verdict; the steering nudge fires at most once and never blocks an
action, so a legitimate dispute can never be permanently suppressed.
"""
from __future__ import annotations

import re
from typing import Optional

# --- transaction-record parsing -------------------------------------------------

# Records arrive inside tool results in a "key: value" block layout. Splitting on
# the transaction_id line yields one chunk of fields per transaction.
_FIELD_RES = {
    "card": re.compile(r"credit_card_type:\s*([^\n]+)"),
    "amount": re.compile(r"transaction_amount:\s*\$?\s*([0-9][0-9,]*\.?[0-9]*)"),
    "date": re.compile(r"transaction_date:\s*([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4})"),
    "category": re.compile(r"category:\s*([^\n]+)"),
    "merchant": re.compile(r"merchant_name:\s*([^\n]+)"),
    "recorded": re.compile(r"rewards_earned:\s*([0-9][0-9,]*)"),
}
_ID_RE = re.compile(r"^\s*(txn_\w+)")


def _clean(value: str) -> str:
    return value.strip().strip(".").strip()


def parse_transactions(text: str) -> dict:
    """Return {transaction_id: {fields}} for every transaction record in `text`."""
    out: dict = {}
    if not isinstance(text, str) or "transaction_id:" not in text:
        return out
    chunks = text.split("transaction_id:")
    for chunk in chunks[1:]:
        id_match = _ID_RE.match(chunk)
        if not id_match:
            continue
        txn_id = id_match.group(1)
        # Stop the chunk at the next record boundary so fields don't bleed over.
        boundary = chunk.find("Record ID:")
        body = chunk if boundary == -1 else chunk[:boundary]
        record = {"id": txn_id}
        for key, pattern in _FIELD_RES.items():
            found = pattern.search(body)
            record[key] = _clean(found.group(1)) if found else None
        out[txn_id] = record
    return out


# --- knowledge-base document detection -----------------------------------------


def looks_like_rate_doc(text) -> bool:
    """True when a tool result looks like a card earning-rate document."""
    if not isinstance(text, str):
        return False
    t = text.lower()
    if "%" not in t:
        return False
    if "cash back" in t or "cashback" in t:
        return "earn" in t or "rate" in t
    if "earning rate" in t:
        return True
    return False


def looks_like_promo_doc(text) -> bool:
    """True when a tool result mentions a promotional rewards offer."""
    if not isinstance(text, str):
        return False
    t = text.lower()
    if "promo" in t or "double cash back" in t or "limited-time" in t:
        return True
    if "2x" in t and ("cash" in t or "reward" in t):
        return True
    if "bonus" in t and ("cash back" in t or "cashback" in t):
        return True
    return False


# --- cash-back write-action detection ------------------------------------------


def _inner_tool_names(call) -> list[str]:
    """Discoverable-tool names referenced by a tool call (lower-cased)."""
    names: list[str] = []
    name = getattr(call, "name", None)
    if isinstance(name, str):
        names.append(name.lower())
    args = getattr(call, "arguments", None)
    if isinstance(args, dict):
        for key in ("discoverable_tool_name", "agent_tool_name"):
            value = args.get(key)
            if isinstance(value, str):
                names.append(value.lower())
    return names


def is_cashback_write_call(call) -> bool:
    """True when a tool call submits a cash-back dispute or a rewards update.

    Keyed off generic English tool-name words ("dispute", "reward(s)"), never
    off task-specific identifiers. Read-only lookups ("get_..._dispute_...")
    are excluded so they do not consume the steering nudge.
    """
    for nm in _inner_tool_names(call):
        if nm.startswith("get_"):
            continue
        if "dispute" in nm:
            return True
        if "reward" in nm and ("update" in nm or "correct" in nm or "adjust" in nm):
            return True
    return False


def message_has_cashback_write(assistant_message) -> bool:
    calls = getattr(assistant_message, "tool_calls", None) or []
    return any(is_cashback_write_call(c) for c in calls)


# --- the guard ------------------------------------------------------------------

_WORKSHEET_HEADER = (
    "Transaction worksheet (parsed verbatim from the tool results in this "
    "conversation — treat these values as authoritative):"
)

_PROTOCOL_REMINDER = (
    "Before you dispute or adjust ANY transaction, verify it against this "
    "worksheet:\n"
    "- For every card type above, make sure you have retrieved its earning-rate "
    "policy AND checked the knowledge base for any active promotion.\n"
    "- A promotion changes the rate only for transactions whose date is inside "
    "the promotion's eligibility window — check each transaction_date, and never "
    "apply a promotional rate outside that window.\n"
    "- expected_points = transaction_amount * applicable_rate_percent "
    "(1% cash back on $1 = 1 point).\n"
    "- A difference of about 1 point between expected and recorded is ordinary "
    "rounding, NOT an error: leave that transaction alone.\n"
    "- Dispute or adjust EXACTLY the transactions that are materially "
    "miscalculated — do not include a correct transaction, and do not skip a "
    "wrong one. Re-derive the set from this worksheet even if it contradicts an "
    "earlier draft analysis."
)


class CashBackAuditGuard:
    """Parses transactions, tracks KB retrieval, and builds audit worksheets."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._transactions: dict = {}
        self._rate_doc_seen = False
        self._promo_doc_seen = False
        self._audit_mode = False
        self._first_action_handled = False

    # -- observation ------------------------------------------------------------

    def observe_tool_result(self, content) -> None:
        """Update the parsed state from one tool-result string."""
        if not isinstance(content, str):
            return
        for txn_id, record in parse_transactions(content).items():
            self._transactions[txn_id] = record
        if looks_like_rate_doc(content):
            self._rate_doc_seen = True
        if looks_like_promo_doc(content):
            self._promo_doc_seen = True

    # -- audit-mode bookkeeping -------------------------------------------------

    @property
    def audit_mode(self) -> bool:
        return self._audit_mode

    def is_first_cashback_turn(self, assistant_message) -> bool:
        """True the first time an assistant turn submits a cash-back write."""
        if self._first_action_handled:
            return False
        return message_has_cashback_write(assistant_message)

    def enter_audit_mode(self) -> None:
        self._audit_mode = True
        self._first_action_handled = True

    # -- worksheet / notes ------------------------------------------------------

    def _card_types(self) -> list[str]:
        seen: list[str] = []
        for record in self._transactions.values():
            card = record.get("card")
            if card and card not in seen:
                seen.append(card)
        return seen

    def worksheet(self) -> str:
        """A fixed-width table of every parsed transaction (most recent batch)."""
        if not self._transactions:
            return ""
        rows = [
            "  {:<22} {:<30} {:<13} {:>11} {:<11} {:>10}".format(
                "transaction_id", "card", "category", "amount", "date", "recorded"
            )
        ]
        for record in list(self._transactions.values())[:40]:
            amount = record.get("amount")
            amount_str = f"${amount}" if amount else "?"
            rows.append(
                "  {:<22} {:<30} {:<13} {:>11} {:<11} {:>10}".format(
                    (record.get("id") or "?")[:22],
                    (record.get("card") or "?")[:30],
                    (record.get("category") or "?")[:13],
                    amount_str[:11],
                    (record.get("date") or "?")[:11],
                    (record.get("recorded") or "?")[:10],
                )
            )
        return _WORKSHEET_HEADER + "\n" + "\n".join(rows)

    def context_note(self) -> Optional[str]:
        """Reference worksheet attached to every turn once the audit starts."""
        if not self._audit_mode:
            return None
        sheet = self.worksheet()
        if not sheet:
            return None
        return sheet + "\n\n" + _PROTOCOL_REMINDER

    def first_action_note(self) -> str:
        """Stronger note used when regenerating the first cash-back write turn."""
        parts: list[str] = []
        sheet = self.worksheet()
        if sheet:
            parts.append(sheet)
        missing: list[str] = []
        if not self._rate_doc_seen:
            missing.append(
                "the earning-rate policy for the card type(s) involved"
            )
        if not self._promo_doc_seen:
            missing.append(
                "any active promotional offer for the card type(s) involved"
            )
        if missing:
            parts.append(
                "You are about to submit a cash-back dispute or rewards "
                "adjustment, but you have not yet retrieved from the knowledge "
                "base: " + "; ".join(missing) + ". Do not submit yet — search "
                "the knowledge base for that information first, then re-check "
                "every transaction."
            )
        cards = self._card_types()
        if cards:
            parts.append("Card types in this customer's transactions: " + ", ".join(cards) + ".")
        parts.append(_PROTOCOL_REMINDER)
        return "\n\n".join(parts)
