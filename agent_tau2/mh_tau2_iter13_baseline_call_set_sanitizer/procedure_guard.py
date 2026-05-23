"""Deterministic precondition gate for account-closure and CLI workflows.

Mechanism targeted
------------------
Credit-card *account closure* and *credit-limit-increase* (CLI) requests are
not single actions: the bank's knowledge base documents each as a multi-step
procedure with a mandatory prerequisite eligibility-check phase. Both
procedures require, before the terminal write, verifying that the account has
no active or pending transaction disputes and no pending replacement-card
orders (the closure retention protocol, Step 1; the CLI approval workflow,
Step 2 — "you MUST check ALL of the following eligibility criteria before
making an approval or denial decision").

On the failing closure/CLI simulations the agent skips that phase: it never
unlocks/calls the discoverable read tools for dispute history and pending
replacement orders, then emits the terminal write anyway (closes the account,
or approves/denies the increase). Because a blocking condition discovered by
those reads changes whether — and which — terminal write the gold trajectory
makes, the agent's write set diverges from gold and the database-hash reward
is zero.

What this guard does
--------------------
It tracks, from the agent's own earlier tool calls, whether the dispute-history
read and the pending-replacement-orders read have been performed in PRIOR
turns (so their results were actually observed). When a turn emits a terminal
closure or CLI-decision write before both reads are on record, the agent module
regenerates that turn once with a transient note steering it to complete the
documented prerequisite checks first.

Detection is structural — verb/noun substrings on the discoverable inner tool
name, never a hardcoded tool id — and the guard never blocks a write outright:
after a capped number of nudges the turn is allowed through, so a legitimate
write can never be permanently suppressed. Forced reads are read-only and
cannot themselves change the database hash.
"""
from __future__ import annotations

from typing import Optional

# Conversation-wide cap on "complete the prerequisite checks first" nudges.
# A single conversation may legitimately close several accounts (one task in
# the train set closes four), so the cap is generous; it exists only so a
# stubborn model can never spin in place.
MAX_NUDGES = 8

DISCOVERABLE_CALL = "call_discoverable_agent_tool"


def _inner_tool_name(tool_call) -> str:
    """Lower-cased inner tool name of a call_discoverable_agent_tool call.

    Returns "" for any other tool call or a malformed payload.
    """
    if getattr(tool_call, "name", None) != DISCOVERABLE_CALL:
        return ""
    args = getattr(tool_call, "arguments", None)
    if not isinstance(args, dict):
        return ""
    return str(args.get("agent_tool_name") or "").lower()


def is_dispute_history_read(inner: str) -> bool:
    """A discoverable read of the customer's transaction-dispute history."""
    return "dispute" in inner and "history" in inner


def is_pending_replacement_read(inner: str) -> bool:
    """A discoverable read of pending replacement-card orders.

    Keyed on both "pending" and "replacement" so it cannot collide with the
    write tool that *orders* a replacement card (that name carries neither
    "pending" nor "history").
    """
    return "pending" in inner and "replacement" in inner


def terminal_write_kind(inner: str) -> Optional[str]:
    """Classify a discoverable inner tool name as a terminal closure/CLI write.

    - "closure": permanently closes a credit-card account.
    - "cli_decision": approves or denies a credit-limit-increase request.

    Returns None for everything else, including the read tools above, the
    closure-reason *log* step, and the CLI *submission* step (which the
    documented procedure requires to happen before the eligibility checks).
    """
    if not inner:
        return None
    # Closure: close_credit_card_account_*. Exclude the closure-reason history
    # read ("history") and the closure-reason log write ("log"/"reason").
    if (
        "close" in inner
        and "account" in inner
        and "history" not in inner
        and "reason" not in inner
        and "log" not in inner
    ):
        return "closure"
    # CLI decision: approve_/deny_credit_limit_increase_*. The submission step
    # (submit_*) is intentionally not gated.
    if ("approve" in inner or "deny" in inner) and (
        "limit" in inner or "cli" in inner
    ):
        return "cli_decision"
    return None


def prereq_reads_seen(messages) -> set:
    """Set of prerequisite reads on record from prior assistant tool calls.

    Members are drawn from {"dispute", "replacement"}. Only assistant messages
    already in history are scanned, so a read counts only once its result was
    available to inform the decision — a read issued in the very same turn as
    the terminal write does not count.
    """
    seen: set = set()
    for m in messages or []:
        if getattr(m, "role", None) != "assistant":
            continue
        for call in getattr(m, "tool_calls", None) or []:
            inner = _inner_tool_name(call)
            if not inner:
                continue
            if is_dispute_history_read(inner):
                seen.add("dispute")
            if is_pending_replacement_read(inner):
                seen.add("replacement")
    return seen


def terminal_writes_in(assistant_message) -> set:
    """Kinds of terminal closure/CLI write emitted by this assistant turn."""
    kinds: set = set()
    for call in getattr(assistant_message, "tool_calls", None) or []:
        kind = terminal_write_kind(_inner_tool_name(call))
        if kind:
            kinds.add(kind)
    return kinds


_MISSING_LABEL = {
    "dispute": "the customer's active/pending transaction-dispute history",
    "replacement": "any pending replacement-card orders on the account",
}


class ProcedureGuard:
    """Tracks the closure/CLI prerequisite-check phase for one conversation."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._nudges_used = 0

    def register_nudge(self) -> None:
        self._nudges_used += 1

    def needs_prereq(self, assistant_message, history) -> Optional[str]:
        """Guidance note when a terminal write precedes the prerequisite reads.

        Returns None when the turn is fine (no terminal write, or both reads
        already on record, or the nudge budget is exhausted).
        """
        if self._nudges_used >= MAX_NUDGES:
            return None
        kinds = terminal_writes_in(assistant_message)
        if not kinds:
            return None
        seen = prereq_reads_seen(history)
        missing = [k for k in ("dispute", "replacement") if k not in seen]
        if not missing:
            return None
        return self._guidance(kinds, missing)

    @staticmethod
    def _guidance(kinds: set, missing: list) -> str:
        if "closure" in kinds:
            action = "close this credit-card account"
            procedure = "the credit-card closure / retention procedure"
        else:
            action = "finalize this credit-limit-increase decision"
            procedure = "the credit-limit-increase approval/denial procedure"
        missing_text = "; ".join(_MISSING_LABEL[m] for m in missing)
        return (
            f"You are about to {action}, but {procedure} requires a prerequisite "
            "eligibility-check phase that you have not completed. You have not "
            f"yet verified: {missing_text}. Per the documented procedure these "
            "checks are mandatory before the terminal write — a pending dispute "
            "or a pending replacement-card order is a blocking condition. Do "
            "not perform the closure or limit-increase decision yet. First "
            "retrieve the full procedure from the knowledge base if you have "
            "not, then unlock and call the discoverable read tools needed to "
            "check the items above (and any other documented prerequisite — "
            "outstanding balance, account age, prior retention attempts, "
            "request cooldown, payment history, utilization). Only after every "
            "prerequisite is checked and satisfied should you proceed; if a "
            "blocking condition exists, follow the procedure's alternative path "
            "instead of forcing the write."
        )
