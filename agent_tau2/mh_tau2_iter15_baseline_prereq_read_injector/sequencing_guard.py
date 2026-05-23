"""Deterministic ordering gate for interacting multi-request conversations.

Mechanism targeted
------------------
Some banking tasks ask the agent to handle two requests in one conversation
whose order matters because an action taken for one request creates an account
condition that blocks the other. The clearest case in the train set:

  * task_053 — the customer wants both a transaction dispute and a
    credit-limit increase (CLI). The agent processes them in the order the
    customer mentioned them: it files the transaction dispute first, then runs
    the CLI workflow. Filing the dispute places a PENDING DISPUTE on the
    account, and a pending dispute is a documented blocking condition for a
    CLI. The agent's own eligibility analysis shows the customer qualifies,
    but the CLI is nevertheless DENIED with reason ``pending_disputes`` — a
    blocker the agent created itself. The gold trajectory APPROVES the CLI:
    it completes the credit-limit-increase decision before filing the dispute.

Because the reward is the database hash, a CLI ``deny`` where gold has an
``approve`` (different write value, plus the matching ``approve`` tool absent
from the discoverable-call set) zeroes the reward even though every other step
was correct.

This guard is reused unchanged from the iter14 candidate: it is paired here
with the new prerequisite-read injector because task_053 needs both the
correct CLI ordering and the two prerequisite reads to match gold.

What this guard does
--------------------
It watches for an assistant turn that FILES a transaction dispute while a CLI
workflow is already in progress (some credit-limit-increase tool has been
unlocked or called) but has not yet been decided (no approve/deny on record).
That turn is regenerated once with a transient note steering the agent to
complete the CLI decision first, then file the dispute.

Detection is structural — substring tests on the discoverable inner tool name,
never a hardcoded tool id — and the guard only ever fires when BOTH a CLI
workflow and a transaction-dispute filing are live in the same conversation,
so it cannot perturb a task that involves only one of the two. The nudge is
capped, so a turn can never be permanently suppressed.
"""
from __future__ import annotations

from typing import Optional

# Conversation-wide cap on reorder nudges. The per-turn loop already allows at
# most one steered retry; this only stops a stubborn model spinning in place.
MAX_NUDGES = 3

# Wrapper tool names that carry a discoverable inner tool in their arguments.
_DISCOVERABLE_WRAPPERS = {
    "call_discoverable_agent_tool",
    "unlock_discoverable_agent_tool",
    "give_discoverable_user_tool",
    "call_discoverable_user_tool",
}
_CALL_AGENT_TOOL = "call_discoverable_agent_tool"


def _inner_name(tool_call) -> str:
    """Lower-cased inner tool name of any discoverable-wrapper tool call.

    Returns "" for a non-wrapper call or a malformed payload.
    """
    if getattr(tool_call, "name", None) not in _DISCOVERABLE_WRAPPERS:
        return ""
    args = getattr(tool_call, "arguments", None)
    if not isinstance(args, dict):
        return ""
    for key in ("agent_tool_name", "discoverable_tool_name"):
        value = args.get(key)
        if value:
            return str(value).lower()
    return ""


def _is_agent_call(tool_call) -> bool:
    """True for an executed agent-side discoverable call (not an unlock/give)."""
    return getattr(tool_call, "name", None) == _CALL_AGENT_TOOL


def is_cli_tool(inner: str) -> bool:
    """True for any credit-limit-increase workflow tool (read, submit, decide)."""
    if not inner:
        return False
    if "credit_limit_increase" in inner:
        return True
    return "credit" in inner and "limit" in inner and "increase" in inner


def is_cli_decision(inner: str) -> bool:
    """True for the terminal CLI decision tool (approve / deny)."""
    return is_cli_tool(inner) and ("approve" in inner or "deny" in inner)


def is_dispute_filing(inner: str) -> bool:
    """True for the tool that FILES a credit-card transaction dispute.

    Keyed on both "file" and "dispute" so it cannot collide with a
    dispute-history *read* ("history") or a cash-back dispute submission
    (``submit_cash_back_dispute_*`` — carries no "file").
    """
    return "dispute" in inner and "file" in inner and "history" not in inner


def _calls(message) -> list:
    return list(getattr(message, "tool_calls", None) or [])


def turn_files_dispute(assistant_message) -> bool:
    """True when this assistant turn executes a transaction-dispute filing."""
    return any(
        _is_agent_call(tc) and is_dispute_filing(_inner_name(tc))
        for tc in _calls(assistant_message)
    )


def turn_decides_cli(assistant_message) -> bool:
    """True when this assistant turn itself records the CLI decision."""
    return any(
        _is_agent_call(tc) and is_cli_decision(_inner_name(tc))
        for tc in _calls(assistant_message)
    )


def cli_state(history) -> tuple:
    """Inspect prior assistant turns for the CLI workflow state.

    Returns (cli_in_progress, cli_decided):
      * cli_in_progress — any CLI tool was unlocked or called earlier;
      * cli_decided     — a CLI approve/deny call is already on record.
    """
    in_progress = False
    decided = False
    for m in history or []:
        if getattr(m, "role", None) != "assistant":
            continue
        for tc in _calls(m):
            inner = _inner_name(tc)
            if not inner:
                continue
            if is_cli_tool(inner):
                in_progress = True
            if _is_agent_call(tc) and is_cli_decision(inner):
                decided = True
    return in_progress, decided


_GUIDANCE = (
    "You are about to file a credit-card transaction dispute, but this "
    "customer also has a credit-limit-increase request that has not yet been "
    "approved or denied. Filing the dispute now places a PENDING DISPUTE on "
    "the account, and a pending dispute is a documented blocking condition "
    "for a credit-limit increase — it will force the increase to be denied. "
    "Do not file the dispute in this turn. First complete the "
    "credit-limit-increase decision: finish the documented eligibility checks "
    "and approve or deny the request. Only after the credit-limit-increase "
    "decision has been recorded should you file the transaction dispute."
)


class SequencingGuard:
    """Tracks request-ordering for one conversation."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._nudges_used = 0

    def register_nudge(self) -> None:
        self._nudges_used += 1

    def needs_reorder(self, assistant_message, history) -> Optional[str]:
        """Guidance note when a dispute filing should wait for the CLI decision.

        Returns None when the turn is fine: it files no dispute, no CLI
        workflow is live, the CLI is already decided, the same turn also
        records the CLI decision, or the nudge budget is exhausted.
        """
        if self._nudges_used >= MAX_NUDGES:
            return None
        if not turn_files_dispute(assistant_message):
            return None
        if turn_decides_cli(assistant_message):
            return None
        in_progress, decided = cli_state(history)
        if in_progress and not decided:
            return _GUIDANCE
        return None
