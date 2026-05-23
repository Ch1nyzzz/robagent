"""Deterministic guard for the cash-back-dispute-workflow candidate.

Banking tasks scored by the database hash require the agent's set of
state-modifying tool calls to exactly equal the gold set. On cash-back
discrepancy tasks the gold remediation is the *dispute* workflow: the agent
hands the customer a cash-back-dispute tool and a dispute is filed for each
genuinely wrong transaction. The agent instead reaches for the tool that
*directly overwrites* a transaction's stored rewards value
(``update_transaction_rewards``) — either in place of filing disputes, or in
addition to them. Both break the DB hash: a direct-edit tool used as a
first-line fix is the wrong tool, and edits layered on top of disputes are
extra writes gold never makes.

This guard detects, in a single turn, a call to a tool that directly edits a
transaction's stored rewards value. The agent module reacts by regenerating
the turn once with a note steering it back to the dispute workflow. The guard
is advisory: it nudges at most once per conversation and never blocks the call
outright, so a legitimate post-resolution correction can still go through.
"""
from __future__ import annotations

# Keys under which a discoverable-tool wrapper carries the real tool name.
_INNER_NAME_KEYS = ("agent_tool_name", "discoverable_tool_name", "tool_name")

# Verbs that indicate a tool *overwrites* an existing stored value.
_EDIT_VERBS = (
    "update",
    "adjust",
    "edit",
    "set",
    "correct",
    "modify",
    "change",
    "overwrite",
)

# Total number of "use the dispute workflow" nudges allowed per conversation.
# One is normally enough; the cap only stops a stubborn model from looping.
MAX_NUDGES = 1

DISPUTE_WORKFLOW_NOTE = (
    "You are about to directly overwrite the rewards value stored on a "
    "transaction. For a customer who reports an incorrect cash-back / rewards "
    "amount, bank policy makes the cash-back DISPUTE process the remediation: "
    "give the customer the cash-back dispute tool and have a dispute filed for "
    "each transaction that is genuinely wrong. Filing the dispute IS the "
    "resolution step. Directly editing the stored rewards value is only "
    "appropriate as a post-resolution correction that the dispute procedure "
    "explicitly authorizes after a dispute has been resolved — never as a "
    "first-line fix and never layered on top of a dispute you just filed. "
    "Do not take this shortcut. Re-check the knowledge-base cash-back dispute "
    "procedure; unless it explicitly directs you to apply a correction right "
    "now, file cash-back disputes instead of editing the rewards value "
    "directly. Also confirm you are acting only on transactions that are "
    "materially wrong — a difference of about one point is ordinary rounding, "
    "not an error."
)


def _inner_tool_name(tool_call) -> str:
    """Best-effort real tool name behind a (possibly wrapped) tool call."""
    name = getattr(tool_call, "name", None) or ""
    args = getattr(tool_call, "arguments", None)
    if isinstance(args, dict):
        for key in _INNER_NAME_KEYS:
            val = args.get(key)
            if isinstance(val, str) and val:
                # The wrapper name (e.g. call_discoverable_agent_tool) is
                # uninformative; the inner name is what matters.
                return val
    return name


def is_direct_rewards_edit(tool_name) -> bool:
    """True when a tool name names a direct overwrite of transaction rewards.

    The detector keys off a tool that mentions both a transaction and its
    rewards together with an edit verb (``update_transaction_rewards`` and any
    similarly named variant). It deliberately does NOT match the dispute tool
    or any read/lookup tool. A miss only degrades to baseline behaviour; a
    false hit only costs one advisory regeneration.
    """
    if not isinstance(tool_name, str):
        return False
    t = tool_name.lower()
    if "transaction" not in t:
        return False
    if "reward" not in t and "cash_back" not in t and "cashback" not in t:
        return False
    if "dispute" in t:  # the dispute tool is the correct workflow, never flag it
        return False
    return any(verb in t for verb in _EDIT_VERBS)


class CashBackRemediationGuard:
    """Steers cash-back remediation toward the dispute workflow."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._nudges_used = 0

    @staticmethod
    def _calls_direct_rewards_edit(assistant_message) -> bool:
        calls = getattr(assistant_message, "tool_calls", None) or []
        for call in calls:
            if is_direct_rewards_edit(_inner_tool_name(call)):
                return True
        return False

    def needs_redirect(self, assistant_message) -> bool:
        """True when this turn directly edits transaction rewards.

        Returns False once the per-conversation nudge budget is spent, so the
        edit is never permanently suppressed.
        """
        if self._nudges_used >= MAX_NUDGES:
            return False
        return self._calls_direct_rewards_edit(assistant_message)

    def register_nudge(self) -> None:
        self._nudges_used += 1

    def guidance(self) -> str:
        return DISPUTE_WORKFLOW_NOTE
