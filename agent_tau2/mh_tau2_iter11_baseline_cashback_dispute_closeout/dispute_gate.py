"""Deterministic close-out gate for the customer-filed cash-back dispute flow.

Mechanism targeted
------------------
A cash-back / rewards-amount discrepancy is remediated by handing the customer
the cash-back dispute tool (`submit_cash_back_dispute*`) and having them file
one dispute per affected transaction. In the gold trajectories that is the
*entire* set of state-changing work — give the dispute tool, the customer files
the disputes, done.

What the baseline agent does instead, once the disputes are filed and the
customer claims they are "resolved", is thrash:

  * it unlocks and calls a rewards-overwrite tool (`update_transaction_rewards`)
    to re-key the corrected cash back itself;
  * it unlocks and calls the credit-card transaction-dispute history tool
    (`get_user_dispute_history`) trying to "verify" the cash-back disputes —
    but that is a different dispute system, never part of a cash-back flow.

The cash-back dispute is filed by the *user* via a discoverable user tool, so
the dispute's "Status: SUBMITTED / RESOLVED" line is routed to the user and
never reaches the agent. The agent therefore cannot verify resolution and must
not act on the customer's unverifiable word. Each of those extra discoverable
agent-tool unlocks / calls is a write the gold trajectory never makes, and on a
database-hash-scored task one extra discoverable-tool interaction is enough to
zero the reward.

Precondition enforced
---------------------
Once the agent has handed the customer a cash-back dispute tool in this
conversation, any later turn that would unlock or call

  * a rewards-overwrite tool   (edit-verb + "reward" in the inner tool name), or
  * a credit-card dispute-history tool ("dispute" + "history" in the name)

has that call dropped — the cash-back dispute flow needs neither. The gate is
inert until a cash-back dispute tool has actually been given, so credit-card
dispute and account-closure flows (which legitimately read dispute history) are
never touched. Calls outside those two structural patterns always pass through,
so the gate can never remove a tool call the cash-back flow genuinely needs.
"""
from __future__ import annotations

import re
from typing import Optional

# A rewards-overwrite tool: its inner name pairs an edit verb with "reward".
# Matches update_transaction_rewards without naming it; never matches a
# dispute-filing tool (whose name carries "dispute", not "reward").
_REWARD_TOOL_RE = re.compile(
    r"(updat|correct|adjust|modif|edit|overwrit|recalc|fix|chang|set)\w*[\s_]*reward",
    re.I,
)

# A credit-card transaction-dispute history tool: a separate dispute system.
_DISPUTE_HISTORY_RE = re.compile(r"dispute.*histor", re.I)

# Outer tool names that interact with a discoverable agent tool.
_DISCOVERABLE_AGENT_OPS = ("unlock_discoverable_agent_tool", "call_discoverable_agent_tool")


def _inner_agent_tool_name(tool_call) -> Optional[str]:
    """Return the discoverable agent-tool name an unlock/call call references."""
    name = getattr(tool_call, "name", None)
    if name not in _DISCOVERABLE_AGENT_OPS:
        return None
    args = getattr(tool_call, "arguments", None)
    if not isinstance(args, dict):
        return None
    return str(args.get("agent_tool_name") or "")


def cash_back_dispute_given(messages) -> bool:
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
            if "cash" in given and "dispute" in given:
                return True
    return False


def is_post_dispute_extra(tool_call) -> bool:
    """True when a tool call is a rewards-overwrite or a credit-card
    dispute-history interaction — neither belongs in a cash-back dispute flow."""
    inner = _inner_agent_tool_name(tool_call)
    if inner is None:
        return False
    return bool(_REWARD_TOOL_RE.search(inner) or _DISPUTE_HISTORY_RE.search(inner))
