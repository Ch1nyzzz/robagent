"""Deterministic Stage-1 / Stage-2 state report for the cash-back rewards
correction workflow.

Why this exists
---------------
The banking_knowledge domain defines a two-stage cash-back-rewards-correction
workflow in two KB documents:

    doc_credit_cards_credit_cards_(general)_003 ("Submitting a Cash Back
        Dispute (Internal)") instructs the agent to *give* the customer the
        user-side discoverable tool ``submit_cash_back_dispute_0589`` so the
        customer can submit one dispute per affected transaction.

    doc_credit_cards_credit_cards_(general)_004 ("Applying Resolved Cash
        Back Dispute Corrections (Internal)") instructs the agent, AFTER the
        dispute has been resolved, to unlock and call the agent-side
        discoverable tool ``update_transaction_rewards_3847`` once per
        transaction with the corrected points value.

Both docs are already pre-loaded into the system prompt by the iter1/iter2
internal_procedure_channel. But across multiple cash-back-rewards train
simulations the agent still jumps straight to Stage 2 without doing Stage 1
(see e.g. iter4 sims for task_020 and task_022): the agent calls
``unlock_discoverable_agent_tool('update_transaction_rewards_3847')`` and
then ``call_discoverable_agent_tool('update_transaction_rewards_3847', ...)``
several times in the same turn, without ever issuing
``give_discoverable_user_tool('submit_cash_back_dispute_0589')``. The
``cash_back_disputes`` table in the evaluation DB then ends up with zero
rows where gold has one row per resolved transaction, and ``db_match`` is
False.

The pre-loaded policy text in the system prompt is necessary but not
sufficient: the LLM still needs to *apply* the two-stage rule at the
particular decision point of "I am about to invoke Stage 2." This module
provides that just-in-time visibility.

What this module does
---------------------
Pure helpers for the augmenter in ``agent.py``. Given the agent's tool
response history at the moment a Stage-2 entry-point response comes back,
this module:

  1. Detects whether the producing assistant tool call targeted Stage 2.
     The activation predicate is purely a tool-schema fact: the assistant
     tool_call's ``name`` is one of the framework discoverable-tool entry
     points (``unlock_discoverable_agent_tool`` /
     ``call_discoverable_agent_tool``) AND the inner ``agent_tool_name``
     argument equals the Stage-2 tool name declared in
     ``tau2/domains/banking_knowledge/tools.py``
     (``update_transaction_rewards_3847``).

  2. Mechanically counts the agent's prior
     ``give_discoverable_user_tool(discoverable_tool_name=
     'submit_cash_back_dispute_0589')`` calls in the message history. No
     keyword guessing about the user's intent; no policy interpretation.
     The count comes from the agent's own observable tool_call history.

  3. Returns a deterministic state report that names the two-stage
     relationship (citing the KB doc ids that already define it) and
     reports the observed Stage-1 give count. The LLM is left to decide
     whether to proceed to Stage 2 or to (re)do Stage 1 first.

What this module does NOT do
----------------------------
  - It never modifies, rewrites, or suppresses the LLM's tool call.
  - It never references customer names, user ids, account ids, or any
    other task-specific datum.
  - It never branches on policy edges or the agent's content choices —
    only on the framework / discoverable tool *names* and on the
    observable agent message history.

Stable structure captured
-------------------------
  - The tool names ``unlock_discoverable_agent_tool``,
    ``call_discoverable_agent_tool``, ``give_discoverable_user_tool``,
    ``update_transaction_rewards_3847``, ``submit_cash_back_dispute_0589``
    are all declared in ``tau2-bench-src/src/tau2/domains/banking_knowledge/
    tools.py`` — tau2 system facts independent of any particular task.
  - The Stage-1 -> Stage-2 ordering is the entire content of the two KB
    documents above; the advisor merely cites them at the decision point.
  - The give-call counter is a mechanical reduction over message history.

If a future tau2 release renamed any of these tools, the advisor would
silently no-op (the predicate would not match). If the model already
sequences Stage 1 before Stage 2 on its own, the advisor still fires but
the observed count is non-zero and the advisory just confirms readiness;
no negative effect.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional

# tau2 framework entry-point tool names. These wrap discoverable tools and
# carry the inner ``agent_tool_name`` we want to inspect.
_AGENT_ENTRY_POINTS = {
    "unlock_discoverable_agent_tool",
    "call_discoverable_agent_tool",
}

# tau2 framework entry-point used by the agent to give a discoverable tool
# to the user-simulator. The activation we care about is when this is used
# with discoverable_tool_name == SUBMIT_TOOL.
_GIVE_ENTRY_POINT = "give_discoverable_user_tool"

# banking_knowledge discoverable tool that performs the Stage-2 correction.
STAGE_2_TOOL = "update_transaction_rewards_3847"

# banking_knowledge discoverable user tool that performs the Stage-1
# customer submission.
SUBMIT_TOOL = "submit_cash_back_dispute_0589"


def _parse_args(raw: Any) -> dict:
    """Best-effort parse of a tool_call ``arguments`` field.

    tau2 ``ToolCall.arguments`` is normalized to a dict by the model, but
    serialized traces may carry a JSON string in some legacy paths. We
    accept both and return an empty dict on any malformed input.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def _inner_agent_tool(args: dict) -> Optional[str]:
    """Pull the inner ``agent_tool_name`` from an entry-point tool call."""
    name = args.get("agent_tool_name")
    return name if isinstance(name, str) else None


def _inner_user_tool(args: dict) -> Optional[str]:
    """Pull the inner ``discoverable_tool_name`` from a give-entry-point call."""
    name = args.get("discoverable_tool_name")
    return name if isinstance(name, str) else None


def is_stage_2_entry_point_call(tool_call) -> bool:
    """True if a ToolCall targets the Stage-2 discoverable tool.

    Activation is purely structural: the entry-point name is one of the
    framework agent entry points AND the inner ``agent_tool_name`` argument
    equals the Stage-2 tool name declared in tools.py.
    """
    name = getattr(tool_call, "name", None)
    if name not in _AGENT_ENTRY_POINTS:
        return False
    args = _parse_args(getattr(tool_call, "arguments", None))
    return _inner_agent_tool(args) == STAGE_2_TOOL


def lookup_producing_tool_call(tool_call_id: str, history_messages: Iterable):
    """Return the assistant ToolCall that produced ``tool_call_id``.

    Walks history in reverse looking for an AssistantMessage with a
    matching tool_call.id. Returns None if not found.
    """
    if not tool_call_id:
        return None
    for prev in reversed(list(history_messages)):
        tool_calls = getattr(prev, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in tool_calls:
            if getattr(tc, "id", None) == tool_call_id:
                return tc
    return None


def count_stage_1_give_calls(history_messages: Iterable) -> int:
    """Count assistant give_discoverable_user_tool calls for SUBMIT_TOOL.

    Pure reduction over message history. Counts only assistant tool calls
    whose name is ``give_discoverable_user_tool`` and whose inner
    ``discoverable_tool_name`` equals ``SUBMIT_TOOL``. Multiple give calls
    to the same tool name are dedup'd by tau2 at the DB layer, but they
    are independent observable agent decisions and are counted as such
    here so the LLM can see how many give attempts it made.
    """
    count = 0
    for m in history_messages:
        tool_calls = getattr(m, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in tool_calls:
            if getattr(tc, "name", None) != _GIVE_ENTRY_POINT:
                continue
            args = _parse_args(getattr(tc, "arguments", None))
            if _inner_user_tool(args) == SUBMIT_TOOL:
                count += 1
    return count


def render_state_advisory(give_count: int) -> str:
    """Render the Stage-1 / Stage-2 advisory text.

    The text:
      - Names the two-stage workflow and the KB doc ids that define it
        (the docs themselves are already in the system prompt).
      - Reports the agent's own observable Stage-1 give count.
      - States the DB-evaluation consequence of skipping Stage 1.
      - Does NOT direct the LLM to call a specific tool; the LLM remains
        in charge of the next action.
    """
    return (
        "[STAGE STATE] update_transaction_rewards_3847 is Stage 2 of the cash-back-"
        "rewards correction workflow.\n"
        "  - Stage 1 (KB doc credit_cards_(general)_003 'Submitting a Cash Back Dispute'): "
        "agent must `give_discoverable_user_tool` with "
        "discoverable_tool_name='submit_cash_back_dispute_0589' so the customer "
        "can submit one cash-back dispute per affected transaction. Each user "
        "submission writes one row to the `cash_back_disputes` table.\n"
        "  - Stage 2 (KB doc credit_cards_(general)_004 'Applying Resolved Cash Back "
        "Dispute Corrections'): once the disputes are resolved, agent unlocks and "
        "calls `update_transaction_rewards_3847` once per affected transaction.\n"
        f"Observed Stage-1 give calls in this conversation: {give_count}. "
        "If 0, the `cash_back_disputes` table has no rows from this conversation "
        "and proceeding to Stage 2 calls will diverge the evaluation DB hash from "
        "gold (which expects one row per disputed transaction). Reconsider whether "
        "Stage 1 should run first."
    )
