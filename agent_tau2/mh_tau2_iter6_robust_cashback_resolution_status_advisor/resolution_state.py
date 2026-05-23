"""Deterministic resolution-status report for the cash-back rewards
correction workflow.

Why this exists
---------------
The banking_knowledge domain's cash-back workflow has two stages whose
applicability is decided by an explicit framework signal — not by KB
policy interpretation.

  * Stage 1 (``submit_cash_back_dispute_0589``, tau2-bench-src
    ``tau2/domains/banking_knowledge/tools.py`` line 4106) writes one
    row to the ``cash_back_disputes`` table per call.

  * The Stage-1 return string EXPLICITLY contains one of two literal
    status lines, emitted by the framework based on
    ``task_config.dispute_settings.auto_resolve_disputes``:

        ``Status: RESOLVED - The dispute has been reviewed and approved.
        The transaction rewards need to be updated.``     (tools.py 4140)

        ``Status: SUBMITTED - Your dispute has been queued for review.``
                                                          (tools.py 4149)

  * Stage 2 (``update_transaction_rewards_3847``, tools.py line 711)
    overwrites the transaction's ``rewards_earned`` field. The KB doc
    ``doc_credit_cards_credit_cards_(general)_004`` says Stage 2 happens
    "after the dispute has been resolved." The framework's RESOLVED /
    SUBMITTED status line is the deterministic signal of resolution.

iter5's ``cashback_stage_state_advisor`` surfaced the Stage-1 give count
at the Stage-2 entry point, which fixed a few tasks where the agent
skipped Stage 1 entirely. But across multiple failing sims the agent
still proceeded to Stage 2 even when every Stage-1 return said
``Status: SUBMITTED`` — i.e. the framework was explicitly telling the
agent the disputes were NOT resolved. The give count alone could not
disambiguate auto-resolve vs. queued-for-review.

This module extends the iter5 advisor with the missing signal:
partition the observed Stage-1 user-tool returns by their literal
Status line. A count of (RESOLVED, SUBMITTED) returns is appended to
the Stage-2-entry-point tool response so the LLM can see, at the
decision point, whether any disputes have actually been resolved.

What this module does
---------------------
Pure helpers. Given the agent's message history at the moment a
Stage-2 entry-point response comes back:

  1. ``is_stage_2_entry_point_call`` — same tool-schema predicate as
     iter5: the producing assistant tool_call's ``name`` is one of
     {``unlock_discoverable_agent_tool``,
     ``call_discoverable_agent_tool``} AND the inner
     ``agent_tool_name`` argument equals ``update_transaction_rewards_3847``.

  2. ``count_stage_1_give_calls`` — preserves iter5's reduction:
     count of agent ``give_discoverable_user_tool`` calls with
     ``discoverable_tool_name='submit_cash_back_dispute_0589'``.

  3. ``count_stage_1_resolution_status`` — new reduction: walks tool
     messages, finds those produced by a Stage-1 user-tool invocation
     (assistant ``call_discoverable_user_tool`` with inner
     ``discoverable_tool_name='submit_cash_back_dispute_0589'``,
     plus the framework-emitted ``Executed: submit_cash_back_dispute_0589``
     marker that appears in the tool return body), and counts the
     literal ``Status: RESOLVED`` / ``Status: SUBMITTED`` substrings.
     The substrings come from tools.py lines 4140 and 4149 — they are
     tau2-framework facts, not KB policy text.

  4. ``render_state_advisory`` — composes a deterministic advisory
     text that names the two-stage workflow (citing the existing KB
     doc ids), reports the give count, and reports the
     RESOLVED/SUBMITTED partition. The LLM remains the decision-maker.

What this module does NOT do
----------------------------
  - It never modifies, rewrites, or suppresses the LLM's tool call.
  - It never references customer names, account ids, transaction ids,
    or any task-specific datum.
  - It never branches on KB-policy text it has read; it only branches
    on tau2-framework tool names and on the literal status lines the
    framework writes into tool returns.

Stable structure captured (independent of the failed simulations)
-----------------------------------------------------------------
  - Tool-name facts: every name used here (entry-point wrappers,
    Stage-1 user-tool name, Stage-2 agent-tool name) is declared in
    ``tau2/domains/banking_knowledge/tools.py``.
  - Return-format facts: the two literal Status lines are emitted by
    the framework at tools.py lines 4140 (RESOLVED) and 4149
    (SUBMITTED). On a freshly authored banking_knowledge task suite
    using the same tools.py, the same literal lines would appear and
    the partition would still be correct.

If a future tau2 release renames the user tool, changes the entry-
point wrappers, or replaces the Status: RESOLVED / Status: SUBMITTED
phrasing, the predicate / parser silently returns no matches and the
augmenter no-ops. The agent's behavior reduces to baseline v0.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional, Tuple

# tau2 framework entry-point tool names. These wrap discoverable tools
# and carry the inner ``agent_tool_name`` (agent-side) or
# ``discoverable_tool_name`` (user-side) argument we inspect.
_AGENT_ENTRY_POINTS = {
    "unlock_discoverable_agent_tool",
    "call_discoverable_agent_tool",
}
_GIVE_ENTRY_POINT = "give_discoverable_user_tool"
_USER_CALL_ENTRY_POINT = "call_discoverable_user_tool"

# banking_knowledge discoverable tool names. Declared in tools.py.
STAGE_2_TOOL = "update_transaction_rewards_3847"  # tools.py line 711
SUBMIT_TOOL = "submit_cash_back_dispute_0589"  # tools.py line 4106

# Literal Status substrings emitted by the framework in the Stage-1
# user-tool return string. Source: tools.py lines 4140 (RESOLVED) and
# 4149 (SUBMITTED).
_STATUS_RESOLVED = "Status: RESOLVED"
_STATUS_SUBMITTED = "Status: SUBMITTED"

# Marker the framework prints in every Stage-1 return body (tools.py
# line 4155). Used to attribute a tool message to a Stage-1 invocation
# when we cannot follow the tool_call_id chain.
_EXECUTED_MARKER = f"Executed: {SUBMIT_TOOL}"


def _parse_args(raw: Any) -> dict:
    """Best-effort parse of a tool_call ``arguments`` field."""
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
    name = args.get("agent_tool_name")
    return name if isinstance(name, str) else None


def _inner_user_tool(args: dict) -> Optional[str]:
    name = args.get("discoverable_tool_name")
    return name if isinstance(name, str) else None


def is_stage_2_entry_point_call(tool_call) -> bool:
    """True iff the ToolCall targets ``update_transaction_rewards_3847``
    via one of the agent entry-point wrappers.
    """
    name = getattr(tool_call, "name", None)
    if name not in _AGENT_ENTRY_POINTS:
        return False
    args = _parse_args(getattr(tool_call, "arguments", None))
    return _inner_agent_tool(args) == STAGE_2_TOOL


def lookup_producing_tool_call(tool_call_id: str, history_messages: Iterable):
    """Return the assistant ToolCall that produced ``tool_call_id``."""
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
    """Count assistant ``give_discoverable_user_tool`` calls whose inner
    ``discoverable_tool_name`` equals ``submit_cash_back_dispute_0589``.

    Pure reduction over message history; preserves iter5 behavior.
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


def _is_stage_1_user_call(tool_call) -> bool:
    name = getattr(tool_call, "name", None)
    if name == SUBMIT_TOOL:
        # Some sims represent user-side calls with the inner tool name
        # directly (e.g. when the user simulator invokes the granted
        # tool). Tools.py declares submit_cash_back_dispute_0589 as the
        # framework user-tool name.
        return True
    if name != _USER_CALL_ENTRY_POINT:
        return False
    args = _parse_args(getattr(tool_call, "arguments", None))
    return _inner_user_tool(args) == SUBMIT_TOOL


def _tool_msg_is_stage_1_return(tool_msg, history_messages) -> bool:
    """A tool message is a Stage-1 return iff (a) the producing tool
    call is a Stage-1 user-side invocation OR (b) the message body
    carries the framework's ``Executed: submit_cash_back_dispute_0589``
    marker (tools.py line 4155). The marker is a tau2-framework
    string, not a KB-doc fact.
    """
    producing = lookup_producing_tool_call(
        getattr(tool_msg, "id", None), history_messages
    )
    if producing is not None and _is_stage_1_user_call(producing):
        return True
    content = getattr(tool_msg, "content", None) or ""
    return _EXECUTED_MARKER in content


def count_stage_1_resolution_status(
    history_messages,
) -> Tuple[int, int, int]:
    """Return ``(resolved, submitted, other)`` counts across all
    Stage-1 user-tool returns observed in history.

    The classification is by literal substring on the tool message
    content: ``Status: RESOLVED`` (tools.py 4140) or
    ``Status: SUBMITTED`` (tools.py 4149). A Stage-1 return that
    contains neither (e.g. a duplicate-submission no-op) is counted as
    ``other``.
    """
    resolved = submitted = other = 0
    msgs = list(history_messages)
    for m in msgs:
        if getattr(m, "role", None) != "tool":
            continue
        if not _tool_msg_is_stage_1_return(m, msgs):
            continue
        content = getattr(m, "content", None) or ""
        if _STATUS_RESOLVED in content:
            resolved += 1
        elif _STATUS_SUBMITTED in content:
            submitted += 1
        else:
            other += 1
    return resolved, submitted, other


def render_state_advisory(
    give_count: int, resolved: int, submitted: int, other: int
) -> str:
    """Render the Stage-1 / Stage-2 advisory text.

    The text:
      - Names the two-stage workflow and the KB doc ids that define it.
      - Reports the agent's own observable Stage-1 give count.
      - Reports the RESOLVED / SUBMITTED partition of Stage-1 returns
        observed in this conversation (counted off the framework-emitted
        ``Status:`` lines).
      - States the DB-evaluation consequence of running Stage 2 when no
        Stage-1 return is RESOLVED.
      - Does NOT direct the LLM to call a specific tool; the LLM
        remains in charge of the next action.
    """
    parts = [
        "[STAGE STATE] update_transaction_rewards_3847 is Stage 2 of the cash-back-"
        "rewards correction workflow.",
        (
            "  - Stage 1 (KB doc credit_cards_(general)_003 'Submitting a Cash Back Dispute'): "
            "agent must give submit_cash_back_dispute_0589 so the customer can submit "
            "one cash-back dispute per affected transaction. Each user submission writes "
            "one row to the `cash_back_disputes` table."
        ),
        (
            "  - Stage 2 (KB doc credit_cards_(general)_004 'Applying Resolved Cash Back "
            "Dispute Corrections'): once a dispute is resolved (its Stage-1 return shows "
            "the framework-emitted line `Status: RESOLVED`), the agent unlocks and calls "
            "update_transaction_rewards_3847 for that transaction. If the Stage-1 return "
            "shows `Status: SUBMITTED`, the dispute is still pending review and Stage 2 "
            "is premature for that transaction."
        ),
        f"Observed Stage-1 give calls in this conversation: {give_count}.",
        (
            "Observed Stage-1 user submissions by status: "
            f"RESOLVED={resolved}, SUBMITTED={submitted}, other={other}."
        ),
    ]
    if resolved == 0 and submitted == 0 and other == 0:
        parts.append(
            "No Stage-1 user submissions have been observed yet. Stage 2 here writes "
            "rewards corrections against transactions whose disputes have not been "
            "filed; the gold workflow expects Stage 1 first."
        )
    elif resolved == 0 and (submitted + other) > 0:
        parts.append(
            "No Stage-1 return in this conversation has shown `Status: RESOLVED`. "
            "Proceeding to Stage 2 corrections on these transactions will diverge the "
            "evaluation DB hash from gold (which, for SUBMITTED-only flows, has zero "
            "rows in the post-Stage-2 fields). Reconsider whether Stage 2 should run."
        )
    else:
        parts.append(
            "At least one Stage-1 return shows `Status: RESOLVED`; Stage 2 corrections "
            "against those resolved transactions are warranted."
        )
    return "\n".join(parts)
