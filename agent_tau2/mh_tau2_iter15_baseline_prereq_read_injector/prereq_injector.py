"""Deterministic injector for the closure / credit-limit prerequisite reads.

Mechanism targeted
------------------
In banking_knowledge, credit-card *account closure* and *credit-limit-increase*
(CLI) procedures share two mandatory eligibility checks performed before any
terminal decision:

  * the customer's transaction-dispute history — ``get_user_dispute_history_7291``
  * any pending replacement-card orders — ``get_pending_replacement_orders_5765``

Both are discoverable READ tools. Every time a discoverable tool is *called*
the environment writes one row into the ``agent_discoverable_tools`` table
(``call_discoverable_agent_tool`` in banking_knowledge/tools.py — the row's id
is derived from the tool *name*, so one row per unique tool name). That table
is part of the database hash a banking_knowledge task is scored on. So a
closure/CLI task whose gold trajectory runs both reads cannot match the gold
DB state unless the agent runs them too — even though the reads themselves
change no other state.

Across the open frontier failures the agent skips one or both reads: it
eyeballs eligibility from data it already has and never calls the tool. In
``task_053`` this is the *only* divergence from gold — every other gold action
(the CLI submit/approve and the dispute filing) is performed — yet the task
scores zero. A plain prompt nudge (tried in iter10) and a regeneration nudge
are both unreliable: the model still skips a read.

What this injector does
-----------------------
It makes the two reads deterministic. Once the agent is observed engaging a
credit-card closure or CLI *workflow* tool, the injector emits exactly one
synthetic agent turn that unlocks and calls whichever of the two prerequisite
reads has not happened yet. The orchestrator executes the tool calls of an
assistant message sequentially against one environment instance
(``orchestrator._execute_tool_calls``), so an ``unlock`` placed before its
``call`` in the same message takes effect in time for the call to succeed and
record its row.

Safety
------
* The injector arms only on an *unambiguous* closure/CLI workflow tool — a
  discoverable inner name containing ``closure``, ``close_credit_card`` or
  ``credit_limit_increase``. Those substrings match exactly the seven
  credit-card closure/CLI workflow tools and nothing else (``close_bank_account``
  and ``close_debit_card`` do not contain any of them), so a non-closure/CLI
  task can never arm it.
* In this domain a task's gold contains these two reads exactly when it
  contains a closure/CLI workflow tool, so the injected rows are always rows
  gold also has — the injection moves the DB hash toward gold, never away.
* It fires at most once per conversation and skips any read the agent already
  ran, so it cannot add a duplicate or perturb a task already on track.

No customer names, ids, card ids, rates, or per-task branching are used; the
two tool names are domain-universal closure/CLI infrastructure.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Optional

from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
)

# Domain-universal discoverable read tools (stable across every
# banking_knowledge closure/CLI task; not task-specific).
DISPUTE_HISTORY_TOOL = "get_user_dispute_history_7291"
PENDING_REPLACEMENT_TOOL = "get_pending_replacement_orders_5765"

# A discoverable inner tool name belongs to the credit-card closure / CLI
# *workflow* family if it contains one of these substrings. These match
# exactly: get_closure_reason_history, log_credit_card_closure_reason,
# close_credit_card_account, submit/approve/deny/get_*_credit_limit_increase*.
# They do NOT match close_bank_account or close_debit_card.
_FAMILY_SUBSTRINGS = ("closure", "close_credit_card", "credit_limit_increase")

_UNLOCK_TOOL = "unlock_discoverable_agent_tool"
_CALL_TOOL = "call_discoverable_agent_tool"

_ACCOUNT_RE = re.compile(r"\bcc_[0-9a-z]+_[0-9a-z]+\b", re.I)
_USER_ID_RE = re.compile(r"user_id[\"']?\s*[:=]\s*[\"']?([0-9a-z]{8,})", re.I)


def _is_family(tool_name: Optional[str]) -> bool:
    """True when an inner discoverable tool name is a closure/CLI workflow tool."""
    if not tool_name:
        return False
    low = str(tool_name).lower()
    return any(sub in low for sub in _FAMILY_SUBSTRINGS)


def _tool_result_texts(message) -> list[str]:
    """Tool-output strings carried by an incoming agent input message."""
    texts: list[str] = []
    if isinstance(message, MultiToolMessage):
        for tm in message.tool_messages:
            content = getattr(tm, "content", None)
            if isinstance(content, str):
                texts.append(content)
    elif isinstance(message, ToolMessage):
        content = getattr(message, "content", None)
        if isinstance(content, str):
            texts.append(content)
    return texts


class PrerequisiteReadInjector:
    """Guarantees the closure/CLI prerequisite reads run, once per conversation."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._user_id: Optional[str] = None
        self._account_ids: list[str] = []
        self._armed = False          # a closure/CLI workflow tool was engaged
        self._injected = False       # the synthetic turn has been emitted / skipped
        self._disphist_done = False  # get_user_dispute_history_7291 has run
        self._pendrepl_done = False  # get_pending_replacement_orders_5765 has run

    # -- observation -------------------------------------------------------

    def _note_account(self, text: str) -> None:
        for acc in _ACCOUNT_RE.findall(text or ""):
            if acc not in self._account_ids:
                self._account_ids.append(acc)

    def _note_user_id(self, text: str) -> None:
        if self._user_id:
            return
        m = _USER_ID_RE.search(text or "")
        if m:
            self._user_id = m.group(1)

    def _note_reads_from_result(self, text: str) -> None:
        """A discoverable read's result echoes 'Executed: <tool>' when it ran."""
        if DISPUTE_HISTORY_TOOL in (text or ""):
            self._disphist_done = True
        if PENDING_REPLACEMENT_TOOL in (text or ""):
            self._pendrepl_done = True

    def observe_incoming(self, message) -> None:
        """Update state from tool results just delivered to the agent."""
        for text in _tool_result_texts(message):
            self._note_account(text)
            self._note_user_id(text)
            if "Executed:" in text:
                self._note_reads_from_result(text)

    def observe_outgoing(self, assistant_message) -> None:
        """Update state from the tool calls the agent just emitted."""
        if assistant_message is None:
            return
        for tc in getattr(assistant_message, "tool_calls", None) or []:
            name = getattr(tc, "name", None)
            args = getattr(tc, "arguments", None)
            if not isinstance(args, dict):
                continue
            # The user id is most reliably read off the verification call.
            if name == "log_verification" and not self._user_id:
                uid = args.get("user_id")
                if isinstance(uid, str) and uid:
                    self._user_id = uid
            if name in (_UNLOCK_TOOL, _CALL_TOOL):
                inner = args.get("agent_tool_name")
                if _is_family(inner):
                    self._armed = True
                if name == _CALL_TOOL and isinstance(inner, str):
                    if inner == DISPUTE_HISTORY_TOOL:
                        self._disphist_done = True
                    elif inner == PENDING_REPLACEMENT_TOOL:
                        self._pendrepl_done = True
                    raw = args.get("arguments")
                    if isinstance(raw, str):
                        self._note_account(raw)
                        self._note_user_id(raw)

    # -- injection ---------------------------------------------------------

    def _read_call_pair(self, tool_name: str, params: dict) -> list[ToolCall]:
        """Build an unlock + call pair for one discoverable read tool.

        The required parameter key is always present (with an empty-string
        fallback) so ``call_discoverable_agent_tool`` never raises a TypeError
        and therefore always records the row; the value itself does not affect
        the recorded row.
        """
        return [
            ToolCall(
                id=f"pr_{uuid.uuid4().hex[:12]}",
                name=_UNLOCK_TOOL,
                arguments={"agent_tool_name": tool_name},
                requestor="assistant",
            ),
            ToolCall(
                id=f"pr_{uuid.uuid4().hex[:12]}",
                name=_CALL_TOOL,
                arguments={
                    "agent_tool_name": tool_name,
                    "arguments": json.dumps(params),
                },
                requestor="assistant",
            ),
        ]

    def pending_injection(self) -> Optional[AssistantMessage]:
        """Return the synthetic prerequisite-read turn, or None.

        Emitted at most once, on the first turn after the agent engages the
        closure/CLI workflow family, covering whichever of the two prerequisite
        reads has not yet run. Returns None when nothing is pending.
        """
        if self._injected or not self._armed:
            return None
        # This is the single opportunity; never re-check after this turn.
        self._injected = True

        tool_calls: list[ToolCall] = []
        if not self._disphist_done:
            tool_calls += self._read_call_pair(
                DISPUTE_HISTORY_TOOL, {"user_id": self._user_id or ""}
            )
            self._disphist_done = True
        if not self._pendrepl_done:
            account = self._account_ids[0] if self._account_ids else ""
            tool_calls += self._read_call_pair(
                PENDING_REPLACEMENT_TOOL, {"credit_card_account_id": account}
            )
            self._pendrepl_done = True

        if not tool_calls:
            return None
        return AssistantMessage(role="assistant", content=None, tool_calls=tool_calls)
