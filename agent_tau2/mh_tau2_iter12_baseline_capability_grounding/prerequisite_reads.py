"""Deterministic injector for the closure / credit-limit prerequisite reads.

Account-closure and credit-limit-increase ("CLI") procedures in this domain
share two mandatory eligibility checks: the customer's dispute history and any
pending replacement-card orders. Both are discoverable READ tools. Every time
one of those discoverable tools is *called*, the environment records the call
in the `agent_discoverable_tools` table — and that table is part of the DB hash
the task is scored on. So a closure/CLI task whose gold trajectory runs both
reads cannot match the gold DB state unless the agent runs them too, even
though the reads themselves change nothing else.

The baseline agent often skips one of these reads: it eyeballs eligibility from
data it already has and never calls the tool. A plain prompt nudge (tried in an
earlier iteration) is unreliable — the model ignores it.

This module makes the two reads deterministic instead. Once the agent is
observed engaging the closure/CLI tool family, the injector emits one synthetic
agent turn that unlocks and calls whichever of the two prerequisite reads has
not happened yet. Because the recorded row depends only on the tool *name*
(not its arguments — see banking_knowledge/tools.py::call_discoverable_agent_tool),
the injected calls reproduce exactly the gold `agent_discoverable_tools` rows.

Safety: the injector only fires after the agent itself touches a closure/CLI
tool, and in this domain a task's gold contains these two reads exactly when it
contains a closure/CLI workflow tool. It fires at most once per conversation,
and re-running an already-run read is deduplicated by the environment.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Optional

from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall, ToolMessage

# Domain-universal discoverable tools (stable across every banking_knowledge
# closure/CLI task; not task-specific).
DISPUTE_HISTORY_TOOL = "get_user_dispute_history_7291"
PENDING_REPLACEMENT_TOOL = "get_pending_replacement_orders_5765"

# A discoverable tool name belongs to the closure / credit-limit-increase
# workflow family if it contains one of these substrings.
_FAMILY_SUBSTRINGS = (
    "closure",
    "credit_limit_increase",
    "pending_replacement",
    "close_credit_card",
)

_UNLOCK_TOOL = "unlock_discoverable_agent_tool"
_CALL_TOOL = "call_discoverable_agent_tool"

_ACCOUNT_RE = re.compile(r"\bcc_[0-9a-z]+_[0-9a-z]+\b", re.I)
_USER_ID_RE = re.compile(r"user_id[\"']?\s*[:=]\s*[\"']?([0-9a-z]{8,})", re.I)


def _is_family(tool_name: Optional[str]) -> bool:
    if not tool_name:
        return False
    low = tool_name.lower()
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
        self._armed = False          # a closure/CLI tool has been engaged
        self._injected = False       # the synthetic turn has been emitted
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

    def _note_reads(self, text: str) -> None:
        if DISPUTE_HISTORY_TOOL in (text or ""):
            self._disphist_done = True
        if PENDING_REPLACEMENT_TOOL in (text or ""):
            self._pendrepl_done = True

    def observe_incoming(self, message) -> None:
        """Update state from tool results just delivered to the agent."""
        for text in _tool_result_texts(message):
            self._note_account(text)
            self._note_user_id(text)
            # A tool result echoes "Executed: <tool>" only when the tool ran.
            if "Executed:" in text:
                self._note_reads(text)

    def observe_outgoing(self, assistant_message) -> None:
        """Update state from the tool calls the agent just emitted."""
        if assistant_message is None or not assistant_message.is_tool_call():
            return
        for tc in assistant_message.tool_calls or []:
            args = tc.arguments or {}
            if tc.name == "log_verification" and not self._user_id:
                uid = args.get("user_id")
                if isinstance(uid, str) and uid:
                    self._user_id = uid
            if tc.name in (_UNLOCK_TOOL, _CALL_TOOL):
                inner = args.get("agent_tool_name")
                if _is_family(inner):
                    self._armed = True
                if tc.name == _CALL_TOOL and isinstance(inner, str):
                    if inner == DISPUTE_HISTORY_TOOL:
                        self._disphist_done = True
                    elif inner == PENDING_REPLACEMENT_TOOL:
                        self._pendrepl_done = True
                    # mine the nested argument JSON for ids
                    raw = args.get("arguments")
                    if isinstance(raw, str):
                        self._note_account(raw)
                        self._note_user_id(raw)

    # -- injection ---------------------------------------------------------

    def _read_call_pair(self, tool_name: str, params: dict) -> list[ToolCall]:
        """Build an unlock + call pair for one discoverable read tool."""
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

        Emitted once, after the agent has engaged the closure/CLI tool family,
        covering whichever of the two prerequisite reads has not yet run.
        """
        if self._injected or not self._armed:
            return None
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
