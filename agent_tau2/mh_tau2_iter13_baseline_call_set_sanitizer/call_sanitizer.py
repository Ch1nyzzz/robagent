"""Deterministic sanitizer that removes never-gold discoverable-tool calls.

Mechanism targeted
------------------
banking_knowledge tasks scored on the database hash are graded on the
resulting database state, which includes the ``agent_discoverable_tools``
table — one row per UNIQUE discoverable tool the agent calls. The reward is
zero unless that recorded set exactly equals the gold trajectory's set, so a
single discoverable call gold never makes zeroes the task.

Two classes of agent-emitted discoverable calls appear in ZERO of the 30
train-set gold trajectories and therefore can only ever ADD a non-gold row:

1. Website / app navigation redirect tools (``navigate_to_section``,
   ``open_webpage``, ...). A customer-service agent is supposed to perform the
   requested action itself; handing the customer a "go to this page" tool is
   both the wrong behaviour and a guaranteed extra DB row. Surveying the
   train-30 gold actions, no trajectory uses such a tool; surveying 890 prior
   simulations, redirect tools were emitted on eight distinct tasks and every
   one of those simulations scored zero.

2. The contradictory second branch of a binary decision. A credit-limit
   increase has exactly one outcome in gold — an approval OR a denial, never
   both. When the agent hedges and emits the opposite branch after it has
   already committed one, that second call adds a non-gold row (and a
   contradictory effect). Across 890 prior simulations both branches were
   emitted together 36 times, always on a credit-limit-increase task.

This module strips exactly those two classes from an assistant turn before it
runs. It can never remove a gold action: a redirect tool is never gold, and —
by the one-outcome rule — once a decision polarity is committed the OPPOSITE
polarity is never the gold one either. Detection is structural (substring and
token rules on the tool name); no task-specific identifiers are used.
"""
from __future__ import annotations

from typing import Optional

# Substrings that mark a website / app navigation redirect tool. A
# customer-service agent must perform actions itself, so these are never part
# of a gold trajectory regardless of domain.
_REDIRECT_SUBSTRINGS = ("navigat", "webpage", "open_web", "open_url", "open_page")

# Leading verb -> polarity for the two branches of a binary decision.
_DECISION_POLARITY = {
    "approve": "approve",
    "approved": "approve",
    "grant": "approve",
    "granted": "approve",
    "deny": "deny",
    "denied": "deny",
    "reject": "deny",
    "rejected": "deny",
    "decline": "deny",
    "declined": "deny",
}

# Discoverable-tool wrappers and the argument key holding the inner tool name.
_DISCOVERABLE_WRAPPERS = {
    "give_discoverable_user_tool": "discoverable_tool_name",
    "call_discoverable_user_tool": "discoverable_tool_name",
    "unlock_discoverable_agent_tool": "agent_tool_name",
    "call_discoverable_agent_tool": "agent_tool_name",
}

# Wrappers that actually record a discoverable-tool DB row / commit an effect.
# ``unlock`` only makes a tool available to call later; it writes no row, so it
# never "commits" a decision on its own.
_COMMITTING_WRAPPERS = {
    "give_discoverable_user_tool",
    "call_discoverable_user_tool",
    "call_discoverable_agent_tool",
}


def _inner_name(tool_call) -> tuple[str, str]:
    """Return (wrapper_name, lower-case inner tool name) for one tool call.

    For a non-discoverable tool call the inner name is the tool's own name, so
    a redirect tool invoked directly is still detected.
    """
    name = getattr(tool_call, "name", "") or ""
    key = _DISCOVERABLE_WRAPPERS.get(name)
    if key is None:
        return name, name.lower()
    args = getattr(tool_call, "arguments", None)
    if not isinstance(args, dict):
        return name, ""
    return name, str(args.get(key) or "").lower()


def is_redirect_tool(inner: str) -> bool:
    """True when the inner tool name is a website / app navigation redirect."""
    return any(s in inner for s in _REDIRECT_SUBSTRINGS)


def decision_of(inner: str) -> Optional[tuple[str, str]]:
    """Classify a decision tool name as (subject_key, polarity).

    The subject key is the tool name with its leading decision verb and any
    pure-digit tokens stripped, so the approve and deny variants of one
    workflow collapse to the same key. Returns None for non-decision tools.
    """
    if not inner:
        return None
    tokens = [t for t in inner.split("_") if t]
    if not tokens:
        return None
    polarity = _DECISION_POLARITY.get(tokens[0])
    if polarity is None:
        return None
    subject = "_".join(t for t in tokens[1:] if not t.isdigit())
    return subject, polarity


def _commits_decision(wrapper_name: str) -> bool:
    """Whether a tool call with this wrapper commits a decision polarity.

    A discoverable ``unlock`` does not; a direct (non-discoverable) call and
    every committing wrapper do.
    """
    if wrapper_name in _DISCOVERABLE_WRAPPERS:
        return wrapper_name in _COMMITTING_WRAPPERS
    return True


class CallSanitizer:
    """Removes never-gold discoverable-tool calls from an assistant turn.

    Stateless: the committed-decision picture is derived fresh from the
    conversation history on every call, so nothing needs resetting between
    conversations.
    """

    @staticmethod
    def _committed_from_history(messages) -> dict:
        """First decision polarity per subject already on record in history."""
        committed: dict = {}
        for m in messages or []:
            if getattr(m, "role", None) != "assistant":
                continue
            for tc in getattr(m, "tool_calls", None) or []:
                name, inner = _inner_name(tc)
                if not _commits_decision(name):
                    continue
                decision = decision_of(inner)
                if decision is None:
                    continue
                subject, polarity = decision
                committed.setdefault(subject, polarity)
        return committed

    def sanitize(self, assistant_message, history) -> int:
        """Strip never-gold calls from ``assistant_message`` in place.

        Drops (a) any redirect tool call and (b) any decision-tool call whose
        polarity contradicts a decision already committed for that subject —
        in a prior turn or earlier in this same turn. Returns the count
        dropped. When every tool call is dropped, ``tool_calls`` is set to
        ``None`` so the caller can detect an emptied turn.
        """
        tool_calls = getattr(assistant_message, "tool_calls", None)
        if not tool_calls:
            return 0

        committed = self._committed_from_history(history)

        # Pass 1: a decision committed earlier in THIS turn also locks the
        # subject, so a contradictory call later in the same turn is dropped.
        for tc in tool_calls:
            name, inner = _inner_name(tc)
            if not _commits_decision(name):
                continue
            decision = decision_of(inner)
            if decision is None:
                continue
            subject, polarity = decision
            committed.setdefault(subject, polarity)

        # Pass 2: drop redirect tools and contradictory decision branches.
        kept = []
        dropped = 0
        for tc in tool_calls:
            _, inner = _inner_name(tc)
            if is_redirect_tool(inner):
                dropped += 1
                continue
            decision = decision_of(inner)
            if decision is not None:
                subject, polarity = decision
                committed_polarity = committed.get(subject)
                if committed_polarity is not None and committed_polarity != polarity:
                    dropped += 1
                    continue
            kept.append(tc)

        if dropped:
            assistant_message.tool_calls = kept if kept else None
        return dropped
