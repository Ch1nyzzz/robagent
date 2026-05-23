"""Deterministic guard for the transfer-reason-lookup candidate.

The `transfer_to_human_agents` tool takes a `reason` argument that must be an
exact code from the bank's tiered transfer-reason-code policy. That policy
lives in the knowledge base, and the tool's own description tells the agent to
look it up before calling. When the agent transfers without first retrieving
that policy it just guesses a plausible-sounding code, which rarely equals the
gold code, so the action check fails.

This guard tracks whether the transfer reason-code policy has been retrieved in
the conversation, and flags a turn that calls `transfer_to_human_agents` before
that lookup has happened. The agent module reacts by regenerating the turn with
a note steering it to search the knowledge base first. The guard never blocks a
transfer outright (after a capped number of nudges it lets the turn through),
so a legitimate transfer can never be permanently suppressed.
"""
from __future__ import annotations

TRANSFER_TOOL = "transfer_to_human_agents"

# Total number of "look up the policy first" nudges allowed per conversation.
# A conversation almost never makes more than one or two transfer attempts;
# this cap only exists so a stubborn model can never cause an infinite loop.
MAX_NUDGES = 4

TRANSFER_LOOKUP_NOTE = (
    "You are about to transfer the customer to a human agent, but you have not "
    "yet retrieved the bank's official transfer reason-code policy in this "
    "conversation. The `reason` argument of transfer_to_human_agents must be an "
    "exact code from that tiered policy, not a phrase you choose yourself. Do "
    "not transfer yet. First make a KB_search call to retrieve the transfer "
    "reason-code policy (for example with a query like "
    "\"transfer to human agents reason codes\"). Once you have that policy, "
    "pick the most specific reason code from the highest-priority tier that "
    "genuinely applies to the customer's situation, then call "
    "transfer_to_human_agents with that exact code."
)


def looks_like_transfer_reason_doc(text) -> bool:
    """True when a tool-result blob contains the transfer reason-code policy.

    The detector is intentionally conservative: it keys off the literal tool
    name plus the words used in the policy's own heading. A miss only causes a
    harmless redundant lookup; a false hit only degrades to baseline behaviour.
    """
    if not isinstance(text, str):
        return False
    t = text.lower()
    if TRANSFER_TOOL in t and "reason" in t:
        return True
    if "transfer reason code" in t:
        return True
    if "human agent transfer reason" in t:
        return True
    return False


class TransferReasonGuard:
    """Tracks retrieval of the transfer reason-code policy."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._reason_doc_seen = False
        self._nudges_used = 0

    def observe_tool_result(self, content) -> None:
        """Note any tool output that contains the transfer reason-code policy."""
        if looks_like_transfer_reason_doc(content):
            self._reason_doc_seen = True

    @staticmethod
    def _has_transfer_call(assistant_message) -> bool:
        calls = getattr(assistant_message, "tool_calls", None) or []
        return any(getattr(c, "name", None) == TRANSFER_TOOL for c in calls)

    def needs_lookup(self, assistant_message) -> bool:
        """True when this turn transfers but the policy was never retrieved."""
        if self._reason_doc_seen:
            return False
        if self._nudges_used >= MAX_NUDGES:
            return False
        return self._has_transfer_call(assistant_message)

    def register_nudge(self) -> None:
        self._nudges_used += 1

    def guidance(self) -> str:
        return TRANSFER_LOOKUP_NOTE
