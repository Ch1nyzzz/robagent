"""Deterministic guard against hallucinated incapability.

The agent sometimes declines a customer request it actually holds a tool for —
it says "I'm not able to do that on my behalf" or "you'll need to apply through
your online dashboard" even though a matching tool is available. The required
tool call never happens and the DB-hash check fails.

This guard inspects a message-only turn (a turn that sends text and makes no
tool call). When that text reads as an unfounded capability refusal or an
external-channel redirect AND the customer's most recent message was an
explicit request, the agent's turn is regenerated once with a transient note
steering it to check its tools and execute instead of declining.

The guard never blocks a message permanently: after the nudge budget is spent
the message is sent as-is, so a legitimate, policy-grounded refusal is never
suppressed — at worst one extra generation is spent re-confirming it.
"""
from __future__ import annotations

import re
from typing import Optional

# The agent states it cannot do something.
_NEGATED_CAPABILITY = [
    re.compile(r"\bunable to\b", re.I),
    re.compile(r"\bnot able to\b", re.I),
    re.compile(r"\bi (can'?t|cannot)\b", re.I),
    re.compile(r"\bi (do not|don'?t) have (the |a |an )?(ability|access|means|way|tools?)\b", re.I),
    re.compile(r"\b(isn'?t|is not|not) something i (can|am able to)\b", re.I),
    re.compile(r"\bi'?m not (\w+\s){0,3}(able|allowed|permitted) to\b", re.I),
    re.compile(r"\bon my end\b", re.I),
]

# The agent tells the customer to perform the action through another channel.
_CHANNEL_REDIRECT = [
    re.compile(r"\bthrough (your|the|our)\b[^.!?\n]{0,70}\b(dashboard|online banking|mobile app|web ?site|portal|branch)\b", re.I),
    re.compile(r"\b(visit|log ?in(to)?|sign in|go to|head to)\b[^.!?\n]{0,45}\b(dashboard|online banking|web ?site|portal|branch|app)\b", re.I),
    re.compile(r"\byou'?ll need to\b[^.!?\n]{0,70}\b(yourself|dashboard|online banking|web ?site|branch|app)\b", re.I),
    re.compile(r"\bdo (this|that|it)\s+yourself\b", re.I),
    re.compile(r"\b(yourself|on your own) through\b", re.I),
    re.compile(r"\bon your behalf\b", re.I),
]

# The customer's last message asks the agent to do something.
_REQUEST = [
    re.compile(r"\bcan you\b", re.I),
    re.compile(r"\bcould you\b", re.I),
    re.compile(r"\bwould you\b", re.I),
    re.compile(r"\bwill you\b", re.I),
    re.compile(r"\bcan i\b", re.I),
    re.compile(r"\bcan we\b", re.I),
    re.compile(r"\bplease\b", re.I),
    re.compile(r"\bhelp me\b", re.I),
    re.compile(r"\bgo ahead\b", re.I),
    re.compile(r"\bi'?d (like|love)\b", re.I),
    re.compile(r"\bi would like\b", re.I),
    re.compile(r"\bi want\b", re.I),
    re.compile(r"\blet'?s\b", re.I),
    re.compile(r"\b(apply|application|submit|proceed|sign up|open an?)\b", re.I),
]

# Maximum number of capability nudges per conversation.
MAX_NUDGES = 2


def _any(patterns, text: str) -> bool:
    return any(p.search(text) for p in patterns)


class CapabilityGuard:
    """Flags message-only turns that decline a request the agent can fulfil."""

    def __init__(self) -> None:
        self._nudges = 0

    def reset(self) -> None:
        self._nudges = 0

    @staticmethod
    def _last_user_text(state) -> str:
        for msg in reversed(getattr(state, "messages", []) or []):
            if getattr(msg, "role", None) == "user":
                content = getattr(msg, "content", None)
                if isinstance(content, str):
                    return content
        return ""

    def needs_regeneration(self, assistant_message, state) -> bool:
        """True when the agent's turn is an unfounded refusal worth retrying."""
        if self._nudges >= MAX_NUDGES:
            return False
        # Only message-only turns; a turn that already makes a tool call is fine.
        if assistant_message is None or assistant_message.is_tool_call():
            return False
        content = assistant_message.content or ""
        if not content.strip():
            return False
        if not _any(_REQUEST, self._last_user_text(state)):
            return False
        if not (_any(_NEGATED_CAPABILITY, content) or _any(_CHANNEL_REDIRECT, content)):
            return False
        return True

    def register_nudge(self) -> None:
        self._nudges += 1

    @staticmethod
    def guidance() -> str:
        return (
            "Your draft reply declines the customer's request or redirects them "
            "to another channel (a website, dashboard, app, branch, or another "
            "team). Before sending it, re-check the tools available to you in "
            "this conversation. You are an agent that performs actions through "
            "tools: if a tool can perform what the customer just asked for, call "
            "that tool now instead of declining or redirecting them. If you are "
            "missing a required argument for that tool, ask the customer for the "
            "missing detail rather than refusing. Only send a refusal if no "
            "available tool can do it or the policy explicitly forbids it."
        )
