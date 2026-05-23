"""Deterministic write-action classification for the write-discipline agent.

The reward for most banking_knowledge tasks is a DB-hash comparison: the set
of state-modifying tool calls the agent makes must exactly equal the gold set.
Any *extra* write the agent performs on its own initiative (a courtesy
statement credit, an additional account closure, an extra dispute) changes the
final DB hash and zeroes the reward even when every requested action was done.

This module holds only deterministic logic: it decides, from a tool call's
name alone, whether the call is a state-modifying "write". No task-specific
identifiers appear here — classification is purely verb-based and therefore
applies to any tool in any domain.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

# Leading verbs that create / update / delete data.
WRITE_VERBS = {
    "open", "close", "transfer", "pay", "submit", "deny", "approve",
    "order", "file", "deposit", "withdraw", "update", "apply", "log",
    "create", "delete", "cancel", "issue", "reissue", "send", "set",
    "change", "add", "remove", "modify", "charge", "refund", "process",
    "make", "place", "schedule", "enroll", "activate", "deactivate",
    "reset", "assign", "grant", "revoke", "credit", "debit", "waive",
    "book", "register", "renew", "upgrade", "downgrade",
}

# Leading verbs that only read state.
READ_VERBS = {
    "get", "list", "search", "view", "retrieve", "check", "find",
    "lookup", "read", "fetch", "show", "describe", "count", "is",
    "has", "calculate", "compute", "validate", "verify", "lookup",
}

# Tools that modify state but are mandatory / never "extra"; performing them
# should not, on its own, trigger a discipline review.
NON_TRIGGER = {
    "log_verification",
    "transfer_to_human_agents",
    "done",
    "think",
    "send_message_to_user",
}

# Discoverable-tool meta-call names.
CALL_AGENT_TOOL = "call_discoverable_agent_tool"
CALL_USER_TOOL = "call_discoverable_user_tool"
GIVE_USER_TOOL = "give_discoverable_user_tool"
UNLOCK_AGENT_TOOL = "unlock_discoverable_agent_tool"

_ID_SUFFIX = re.compile(r"_\d+$")


def _strip_id(name: str) -> str:
    """Drop a trailing numeric id (`apply_statement_credit_8472` -> base)."""
    return _ID_SUFFIX.sub("", name or "")


def _first_verb(name: str) -> str:
    """The leading underscore-token of a tool name, lower-cased."""
    base = _strip_id(name).lower().strip()
    if not base:
        return ""
    return base.split("_")[0]


def classify(name: Optional[str]) -> str:
    """Return 'write', 'read', or 'unknown' for a tool name."""
    if not name:
        return "unknown"
    verb = _first_verb(name)
    if verb in WRITE_VERBS:
        return "write"
    if verb in READ_VERBS:
        return "read"
    return "unknown"


def _arguments(tool_call: Any) -> dict:
    """The outer arguments dict of a tool call, however the provider encoded it."""
    args = getattr(tool_call, "arguments", None)
    if args is None and isinstance(tool_call, dict):
        args = tool_call.get("arguments")
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _name(tool_call: Any) -> str:
    n = getattr(tool_call, "name", None)
    if n is None and isinstance(tool_call, dict):
        n = tool_call.get("name")
    return n or ""


def _inner_args(outer: dict) -> dict:
    """The nested `arguments` payload of a discoverable meta-call."""
    inner = outer.get("arguments")
    if isinstance(inner, dict):
        return inner
    if isinstance(inner, str):
        try:
            parsed = json.loads(inner)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def discoverable_target(tool_call: Any) -> Optional[str]:
    """For a discoverable meta-call, the underlying tool's name."""
    outer = _arguments(tool_call)
    return outer.get("agent_tool_name") or outer.get("discoverable_tool_name")


def is_trigger_write(tool_call: Any) -> bool:
    """True if this agent-emitted call is a state-modifying write worth reviewing.

    Unlocking a discoverable tool, giving a tool to the user, and the mandatory
    NON_TRIGGER tools never trigger a review. A `call_discoverable_agent_tool`
    is judged by the verb of the underlying tool it invokes.
    """
    name = _name(tool_call)
    if name in (UNLOCK_AGENT_TOOL, GIVE_USER_TOOL, CALL_USER_TOOL):
        return False
    if name in NON_TRIGGER:
        return False
    if name == CALL_AGENT_TOOL:
        target = discoverable_target(tool_call)
        if not target or target in NON_TRIGGER:
            return False
        return classify(target) == "write"
    return classify(name) == "write"


def describe(tool_call: Any) -> str:
    """A compact, human-readable rendering of a write call for the reviewer."""
    name = _name(tool_call)
    outer = _arguments(tool_call)
    if name == CALL_AGENT_TOOL:
        target = discoverable_target(tool_call) or "?"
        inner = _inner_args(outer)
        return f"{target}({_fmt_args(inner)})"
    return f"{name}({_fmt_args(outer)})"


def _fmt_args(args: dict) -> str:
    if not args:
        return ""
    parts = []
    for k, v in args.items():
        s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
        if len(s) > 80:
            s = s[:77] + "..."
        parts.append(f"{k}={s}")
    return ", ".join(parts)


def extract_writes(message: Any) -> list[tuple[Any, str]]:
    """All triggering write tool calls in an assistant message, with descriptions."""
    out: list[tuple[Any, str]] = []
    for tc in getattr(message, "tool_calls", None) or []:
        if is_trigger_write(tc):
            out.append((tc, describe(tc)))
    return out


def render_transcript(messages: list[Any]) -> str:
    """Render a conversation as plain text for the verification call.

    User turns are kept in full (they carry the customer's actual requests);
    tool results and long assistant prose are truncated to keep the prompt
    bounded.
    """
    lines: list[str] = []
    for m in messages:
        role = getattr(m, "role", None) or "?"
        content = getattr(m, "content", None) or ""
        tool_calls = getattr(m, "tool_calls", None) or []
        if role == "tool":
            text = content if len(content) <= 200 else content[:200] + "..."
            lines.append(f"[tool result] {text}")
            continue
        if tool_calls:
            for tc in tool_calls:
                name = _name(tc)
                if name == CALL_AGENT_TOOL:
                    tgt = discoverable_target(tc) or "?"
                    lines.append(f"[{role} action] {tgt}")
                else:
                    lines.append(f"[{role} action] {name}")
            if content.strip():
                snippet = content if len(content) <= 350 else content[:350] + "..."
                lines.append(f"[{role}] {snippet}")
            continue
        if role == "user":
            lines.append(f"[user] {content}")
        else:
            snippet = content if len(content) <= 350 else content[:350] + "..."
            lines.append(f"[{role}] {snippet}")
    return "\n".join(lines)
