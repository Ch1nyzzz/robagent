"""Deterministic argument minimizer for tau2 discoverable tools.

A banking "discoverable" tool is announced to the agent in a tool result that
begins with ``Tool unlocked: <name>`` and contains a ``Parameters:`` block
listing each parameter's name, type, and whether it is ``(required)`` or
``(optional)``. A parameter constrained to a fixed set of values additionally
carries a ``Must be one of: ...`` clause in its description.

Observed failure mode
---------------------
The agent loses DB-hash reward when it fabricates an OPTIONAL, FREE-TEXT
string argument that the gold trajectory never sends. The canonical case is a
free-form ``reason`` on the bank-account-closure tool: the agent writes a
narrative like ``"Customer requested closure - consolidating accounts"`` while
gold calls the tool with only the required ``account_id``. The bank's database
stores the argument verbatim, so an invented note changes the stored row and
the database hash no longer equals gold.

This module parses the announced schemas and, for any discoverable-tool call,
strips:

* keys that are not declared parameters of that tool at all, and
* optional, free-text string parameters (declared ``(optional)``, typed
  ``string``, with no ``Must be one of`` enum constraint).

Required parameters and optional parameters that are enum-constrained or typed
as a number / integer / boolean are always kept, so a legitimate argument can
never be removed. The transform only ever moves a call CLOSER to the minimal
gold argument set; a gold tool call cannot contain an undeclared key, and the
gold trajectories never populate a free-text optional note.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

# A single "Parameters:" line, e.g.
#   "  - reason: string (optional) - The reason for closing the account"
_PARAM_RE = re.compile(
    r"^\s*-\s*([A-Za-z_][\w]*)\s*:\s*([A-Za-z]+)\s*\((required|optional)\)\s*(.*)$"
)
# Header line, e.g. "Tool unlocked: close_bank_account_7392".
_UNLOCK_RE = re.compile(r"^Tool unlocked:\s*(\S+)\s*$")


class ParamSpec:
    """The declared shape of a single discoverable-tool parameter."""

    __slots__ = ("type", "required", "is_enum")

    def __init__(self, type_: str, required: bool, is_enum: bool) -> None:
        self.type = type_
        self.required = required
        self.is_enum = is_enum

    def is_droppable(self) -> bool:
        """True when this parameter is an optional, free-text string.

        Those are the arguments the agent fabricates as narrative; gold tool
        calls never populate them. Anything required, enum-constrained, or
        non-string is kept untouched.
        """
        return (not self.required) and self.type == "string" and not self.is_enum


class DiscoverableToolSchemas:
    """Collects discoverable-tool parameter schemas seen in tool results."""

    def __init__(self) -> None:
        # tool name -> {param name -> ParamSpec}
        self._schemas: dict[str, dict[str, ParamSpec]] = {}

    def reset(self) -> None:
        self._schemas.clear()

    def observe_tool_result(self, content: Optional[str]) -> None:
        """Parse a tool result; register any 'Tool unlocked:' schema it carries."""
        if not isinstance(content, str) or "Tool unlocked:" not in content:
            return
        lines = content.splitlines()
        name: Optional[str] = None
        for line in lines:
            m = _UNLOCK_RE.match(line.strip())
            if m:
                name = m.group(1)
                break
        if name is None:
            return
        params: dict[str, ParamSpec] = {}
        for line in lines:
            pm = _PARAM_RE.match(line)
            if not pm:
                continue
            pname, ptype, req, desc = pm.groups()
            params[pname] = ParamSpec(
                type_=ptype.lower(),
                required=(req.lower() == "required"),
                is_enum=("must be one of" in (desc or "").lower()),
            )
        # Only register when at least one parameter parsed cleanly; an empty
        # parse means the format was unexpected, so we fail safe and leave
        # calls to that tool untouched.
        if params:
            self._schemas[name] = params

    def sanitize(self, tool_name: str, args: dict) -> dict:
        """Return a copy of a discoverable-tool call's arguments with fabricated
        / undeclared keys removed.

        ``args`` is the outer argument dict of a ``call_discoverable_agent_tool``
        call: ``{"agent_tool_name": <str>, "arguments": <json str | dict>}``.
        The outer dict is never mutated; a new dict is returned only if the
        inner payload actually changed.
        """
        inner_name = args.get("agent_tool_name")
        if not isinstance(inner_name, str):
            return args
        schema = self._schemas.get(inner_name)
        if not schema:
            return args  # schema unknown -> fail safe, change nothing

        raw = args.get("arguments")
        as_string = isinstance(raw, str)
        if as_string:
            try:
                inner = json.loads(raw)
            except (ValueError, TypeError):
                return args  # not parseable -> leave the call alone
        elif isinstance(raw, dict):
            inner = raw
        else:
            return args
        if not isinstance(inner, dict):
            return args

        cleaned: dict[str, Any] = {}
        dropped: list[str] = []
        for key, value in inner.items():
            spec = schema.get(key)
            if spec is None:
                dropped.append(key)  # not a declared parameter
                continue
            if spec.is_droppable():
                dropped.append(key)  # invented optional free-text argument
                continue
            cleaned[key] = value

        if not dropped:
            return args  # nothing to do

        new_args = dict(args)
        new_args["arguments"] = json.dumps(cleaned) if as_string else cleaned
        return new_args
