"""Tool registry for the GAIA baseline FC agent.

`TOOL_SPECS` is the OpenAI function-call shape list to pass into `agent.llm.chat(tools=...)`.
`dispatch_tool(name, args)` runs the tool and returns a string result for the tool_message.

Adding a new tool: write `agent/tools/<name>.py` exposing `SPEC: dict` (OpenAI
function-call shape) and `run(args: dict) -> str`; then add it to `_TOOLS` below.
"""
from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

# Match the pattern in agent.llm / bench.gaia.loader: .env is loaded eagerly
# so smoke tests / one-off scripts see the configured TAVILY_API_KEY etc.
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

from . import file_read, python_exec, url_fetch, web_search


_TOOLS = {
    "file_read":   file_read,
    "url_fetch":   url_fetch,
    "web_search":  web_search,
    "python_exec": python_exec,
}


TOOL_SPECS: list[dict[str, Any]] = [t.SPEC for t in _TOOLS.values()]

TOOL_NAMES: tuple[str, ...] = tuple(_TOOLS.keys())


def dispatch_tool(name: str, args: dict | None) -> str:
    """Run the named tool with args dict; return string output (never raises)."""
    if name not in _TOOLS:
        return f"ERROR: unknown tool {name!r}. Available: {list(_TOOLS)}"
    try:
        return _TOOLS[name].run(args or {})
    except Exception as e:
        return f"ERROR running {name}: {type(e).__name__}: {e}"
