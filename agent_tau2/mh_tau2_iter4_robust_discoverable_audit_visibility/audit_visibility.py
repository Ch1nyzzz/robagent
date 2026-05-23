"""Deterministic surfacing of tau2 discoverable-tool DB-audit semantics.

Why this exists
---------------
Many DB-hash failures on banking_knowledge train tasks have the same shape:
the agent makes every gold-required tool call, but ALSO makes one or more
extra ``call_discoverable_agent_tool`` invocations (typically read-named tools
like ``get_user_dispute_history_7291`` reached for as "research"). Per tau2
sandbox semantics each unique ``call_discoverable_agent_tool`` invocation
writes one row to the ``agent_discoverable_tools`` table (see
``tau2-bench-src/src/tau2/domains/banking_knowledge/tools.py``: the
``call_discoverable_agent_tool`` body in lines 631-676 ends with
``add_to_db("agent_discoverable_tools", record_id, ...)``); each unique
``give_discoverable_user_tool`` writes one row to ``user_discoverable_tools``.
These rows participate in the evaluation DB hash. An extra row = a mismatch.

The agent has access to the system prompt notice in
``banking_knowledge/prompts/components/additional_instructions.md``:

    "Do not unlock tools that you do not plan on giving to the user and
     actually using: this causes issues in database logging."

but the warning is buried in the global instruction block and the agent
ignores it — especially when contemplating tools whose *name* starts with
``get_`` (which read like reads but are tau2 WRITE tools).

What this module does
---------------------
Provides a small, system-fact-anchored augmenter that appends an explicit
audit notice to the **tool response** of every discoverable-tool entry-point
call. The notice describes what the tau2 sandbox actually did (or, in the
unlock case, what a subsequent call would do):

  * After ``unlock_discoverable_agent_tool`` — remind the agent that any
    subsequent ``call_discoverable_agent_tool`` to the just-unlocked tool
    will write one row, including tools whose name starts with ``get_``.
  * After ``call_discoverable_agent_tool`` — confirm that one row has been
    written under the inner discoverable tool name; flag that additional
    discoverable-tool calls not strictly required for the user's stated
    request will diverge the evaluation DB hash from gold.
  * After ``give_discoverable_user_tool`` — confirm a row has been written
    to ``user_discoverable_tools`` and that the agent should not give the
    same tool again redundantly.

The augmenter is keyed only to the framework-level entry-point tool *name*
(``unlock_discoverable_agent_tool`` / ``call_discoverable_agent_tool`` /
``give_discoverable_user_tool``) — a tau2-system schema fact independent of
any specific KB or task. The content of the notice describes tau2 sandbox
semantics. No policy interpretation is encoded.

Anti-overfitting
----------------
No task ids, customer names, KB content, or gold action sets are referenced.
The augmenter would attach the same notice to any future task that uses
these tau2 entry-point tools. If the underlying tau2 audit semantics ever
changed (the ``add_to_db`` calls were removed from the WRITE entry points),
the notice would still be merely informational — it never overrides the
LLM's decision to call or not call a tool.
"""
from __future__ import annotations

from typing import Optional

# Set of tau2 framework entry-point tool names that wrap discoverable-tool
# operations. Each is a stable tau2 method name (see
# ``tau2/domains/banking_knowledge/tools.py``).
_DISC_ENTRY_POINTS = {
    "unlock_discoverable_agent_tool",
    "call_discoverable_agent_tool",
    "give_discoverable_user_tool",
}

# Notices keyed to the entry-point tool name. Each describes a tau2 sandbox
# fact verifiable from tau2 source code — not a policy verdict.
_NOTICES = {
    "unlock_discoverable_agent_tool": (
        "[AUDIT NOTE] Unlocking does NOT write to the evaluation DB. However, "
        "any subsequent `call_discoverable_agent_tool(agent_tool_name=...)` "
        "to this just-unlocked tool WILL write one row to the "
        "`agent_discoverable_tools` table, keyed by the inner discoverable "
        "tool name (this is true even when the tool name starts with `get_` "
        "— tau2 marks `call_discoverable_agent_tool` as a WRITE tool). The "
        "evaluation DB hash includes every such row. Only proceed to call if "
        "the tool's output will materially change a tool call you would "
        "otherwise make for the user's stated request."
    ),
    "call_discoverable_agent_tool": (
        "[AUDIT NOTE] One row has been written to the `agent_discoverable_tools` "
        "table under this discoverable tool name. Additional unique-tool calls "
        "not strictly required for the user's stated request will add more rows "
        "and diverge the evaluation DB hash. Re-invoking the same discoverable "
        "tool name (same record id) does not add new rows."
    ),
    "give_discoverable_user_tool": (
        "[AUDIT NOTE] One row has been written to the `user_discoverable_tools` "
        "table under this discoverable tool name. Re-giving the same tool to "
        "the user does not add new rows; giving an additional tool not "
        "required for the user's request will diverge the evaluation DB hash."
    ),
}


def lookup_entry_point(tool_call_id: str, history_messages) -> Optional[str]:
    """Find the assistant tool_call name that produced ``tool_call_id``.

    Walks the conversation history in reverse looking for an assistant
    message whose ``tool_calls`` includes ``tool_call_id``. Returns the
    matching tool_call's ``name`` field, or ``None`` if not found.
    """
    if not tool_call_id:
        return None
    for prev in reversed(history_messages):
        tool_calls = getattr(prev, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in tool_calls:
            if getattr(tc, "id", None) == tool_call_id:
                return getattr(tc, "name", None)
    return None


def is_discoverable_entry_point(tool_name: Optional[str]) -> bool:
    """Return True if ``tool_name`` is a tau2 discoverable-tool entry point."""
    return tool_name in _DISC_ENTRY_POINTS


def notice_for(tool_name: str) -> Optional[str]:
    """Return the deterministic audit notice for an entry-point tool name."""
    return _NOTICES.get(tool_name)


SYSTEM_PROMPT_SUFFIX = (
    "## Discoverable Tool Audit Semantics (tau2 system fact)\n"
    "Two tau2 framework entry-point tools write to the evaluation DB:\n"
    "  - `call_discoverable_agent_tool` writes one row to the "
    "`agent_discoverable_tools` table per unique inner discoverable tool name "
    "(re-calls with the same tool name are deduplicated).\n"
    "  - `give_discoverable_user_tool` writes one row to the "
    "`user_discoverable_tools` table per unique inner tool name.\n"
    "  - `unlock_discoverable_agent_tool` does NOT write to the DB on its own; "
    "the write happens only when the unlocked tool is subsequently called.\n"
    "These rows participate in the DB hash used to evaluate task success. "
    "Tools whose name starts with `get_` (e.g. `get_user_dispute_history_*`, "
    "`get_pending_replacement_orders_*`) are still tau2 WRITE tools when "
    "invoked through `call_discoverable_agent_tool` and add audit rows. "
    "Only unlock, call, or give a discoverable tool when its output will "
    "materially change a tool call you would otherwise make for the user's "
    "stated request. Avoid speculative \"research\" calls."
)
