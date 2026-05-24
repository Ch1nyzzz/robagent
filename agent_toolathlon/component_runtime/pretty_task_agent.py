"""CrPrettyDecoupledTaskAgent — copy of Toolathlon-src
host_agent_loop.py::PrettyDecoupledTaskAgent with the base class swapped
to our CR-aware TaskAgent.

The original `PrettyDecoupledTaskAgent` only overrides
`process_agent_response` for console-pretty-printing. All other lifecycle
behaviour comes from `TaskAgent`. By inheriting from our copy in
`agent_toolathlon.component_runtime.task_agent`, the four CR_HOOK points
(PRE_CONTEXT_BUILD / SESSION_START / USER_PROMPT_SUBMIT / POST_TOOL_USE
flush) are active automatically.

The pretty-print helpers (`render_run_items_to_console_events`,
`print_log_line`, etc.) are imported from upstream — we don't duplicate
them, since the rendering is benchmark-independent.
"""
from __future__ import annotations

from typing import Dict, List

# Pretty-print helpers from upstream — keep importing from the vendored
# source so any rendering tweak there flows through automatically.
from scripts.decoupled.host_agent_loop import (  # type: ignore
    ANSI_BLUE,
    ANSI_GREEN,
    preview_text,
    print_log_line,
    render_run_items_to_console_events,
)

from .task_agent import TaskAgent


class PrettyDecoupledTaskAgent(TaskAgent):
    """Console-pretty variant of CrTaskAgent — see file docstring."""

    async def process_agent_response(self, result) -> List[Dict]:
        recent_tool_calls = await super().process_agent_response(result)
        events = render_run_items_to_console_events(result.new_items)

        print_log_line(
            "TURN",
            f"user_turn={self.stats['interaction_turns']} tool_calls={len(recent_tool_calls)}",
            ANSI_BLUE,
        )
        for tag, message, color in events:
            print_log_line(tag, message, color)
        if result.final_output:
            summary_text = str(result.final_output).strip()
            assistant_messages = [m for tag, m, _ in events if tag == "ASSIST"]
            if summary_text and summary_text not in assistant_messages:
                print_log_line("SUMMARY", preview_text(summary_text), ANSI_GREEN)

        return recent_tool_calls
