"""Component core types for the EnterpriseOps-Gym function-calling harness.

A Component attaches a (matcher, handler) pair to a `Mount` point in the
EnterpriseOpsAgent multi-turn function-calling loop. The handler returns a
`Decision`; the runtime applies it according to the class×mount×decision
permission matrix in `policy.py`.

Sibling of `agent/component_runtime_sopbench/types.py` (SOP-Bench, same
mount set) and the GAIA / tau2 sibling runtimes. EnterpriseOps mounts:

  * static + per-task: SESSION_START, PRE_PROMPT_BUILD, SESSION_END
  * per turn in the FC loop: PRE_LLM_TURN, POST_LLM_RESPONSE
  * per tool call inside a turn: PRE_TOOL_USE, POST_TOOL_USE
  * exit gate: PRE_FINAL_EMIT
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

# Shared-across-siblings types: lifted to the unified core in Phase A of
# the event-runtime migration. ComponentClass / StateScope / Trust are
# byte-identical (5/5 siblings) so they only live in one place now.
from meta_harness.component_runtime_core.shared_types import (
    ComponentClass,
    StateScope,
    Trust,
)
# Phase C: ComponentContext inherits EventContext for capability methods
# (ctx.chat / ctx.emit / ctx.emit_upstream) and the shared cross-event
# scratchpads. The per-bench Dispatcher wires `_impl_*` at construction.
from meta_harness.component_runtime_core.event_context import EventContext


# --- enums -------------------------------------------------------------------




class DecisionKind(str, Enum):
    ALLOW          = "allow"
    BLOCK          = "block"            # terminate loop / drop tool call
    REWRITE        = "rewrite"          # replace mount-specific in-flight payload
    INJECT_CONTEXT = "inject_context"   # append text into system_prompt or post-llm inject queue


class Capability(str, Enum):
    NONE           = "none"
    READ_FILE      = "read_file"
    HTTP_GET       = "http_get"
    LLM_CALL       = "llm_call"        # sub-LLM verifier / re-format / critic
    TOOL_CALL      = "tool_call"       # sub-tool dispatch beyond the model's choices
    MUTATE_SHARED  = "mutate_shared"


# --- decision ----------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    kind: DecisionKind
    payload: Any = None
    reason: str = ""

    @staticmethod
    def allow() -> "Decision":
        return Decision(DecisionKind.ALLOW)

    @staticmethod
    def block(reason: str) -> "Decision":
        return Decision(DecisionKind.BLOCK, reason=reason)

    @staticmethod
    def rewrite(payload: Any) -> "Decision":
        """Replace the live payload at this mount.

        Mount → payload semantics:
          PRE_PROMPT_BUILD  → str: new user_prompt
          PRE_LLM_TURN      → list[dict]: new messages list
          POST_LLM_RESPONSE → str: new assistant content
                              (does NOT touch tool_calls — to block a tool
                              call, use PRE_TOOL_USE BLOCK)
          PRE_TOOL_USE      → dict: new tool args (name stays the same)
          POST_TOOL_USE     → str: new tool-result content fed back to model
          PRE_FINAL_EMIT    → str | None: final output (None marks blocked)
        """
        return Decision(DecisionKind.REWRITE, payload=payload)

    @staticmethod
    def inject_context(text: str) -> "Decision":
        """Append text to system_prompt (SESSION_START / PRE_PROMPT_BUILD)
        or to the post-LLM inject queue (POST_LLM_RESPONSE). The agent reads
        the queue at the start of the next turn and adds it as a synthetic
        system reminder message before the next chat call.
        """
        return Decision(DecisionKind.INJECT_CONTEXT, payload=text)


# --- context -----------------------------------------------------------------


@dataclass(kw_only=True)
class ComponentContext(EventContext):
    """Argument to every matcher / handler.

    Threaded through every mount/event in one task. Runtime-owned fields
    are mutated by the dispatcher applying Decisions or by the agent
    between mount points. Handlers may freely read/write `ctx.shared`
    and `ctx.state[component_name]`.

    Inherits from EventContext (Phase C): event, task_id, shared, state,
    persistent_state, upstream, blocked, blocked_reason, plus capability
    methods (ctx.chat / ctx.emit / ctx.emit_upstream / ctx.fetch / ctx.read_file).
    `@dataclass(kw_only=True)` lets non-default mount/benchmark fields
    come after EventContext's default fields.
    """
    # Stable per-task context (read-only after PRE_PROMPT_BUILD):
    benchmark: str                                  # domain slug, e.g. "calendar", "itsm"
    user_info: dict = field(default_factory=dict)   # {user_id, name, email, timezone}
    gym_servers: list = field(default_factory=list) # list of {mcp_server_name, mcp_server_url, ...}
    tool_specs: list = field(default_factory=list)  # list of {name, description, input_schema, _mcp_server_name, ...}
    selected_tools: list = field(default_factory=list)  # oracle / plus_N expected names
    verifiers: list = field(default_factory=list)   # raw verifier configs (for inspection only)
    extras: dict = field(default_factory=dict)

    # Prompt state (mutated at SESSION_START / PRE_PROMPT_BUILD):
    system_prompt: str = ""
    user_prompt: str = ""

    # Per-turn state (mutated at PRE_LLM_TURN / POST_LLM_RESPONSE):
    messages: list[dict[str, Any]] = field(default_factory=list)
    raw_response: str = ""                          # current assistant content
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    turn_index: int = 0
    finish_reason: str = ""

    # Per-tool-call state (mutated at PRE_TOOL_USE / POST_TOOL_USE):
    current_tool_name: str = ""
    current_tool_args: dict[str, Any] = field(default_factory=dict)
    current_tool_result: Any = None
    current_tool_result_str: str = ""               # the string that will be fed back to the model
    current_tool_call_id: str = ""
    current_tool_success: bool = True
    current_tool_error: Optional[str] = None
    current_tool_server: str = ""                    # which mcp_server_name routes this tool

    # Final emit (mutated at PRE_FINAL_EMIT):
    final_output: str = ""

    # Cumulative audit:
    executed_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    log: Any = None


Ctx = ComponentContext

Matcher = Callable[[ComponentContext], bool]
Handler = Callable[[ComponentContext], Decision]


# --- component ---------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Component:
    name: str
    cls: ComponentClass
    matcher: Optional[Matcher]
    handler: Handler
    trust: Trust
    state_scope: StateScope = StateScope.NONE
    capabilities: tuple[Capability, ...] = (Capability.NONE,)
    priority: int = 100              # smaller fires first
    listens: str
    emits: tuple[str, ...] = ()



def matcher_for_tool(tool_name: str) -> Matcher:
    """Build a matcher that fires only when PRE_TOOL_USE / POST_TOOL_USE
    targets one specific tool name. Useful for tool-scoped guards.
    """
    def _m(ctx: ComponentContext) -> bool:
        return ctx.current_tool_name == tool_name
    _m.__name__ = f"matches_tool_{tool_name}"
    return _m


def matcher_for_server(server_name: str) -> Matcher:
    """Build a matcher that fires only when PRE_TOOL_USE / POST_TOOL_USE
    targets one specific MCP gym server. Useful in multi-gym (hybrid) tasks.
    """
    def _m(ctx: ComponentContext) -> bool:
        return ctx.current_tool_server == server_name
    _m.__name__ = f"matches_server_{server_name}"
    return _m
