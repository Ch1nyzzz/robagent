"""Component core types for the SOP-Bench function-calling harness.

A Component attaches a (matcher, handler) pair to a `Mount` point in the
SopBenchAgent multi-turn FC loop. The handler returns a `Decision`; the
runtime applies it according to the class×mount×decision permission matrix
in `policy.py`.

Sibling of `agent/component_runtime/types.py` (GAIA, single-shot) and
`agent_tau2/component_runtime/types.py` (tau2). Each runtime defines its
own `Mount` enum matching its agent's lifecycle. SOP-Bench's mounts cover:

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


# --- enums -------------------------------------------------------------------


class Mount(str, Enum):
    """Lifecycle points a SOP-Bench component can attach to.

    Dispatch order within one task:
      SESSION_START
        → PRE_PROMPT_BUILD
        → (loop) PRE_LLM_TURN → (LLM call) → POST_LLM_RESPONSE
            → (for each tool call: PRE_TOOL_USE → (tool) → POST_TOOL_USE)
        → PRE_FINAL_EMIT
        → SESSION_END
    """
    SESSION_START      = "session_start"        # static system-level injection
    PRE_PROMPT_BUILD   = "pre_prompt_build"     # per-task; rewrite user prompt or inject system text
    PRE_LLM_TURN       = "pre_llm_turn"         # per-turn; rewrite messages list
    POST_LLM_RESPONSE  = "post_llm_response"    # per-turn; inspect assistant content + tool_calls
    PRE_TOOL_USE       = "pre_tool_use"         # per tool call; rewrite args or block
    POST_TOOL_USE      = "post_tool_use"        # per tool call; rewrite result string before adding to messages
    PRE_FINAL_EMIT     = "pre_final_emit"       # gate the final XML output
    SESSION_END        = "session_end"          # bookkeeping


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


@dataclass
class ComponentContext:
    """Argument to every matcher / handler.

    One ComponentContext is constructed per task and threaded through every
    mount dispatched for that task. Runtime-owned fields (system_prompt,
    user_prompt, messages, raw_response, tool_calls, current_tool_*,
    final_output) are mutated only by the dispatcher applying Decisions or
    by the agent between mount points. Handlers may freely read/write
    `ctx.shared` and `ctx.state[component_name]`.
    """
    mount: Mount
    # Stable per-task context (read-only after PRE_PROMPT_BUILD):
    benchmark: str                                  # domain slug, e.g. "dangerous_goods"
    task_id: str
    sop_text: str = ""
    task_input: dict = field(default_factory=dict)  # task.inputs dict
    tool_specs: list = field(default_factory=list)  # bedrock-format toolspec list
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

    # Final emit (mutated at PRE_FINAL_EMIT):
    final_output: str = ""

    # Control flow:
    blocked: bool = False
    blocked_reason: str = ""

    # Cumulative audit / shared state:
    executed_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    shared: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)       # per-component name → state dict
    log: Any = None


Ctx = ComponentContext

Matcher = Callable[[ComponentContext], bool]
Handler = Callable[[ComponentContext], Decision]


# --- component ---------------------------------------------------------------


@dataclass(frozen=True)
class Component:
    name: str
    cls: ComponentClass
    mount: Mount
    matcher: Optional[Matcher]
    handler: Handler
    trust: Trust
    state_scope: StateScope = StateScope.NONE
    capabilities: tuple[Capability, ...] = (Capability.NONE,)
    priority: int = 100              # smaller fires first
    # Phase B (event-runtime migration) additions:
    #   `listens` is the dispatcher subscription key. Defaults via
    #   __post_init__ to `mount.value` so existing components migrate
    #   transparently. `emits` self-documents custom Tier-2/3 events.
    listens: str = ""
    emits: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.listens:
            object.__setattr__(self, "listens", self.mount.value)


def matcher_for_tool(tool_name: str) -> Matcher:
    """Build a matcher that fires only when PRE_TOOL_USE / POST_TOOL_USE
    targets one specific tool name. Useful for tool-scoped guards.
    """
    def _m(ctx: ComponentContext) -> bool:
        return ctx.current_tool_name == tool_name
    _m.__name__ = f"matches_tool_{tool_name}"
    return _m
