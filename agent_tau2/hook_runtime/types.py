"""Hook core types.

A Hook attaches a (matcher, handler) pair to a lifecycle event of the agent.
The handler returns a Decision; the runtime applies it according to the
class×event×decision permission matrix in `policy.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

# --- enums -------------------------------------------------------------------


class HookEvent(str, Enum):
    SESSION_START = "session_start"           # once, before any LLM call
    USER_PROMPT_SUBMIT = "user_prompt_submit" # incoming UserMessage, pre-LLM
    PRE_TOOL_USE = "pre_tool_use"             # per generated tool_call, pre-emit
    POST_TOOL_USE = "post_tool_use"           # incoming ToolMessage from env
    STOP = "stop"                             # agent attempts to end the turn
    SESSION_END = "session_end"               # once, after final message


class HookClass(str, Enum):
    """Durability classes admissible into a hook. Enforced at registration
    by `policy.ALLOWED`.

    The two HIGH-risk classes from robust-harness-tau2 (`induced_rule` and
    `predictive_heuristic`) are intentionally not present here: a hook
    whose behaviour is INDUCED from finite training evidence (a reading
    of policy text, a regex over conversation) has no place in a
    deterministic harness. If your hook would need to be one of those,
    redesign it to point at structure — a system field, a tool's declared
    schema, a protocol invariant, a general algorithm — or do not write
    it. (The skill itself runs LLMs; that is where un-anchored judgement
    belongs, not in the harness around it.)
    """
    CHANNEL = "channel"
    REACTIVE_GUARD = "reactive_guard"
    DETERMINISTIC_GLUE = "deterministic_glue"


class DecisionKind(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"                       # drop the tool_call or terminate
    REWRITE_TOOL_ARGS = "rewrite_tool_args"
    DEFER = "defer"                       # postpone the call until predicate fires
    INJECT_CONTEXT = "inject_context"     # append text to prompt / state


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
    def rewrite_tool_args(new_args: dict) -> "Decision":
        return Decision(DecisionKind.REWRITE_TOOL_ARGS, payload=new_args)

    @staticmethod
    def defer(replay_when: Callable[["HookContext"], bool]) -> "Decision":
        return Decision(DecisionKind.DEFER, payload=replay_when)

    @staticmethod
    def inject_context(text: str) -> "Decision":
        return Decision(DecisionKind.INJECT_CONTEXT, payload=text)


# --- context -----------------------------------------------------------------


@dataclass
class HookContext:
    """The argument passed to every hook matcher / handler.

    A single HookContext is reused across all hooks firing at the same event,
    so handlers may *read* `shared` but should not mutate the fields the
    runtime owns (`tool_call`, `incoming_message`, `assistant_message`).
    """
    event: HookEvent
    # Convenience accessors (populated per event):
    tool_call: Optional[Any] = None              # ToolCall, for PRE_TOOL_USE
    incoming_message: Optional[Any] = None       # UserMessage / ToolMessage
    assistant_message: Optional[Any] = None      # AssistantMessage (STOP)
    # Always populated:
    domain_policy: str = ""
    tool_names: tuple[str, ...] = ()
    history: list = field(default_factory=list)  # state.messages snapshot
    shared: dict = field(default_factory=dict)   # cross-hook scratchpad

    # Sugar for the common PRE_TOOL_USE matcher pattern.
    @property
    def tool_name(self) -> Optional[str]:
        return self.tool_call.name if self.tool_call is not None else None

    @property
    def tool_args(self) -> dict:
        return dict(self.tool_call.arguments) if self.tool_call is not None else {}


# --- hook --------------------------------------------------------------------


Matcher = Callable[[HookContext], bool]
Handler = Callable[[HookContext], Decision]


@dataclass(frozen=True)
class Hook:
    """A registered hook.

    `name` is the stable identifier. To MODIFY a hook in a later iteration,
    publish a new file whose HOOK.name reuses the existing name — the
    registry replaces it. To REMOVE, list the name in the manifest's
    `frontier_hooks_remove`.
    """
    name: str
    cls: HookClass
    event: HookEvent
    matcher: Optional[Matcher]
    handler: Handler
    generalization_argument: str
    fallback: str
    dead_when: str
