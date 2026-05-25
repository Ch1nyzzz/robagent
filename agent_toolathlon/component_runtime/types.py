"""Component core types for the toolathlon component runtime.

Ported from `agent_tau2/component_runtime/types.py`; the type surface is
deliberately identical across benchmarks so a single SKILL.md prose set
keeps working. Differences vs tau2 are confined to dispatch (`agent.py`)
and the class×mount permission matrix (`policy.py`):

  * SDK constraint: `AgentHooks.on_tool_start` does not surface the
    ToolCall arguments — only the tool object. So `PRE_TOOL_USE` v1 can
    ALLOW or BLOCK; REWRITE_TOOL_ARGS / DEFER are not yet implementable
    without wrapping every MCP tool as a FunctionTool. `policy.py`
    rejects those decisions at load time for toolathlon.
  * SDK constraint: no mid-turn AssistantMessage hook for `tool_calls`
    emission. `POST_LLM_RESPONSE` (sub-LLM verifier slot) is declared in
    Mount and validated, but not dispatched in v1.
  * `tool_args` is empty on `PRE_TOOL_USE` for toolathlon (see above).

A Component attaches a (matcher, handler) pair to a workflow Mount of the
agent. The handler returns a Decision; the runtime applies it according to
the class×mount×decision permission matrix in `policy.py`.

A `Component` is a frozen dataclass loaded once per file and looked up by
name; it carries no runtime state itself.
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
# scratchpads. The dispatcher wires `_impl_*` at construction time.
from meta_harness.component_runtime_core.event_context import EventContext


# --- enums -------------------------------------------------------------------


class Mount(str, Enum):
    """Lifecycle / workflow points a Component can attach to.

    Dispatch order for a single user→assistant round-trip (toolathlon v1):
      PRE_CONTEXT_BUILD → SESSION_START → USER_PROMPT_SUBMIT → (LLM+tools) →
      POST_TOOL_USE → SESSION_END.

    v1 actually dispatches PRE_CONTEXT_BUILD, SESSION_START,
    USER_PROMPT_SUBMIT, PRE_TOOL_USE (ALLOW/BLOCK only), POST_TOOL_USE.
    POST_LLM_RESPONSE / STOP / SESSION_END are declared and load-time
    validated but not yet dispatched.
    """
    PRE_CONTEXT_BUILD   = "pre_context_build"
    SESSION_START       = "session_start"
    USER_PROMPT_SUBMIT  = "user_prompt_submit"
    PRE_TOOL_USE        = "pre_tool_use"
    POST_LLM_RESPONSE   = "post_llm_response"     # reserved, not dispatched in v1
    POST_TOOL_USE       = "post_tool_use"
    STOP                = "stop"                  # reserved
    SESSION_END         = "session_end"           # reserved


class DecisionKind(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"                       # drop the tool_call / raise / terminate
    REWRITE_TOOL_ARGS = "rewrite_tool_args"
    DEFER = "defer"                       # postpone the call until predicate fires
    INJECT_CONTEXT = "inject_context"     # append text to prompt / state


class Capability(str, Enum):
    """Explicit allowlist of side-effects a Component handler may perform.

    Declared at construction; v1 records it in the manifest but does not
    yet sandbox at fire time. Acts as a structural contract today.
    """
    NONE           = "none"
    READ_FILE      = "read_file"
    HTTP_GET       = "http_get"
    TOOL_CALL      = "tool_call"     # sub-tool-call (retriever pattern)
    LLM_CALL       = "llm_call"      # sub-LLM (verifier pattern)
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
    def rewrite_tool_args(new_args: dict) -> "Decision":
        return Decision(DecisionKind.REWRITE_TOOL_ARGS, payload=new_args)

    @staticmethod
    def defer(replay_when: Callable[["ComponentContext"], bool]) -> "Decision":
        return Decision(DecisionKind.DEFER, payload=replay_when)

    @staticmethod
    def inject_context(text: str) -> "Decision":
        return Decision(DecisionKind.INJECT_CONTEXT, payload=text)


# --- context -----------------------------------------------------------------


@dataclass(kw_only=True)
class ComponentContext(EventContext):
    """The argument passed to every Component matcher / handler.

    Reused across components firing at the same mount/event. Handlers may
    *read* `shared` but should not mutate the runtime-owned fields
    (`tool_call`, `incoming_message`, `assistant_message`, `history`).

    Inherits from EventContext (Phase C): `event`, `task_id`, `shared`,
    `state`, `persistent_state`, `upstream`, `blocked`, `blocked_reason`,
    plus `ctx.chat` / `ctx.emit` / `ctx.emit_upstream` capability methods.
    `@dataclass(kw_only=True)` is required because EventContext has
    default fields and the subclass adds non-default `mount`.

    Toolathlon notes:
      * `tool_call.arguments` is always {} on PRE_TOOL_USE via SDK hook
        (SDK does not surface arguments at on_tool_start). Use `tool_name`
        only. The FunctionTool wrapper (v2) surfaces REAL arguments.
      * `incoming_message` on POST_TOOL_USE is the SDK ToolCallOutput
        item; the runtime exposes a thin dict with `tool_name` and
        `output` for matcher convenience.
    """
    mount: Mount
    tool_call: Optional[Any] = None              # ToolCall-like, for PRE_TOOL_USE
    incoming_message: Optional[Any] = None       # POST_TOOL_USE: dict(tool_name, output)
    assistant_message: Optional[Any] = None      # reserved (POST_LLM_RESPONSE)
    proposed_system_prompt: Optional[str] = None  # PRE_CONTEXT_BUILD only
    domain_policy: str = ""
    tool_names: tuple[str, ...] = ()
    history: list = field(default_factory=list)  # task_agent.logs snapshot

    @property
    def tool_name(self) -> Optional[str]:
        if self.tool_call is None:
            return None
        return getattr(self.tool_call, "name", None) or self.tool_call.get("name")  # type: ignore[union-attr]

    @property
    def tool_args(self) -> dict:
        if self.tool_call is None:
            return {}
        args = getattr(self.tool_call, "arguments", None)
        if args is None and isinstance(self.tool_call, dict):
            args = self.tool_call.get("arguments")
        return dict(args) if args else {}


# --- aliases -----------------------------------------------------------------


Ctx = ComponentContext

Matcher = Callable[[ComponentContext], bool]
Handler = Callable[[ComponentContext], Decision]


# --- component ---------------------------------------------------------------


@dataclass(frozen=True)
class Component:
    """A registered Component.

    `name` is the stable identifier. To MODIFY in a later iteration
    (`replace_node`), publish a new file whose `COMPONENT.name` reuses
    the existing name. To REMOVE (`disable_node`), the workflow YAML
    sets `disabled: [name]` and the file remains on disk for audit.
    """
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


# --- helpers -----------------------------------------------------------------


def matcher_for_tool(tool_name: str) -> Matcher:
    """Build a matcher that fires only for one tool's PRE_TOOL_USE."""
    def _m(ctx: ComponentContext) -> bool:
        return ctx.tool_name == tool_name
    _m.__name__ = f"matches_tool_{tool_name}"
    return _m
