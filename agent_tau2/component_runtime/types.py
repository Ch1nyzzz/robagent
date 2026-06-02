"""Component core types.

A Component attaches a (matcher, handler) pair to a workflow Mount of the
agent. The handler returns a Decision; the runtime applies it according to
the class×mount×decision permission matrix in `policy.py`.

Compared to the predecessor `hook_runtime`:

  * `HookEvent` → `Mount`, with two NEW values `PRE_CONTEXT_BUILD` (fires
    before `system_prompt` is frozen — for retrieval-style channels) and
    `POST_LLM_RESPONSE` (fires after the LLM emits an AssistantMessage,
    before tool dispatch — for sub-LLM verifier patterns).
  * `HookClass` → `ComponentClass`, with 5 values: `MECHANISM_LAYER`
    (renamed from `DETERMINISTIC_GLUE`), `REACTIVE_GUARD`, `CHANNEL`,
    `INDUCED_RULE` (REINSTATED but admitted advisory-only — see policy.py),
    `PREDICTIVE_HEURISTIC` (declared for serialization but rejected at
    load time).
  * `Hook` → `Component`. The three free-form trust strings
    (`generalization_argument` / `fallback` / `dead_when`) collapse into a
    structured `Trust` dataclass with a required `out_of_evidence_probe`
    for INDUCED_RULE.
  * New field: `priority` (ordering within an event bucket).

A `Component` is a frozen dataclass loaded once per file and looked up by
name; it carries no runtime state itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

# Shared-across-siblings types: lifted to the unified core in Phase A of
# the event-runtime migration. ComponentClass / Trust are byte-identical
# (5/5 siblings) so they only live in one place now.
from ballast.component_runtime_core.shared_types import (
    ComponentClass,
    Trust,
)
# Phase C: ComponentContext inherits EventContext for capability methods.
from ballast.component_runtime_core.event_context import EventContext


# --- enums -------------------------------------------------------------------




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
    def defer(replay_when: Callable[["ComponentContext"], bool]) -> "Decision":
        return Decision(DecisionKind.DEFER, payload=replay_when)

    @staticmethod
    def inject_context(text: str) -> "Decision":
        return Decision(DecisionKind.INJECT_CONTEXT, payload=text)


# --- context -----------------------------------------------------------------


@dataclass(kw_only=True)
class ComponentContext(EventContext):
    """The argument passed to every Component matcher / handler.

    A single ComponentContext is reused across all Components firing at
    the same mount/event, so handlers may *read* `shared` but should not
    mutate the runtime-owned fields (`tool_call`, `incoming_message`,
    `assistant_message`, `history`).

    Inherits from EventContext (Phase C): `event`, `task_id`, `shared`,
    `state`, `persistent_state`, `upstream`, `blocked`, `blocked_reason`,
    plus the `ctx.chat` / `ctx.emit` / `ctx.emit_upstream` capability
    methods. `@dataclass(kw_only=True)` is required because EventContext
    has default fields.

    `state` is keyed by Component.name; the dispatcher populates it from
    the agent's `_session_state` (for SESSION scope) or the disk-backed
    cross-session store (CROSS_SESSION scope; reserved in v1).
    """
    # Convenience accessors (populated per mount):
    tool_call: Optional[Any] = None              # ToolCall, for PRE_TOOL_USE
    incoming_message: Optional[Any] = None       # UserMessage / ToolMessage
    assistant_message: Optional[Any] = None      # AssistantMessage, for POST_LLM_RESPONSE / STOP
    proposed_system_prompt: Optional[str] = None  # PRE_CONTEXT_BUILD only
    # Always populated:
    domain_policy: str = ""
    tool_names: tuple[str, ...] = ()
    history: list = field(default_factory=list)  # state.messages snapshot

    # Sugar for the common PRE_TOOL_USE matcher pattern.
    @property
    def tool_name(self) -> Optional[str]:
        return self.tool_call.name if self.tool_call is not None else None

    @property
    def tool_args(self) -> dict:
        return dict(self.tool_call.arguments) if self.tool_call is not None else {}


# --- aliases -----------------------------------------------------------------


# Short alias used inside Component handler signatures so component files
# can write `def _matches(ctx: Ctx) -> bool:` without importing the long
# name.
Ctx = ComponentContext

Matcher = Callable[[ComponentContext], bool]
Handler = Callable[[ComponentContext], Decision]


# --- component ---------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Component:
    """A registered Component.

    `name` is the stable identifier. To MODIFY a component in a later
    iteration (the `replace_node` workflow op), publish a new file whose
    `COMPONENT.name` reuses the existing name — the registry replaces it.
    To REMOVE (`disable_node`), the workflow YAML sets `disabled: [name]`
    and the file remains on disk for the durability audit.
    """
    name: str
    cls: ComponentClass
    matcher: Optional[Matcher]
    handler: Handler
    trust: Trust
    priority: int = 100              # smaller fires first
    listens: str
    emits: tuple[str, ...] = ()



# --- helpers -----------------------------------------------------------------


def matcher_for_tool(tool_name: str) -> Matcher:
    """Build a matcher that fires only for one tool's PRE_TOOL_USE.

    Use this when you want a "wrap-tool" component without a new Mount:
    a PRE_TOOL_USE component whose matcher is gated on `tool_call.name`.
    """
    def _m(ctx: ComponentContext) -> bool:
        return ctx.tool_name == tool_name
    _m.__name__ = f"matches_tool_{tool_name}"
    return _m
