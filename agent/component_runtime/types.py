"""Component core types for the GAIA harness.

A Component attaches a (matcher, handler) pair to a `Mount` point in the
GAIA single-task run loop. The handler returns a Decision; the runtime
applies it according to the class×mount×decision permission matrix in
`policy.py`.

Sibling of `agent_tau2/component_runtime/types.py`. The two runtimes share
the durability axis (ComponentClass, Trust, StateScope, Capability) and
the patch/workflow vocabulary, but each defines its own `Mount` enum
matching the base agent's lifecycle:

  * tau2 agent is multi-turn tool-use → mounts include PRE_TOOL_USE /
    POST_TOOL_USE / POST_LLM_RESPONSE for sub-LLM verifiers.
  * GAIA agent is single-shot prompt→answer → mounts cover prompt
    construction, LLM response post-processing, and answer normalisation.
    No tool-use mounts in v1 (most GAIA agents don't use the tool_call
    protocol; tool-like behaviour lives inside individual components via
    capabilities).

Each runtime evolves independently. Sharing one Mount enum across both
would force one set of semantics to bend; duplication is the cleaner
boundary at v1.
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
# Phase C: ComponentContext inherits EventContext so component handlers can
# call `ctx.chat(...)` / `ctx.emit(custom_event, ...)` / `ctx.emit_upstream(...)`
# uniformly with the other 4 siblings. EventContext owns the cross-event
# scratchpads (`shared`, `state`, `persistent_state`, `upstream`), the
# `blocked` / `blocked_reason` signal pair, and the `_impl_*` capability
# hooks the per-bench dispatcher wires at construction time.
from meta_harness.component_runtime_core.event_context import EventContext


# --- enums -------------------------------------------------------------------


class Mount(str, Enum):
    """Lifecycle points a GAIA component can attach to.

    Dispatch order within one task:
      SESSION_START → PRE_PROMPT_BUILD → (LLM) → POST_LLM_RESPONSE →
      (default answer extract) → PRE_ANSWER_EMIT → SESSION_END.

    v1 dispatches SESSION_START / PRE_PROMPT_BUILD / POST_LLM_RESPONSE /
    PRE_ANSWER_EMIT. SESSION_END is declared and load-time validated but
    only used for bookkeeping in v1 (no decision honoured).
    """
    SESSION_START      = "session_start"        # static, framework-invariant injection
    PRE_PROMPT_BUILD   = "pre_prompt_build"     # per-task; can rewrite prompt or block
    POST_LLM_RESPONSE  = "post_llm_response"    # raw LLM content available; rewrite / block / inject
    PRE_ANSWER_EMIT    = "pre_answer_emit"      # final extracted answer; normalise / block
    SESSION_END        = "session_end"          # bookkeeping only (v1)


class DecisionKind(str, Enum):
    ALLOW          = "allow"
    BLOCK          = "block"            # mark blocked; answer→None with reason
    REWRITE        = "rewrite"          # replace live payload (prompt / response / answer per mount)
    INJECT_CONTEXT = "inject_context"   # append text to system_prompt (PRE) or as recovery context (POST)


class Capability(str, Enum):
    NONE           = "none"
    READ_FILE      = "read_file"
    HTTP_GET       = "http_get"
    LLM_CALL       = "llm_call"        # recovery passes, sub-LLM extraction
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
    def rewrite(payload: str) -> "Decision":
        """Replace the live payload at this mount.

        Mount → payload semantics:
          PRE_PROMPT_BUILD  → payload is the new task_prompt text.
          POST_LLM_RESPONSE → payload is the new raw response content.
          PRE_ANSWER_EMIT   → payload is the new final answer string (or None to mark blocked).
        """
        return Decision(DecisionKind.REWRITE, payload=payload)

    @staticmethod
    def inject_context(text: str) -> "Decision":
        """Append text to the system_prompt (SESSION_START / PRE_PROMPT_BUILD)
        or to the next recovery LLM call's context (POST_LLM_RESPONSE).
        """
        return Decision(DecisionKind.INJECT_CONTEXT, payload=text)


# --- context -----------------------------------------------------------------


@dataclass(kw_only=True)
class ComponentContext(EventContext):
    """Argument to every matcher / handler.

    Reused across components firing at the same mount / event. Components
    may read `shared` / `state` / `upstream`; runtime-owned fields
    (`prompt` / `raw_response` / `answer`) are mutated only by the
    dispatcher applying Decisions.

    Inherits from EventContext (Phase C): `event`, `task_id`, `shared`,
    `state`, `persistent_state`, `upstream`, `blocked`, `blocked_reason`,
    plus the `ctx.chat(...)` / `ctx.fetch(...)` / `ctx.read_file(...)` /
    `ctx.emit(...)` / `ctx.emit_upstream(...)` capability methods.

    `@dataclass(kw_only=True)` is required because EventContext fields all
    have defaults; without kw_only Python would refuse to add non-default
    subclass fields (mount / benchmark / extras) after default fields.
    All existing call sites already use kwargs (see base.py::run_task).
    """
    # Mount-specific payloads (only the relevant ones are populated):
    mount: Mount                                  # most recent Mount enum (legacy)
    benchmark: str                                # bench slug
    extras: dict                                  # per-task extras
    system_prompt: str = ""                       # current base system prompt
    prompt: Optional[str] = None                  # task_prompt at PRE_PROMPT_BUILD
    raw_response: Optional[str] = None            # LLM raw content at POST_LLM_RESPONSE / PRE_ANSWER_EMIT
    answer: Optional[str] = None                  # extracted answer at PRE_ANSWER_EMIT
    log: Any = None                               # EventLog (read-only access for handlers)


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
    priority: int = 100
    # Phase B (event-runtime migration) additions:
    #   `listens` is the dispatcher subscription key. If left empty, the
    #   __post_init__ below back-fills it from `mount.value` so existing
    #   mount-based components keep working under the new dispatcher with
    #   no source change. New components may set `listens="custom_event"`
    #   and use `mount` purely as a placeholder for policy validation.
    #   `emits` self-documents the custom Tier-2/3 events this component
    #   raises via `ctx.emit(...)`; the runtime does not enforce, the
    #   field is read by skill / proposer tooling for event-name discovery.
    listens: str = ""
    emits: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.listens:
            object.__setattr__(self, "listens", self.mount.value)
