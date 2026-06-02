"""Component core types for the GAIA harness.

A Component attaches a (matcher, handler) pair to a `Mount` point in the
GAIA single-task run loop. The handler returns a Decision; the runtime
applies it according to the class×mount×decision permission matrix in
`policy.py`.

Sibling of `agent_tau2/component_runtime/types.py`. The two runtimes share
the durability axis (ComponentClass, Trust) and the patch/workflow
vocabulary, but each defines its own `Mount` enum matching the base
agent's lifecycle:

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
# the event-runtime migration. ComponentClass / Trust are byte-identical
# (5/5 siblings) so they only live in one place now.
from ballast.component_runtime_core.shared_types import (
    ComponentClass,
    Trust,
)
# Phase C: ComponentContext inherits EventContext so component handlers can
# call `ctx.chat(...)` / `ctx.emit(custom_event, ...)` / `ctx.emit_upstream(...)`
# uniformly with the other 4 siblings. EventContext owns the cross-event
# scratchpads (`shared`, `state`, `persistent_state`, `upstream`), the
# `blocked` / `blocked_reason` signal pair, and the `_impl_*` capability
# hooks the per-bench dispatcher wires at construction time.
from ballast.component_runtime_core.event_context import EventContext


# --- enums -------------------------------------------------------------------
#
# Lifecycle event names a GAIA component can subscribe to via `listens=`:
#
# Per-task setup (fires once each):
#   "task_received"        — lifecycle anchor right after ctx construction
#   "session_start"        — static, framework-invariant injection
#   "pre_prompt_build"     — per-task; can rewrite prompt or block
#   "pre_context_build"    — alias of pre_prompt_build (cross-sibling vocab)
#   "pre_agent_construct"  — last hook before messages list is sealed
#
# Per-turn (fires once per FC iteration; up to MAX_ITERATIONS times):
#   "pre_llm_request"      — just before the SUT chat() call this turn
#   "post_llm_response"    — raw LLM content available this turn
#   "post_llm_response_raw"— alias of post_llm_response
#   "on_length_truncation" — synthesised when finish_reason == "length" this turn
#   "on_empty_response"    — synthesised when this turn's raw_response is empty
#
# Per tool call within a turn (fires once per tool_call):
#   "pre_tool_use"         — REWRITE replaces ctx.current_tool_args (dict);
#                             BLOCK skips THIS tool call (returns "[skipped]"
#                             as the tool result; task continues)
#   "post_tool_use"        — REWRITE replaces ctx.current_tool_result (str);
#                             INJECT_CONTEXT is concatenated INTO the tool
#                             result so the LLM sees it on the next turn
#   "on_tool_error"        — synthesised when the tool result starts with
#                             "ERROR:"; typically inject a retry hint
#
# Per-task exit:
#   "pre_answer_emit"      — final extracted answer; normalise / block
#   "session_end"          — bookkeeping at end of task
#
# Plus any custom Tier-2/3 event a component declares via `emits=(...,)`
# and another component subscribes to via `listens="..."`.


class DecisionKind(str, Enum):
    ALLOW          = "allow"
    BLOCK          = "block"            # mark blocked; answer→None with reason
    REWRITE        = "rewrite"          # replace live payload (prompt / response / answer per mount)
    INJECT_CONTEXT = "inject_context"   # append text to system_prompt (PRE) or as recovery context (POST)


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
    def rewrite(payload) -> "Decision":
        """Replace the live payload at the firing event.

        Event → payload semantics:
          pre_prompt_build / pre_context_build  → new task_prompt text (str)
          post_llm_response / on_length_truncation / on_empty_response
                                                → new raw response content (str)
          pre_tool_use                          → new tool arguments (dict)
          post_tool_use / on_tool_error         → new tool result (str)
          pre_answer_emit                       → new final answer string
                                                  (or None to mark blocked)
        """
        return Decision(DecisionKind.REWRITE, payload=payload)

    @staticmethod
    def inject_context(text: str) -> "Decision":
        """Append text. Where it lands depends on the firing event:

          setup / pre-LLM events (session_start, pre_prompt_build,
            pre_context_build, task_received, pre_agent_construct,
            pre_llm_request)               → ctx.system_prompt
          post-LLM events (post_llm_response, on_length_truncation,
            on_empty_response, pre_tool_use)
                                           → ctx.shared['post_llm_inject']
                                              (replayed on next turn)
          post_tool_use / on_tool_error    → concatenated INTO
                                              ctx.current_tool_result so the
                                              LLM sees it on the next turn
        """
        return Decision(DecisionKind.INJECT_CONTEXT, payload=text)


# --- context -----------------------------------------------------------------


@dataclass(kw_only=True)
class ComponentContext(EventContext):
    """Argument to every matcher / handler.

    Reused across components firing at the same event. Components may read
    `shared` / `state` / `upstream`; runtime-owned fields (`prompt` /
    `raw_response` / `answer`) are mutated only by the dispatcher applying
    Decisions. Handlers identify the current event via `ctx.event` (set by
    the dispatcher just before firing each subscriber).

    Inherits from EventContext: `event`, `task_id`, `shared`, `state`,
    `persistent_state`, `upstream`, `blocked`, `blocked_reason`, plus the
    `ctx.chat(...)` / `ctx.fetch(...)` / `ctx.read_file(...)` /
    `ctx.emit(...)` / `ctx.emit_upstream(...)` capability methods.
    """
    # Event-specific payloads (only the relevant ones are populated):
    benchmark: str                                # bench slug
    extras: dict                                  # per-task extras
    system_prompt: str = ""                       # current base system prompt
    prompt: Optional[str] = None                  # task_prompt at pre_prompt_build / pre_context_build
    raw_response: Optional[str] = None            # LLM raw content this turn at post_llm_response / pre_answer_emit
    answer: Optional[str] = None                  # extracted answer at pre_answer_emit
    log: Any = None                               # EventLog (read-only access for handlers)

    # FC-loop state — populated only on per-turn / per-tool events:
    current_iter: int = 0                         # 1-based FC iteration count
    current_tool_name: str = ""                   # at pre_tool_use / post_tool_use / on_tool_error
    current_tool_args: Optional[dict] = None      # at pre_tool_use (mutable via REWRITE)
    current_tool_result: Optional[str] = None     # at post_tool_use / on_tool_error (mutable via REWRITE / INJECT)
    current_tool_call_id: str = ""                # for tool_message round-tripping
    current_tool_success: bool = True             # False on on_tool_error


Ctx = ComponentContext

Matcher = Callable[[ComponentContext], bool]
Handler = Callable[[ComponentContext], Decision]


# --- component ---------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Component:
    """A registered Component.

    `listens` is the dispatcher subscription key — the string name of the
    event this component fires on. Use a runtime-emitted lifecycle name
    (see the comment block at the top of this file) or a custom Tier-2/3
    name you publish elsewhere in the workflow.

    `emits` self-documents the custom event names this component raises
    via `ctx.emit(...)`; the runtime does not enforce, the field is read
    by skill / proposer tooling for event-name discovery.
    """
    name: str
    cls: ComponentClass
    listens: str
    matcher: Optional[Matcher]
    handler: Handler
    trust: Trust
    priority: int = 100
    emits: tuple[str, ...] = ()
