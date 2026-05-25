"""Shared EventContext base (Phase B of event-runtime migration).

EventContext holds the invariant fields every event carries (task_id,
shared / state / persistent_state / upstream scratchpads, blocked flag,
capability method handles). Sibling ComponentContext subclasses extend
this with mount-specific payload fields (e.g. sopbench's
`current_tool_name`, gaia's `raw_response`).

Capability methods (`chat`, `fetch`, `read_file`, `emit`, `emit_upstream`)
are stubs by default — Phase C wires per-sibling implementations via the
`_impl_*` callable hooks set when the dispatcher constructs the context.
A handler that calls a capability the sibling does not support gets a
clear RuntimeError instead of an opaque AttributeError.

`ctx.chat()` does NOT take a `model` kwarg. The implementation goes
through `agent.llm.chat()` (or the per-bench equivalent) which already
enforces the locked SUT model name at call time. Components can vary
max_tokens / temperature / system_override / tools, but not the model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class EventContext:
    """Base context shared by every event the runtime emits.

    Subclassed by each sibling's `ComponentContext` (e.g.
    `agent.component_runtime.types.ComponentContext`) which adds
    mount-specific fields like `proposed_system_prompt`, `raw_response`,
    `current_tool_name`, etc.
    """

    # ------------------------------------------------------------------
    # Event identity
    # ------------------------------------------------------------------
    event: str = ""               # event name (Tier-1 lifecycle string, custom, or `mount.value`)
    task_id: str = ""             # opaque; SKILL forbids matching on it

    # ------------------------------------------------------------------
    # Cross-component scratchpads (per task unless noted)
    # ------------------------------------------------------------------
    # Within-task shared dict; written/read by any component, any event.
    shared: dict = field(default_factory=dict)
    # Per-component scratchpad: state[component_name] persists for the task.
    state: dict = field(default_factory=dict)
    # Cross-session persisted state. Reserved; load/save not enforced in v1
    # (the runtime supplies an empty dict per task).
    persistent_state: dict = field(default_factory=dict)
    # Component→component dataflow within one task. Component A calls
    # `ctx.emit_upstream(key, value)` from its handler; downstream
    # components listening at a later event read `ctx.upstream[key]`.
    upstream: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Block / terminate signals (used by some sibling Decision kinds)
    # ------------------------------------------------------------------
    blocked: bool = False
    blocked_reason: str = ""

    # ------------------------------------------------------------------
    # Capability hooks — runtime injects per-bench implementations at
    # dispatch time. None = sibling does not expose this capability.
    # ------------------------------------------------------------------
    _impl_chat: Optional[Callable[..., Any]] = field(default=None, repr=False)
    _impl_fetch: Optional[Callable[[str], str]] = field(default=None, repr=False)
    _impl_read_file: Optional[Callable[[Any], str]] = field(default=None, repr=False)
    _impl_emit: Optional[Callable[[str, dict], None]] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Public capability methods
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        system_override: Optional[str] = None,
        tools: Optional[list] = None,
    ) -> Any:
        """Sub-LLM helper bound to the LOCKED SUT model name.

        Does NOT accept a `model` kwarg. The model name is fixed by the
        runtime; only inference params and a system-prompt override are
        mutable. Returns whatever the sibling's chat helper returns —
        gaia/sopbench/enterpriseops yield dicts shaped like the openai
        chat completion response, tau2/toolathlon return string content;
        check the sibling's `agent.llm.chat` for the precise shape.
        """
        if self._impl_chat is None:
            raise RuntimeError(
                "ctx.chat is unavailable for this sibling/event "
                "(LLM_CALL capability not wired). If your component declares "
                "Capability.LLM_CALL, make sure the dispatcher was constructed "
                "with `chat=...` for this event."
            )
        return self._impl_chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            system_override=system_override,
            tools=tools,
        )

    def fetch(self, url: str) -> str:
        if self._impl_fetch is None:
            raise RuntimeError("ctx.fetch unavailable (HTTP_GET capability not wired).")
        return self._impl_fetch(url)

    def read_file(self, path: str | Path) -> str:
        if self._impl_read_file is None:
            raise RuntimeError("ctx.read_file unavailable (READ_FILE capability not wired).")
        return self._impl_read_file(path)

    def emit(self, custom_event_name: str, **fields: Any) -> None:
        """Emit a custom (Tier 2/3) event from inside a component handler.

        Convention: name as `iter<N>_<slug>_<event>` for evolution-tier
        events; `on_<thing>` for failure-mode events. The runtime does not
        enforce a namespace, but the skill text recommends declaring
        `emits=[...]` on the component for self-documentation.
        """
        if self._impl_emit is None:
            raise RuntimeError(
                "ctx.emit unavailable (Dispatcher not wired into this context). "
                "Custom event emission requires the dispatcher to be passed in "
                "when the per-sibling runtime constructs the EventContext."
            )
        self._impl_emit(custom_event_name, fields)

    def emit_upstream(self, key: str, value: Any) -> None:
        """Convenience: write to ctx.upstream so downstream listeners
        (later events in the same task) can read via ctx.upstream[key].
        Equivalent to `ctx.upstream[key] = value`."""
        self.upstream[key] = value
