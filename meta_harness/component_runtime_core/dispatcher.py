"""Shared synchronous event dispatcher (Phase B of event-runtime migration).

The dispatcher is generic over per-sibling Component / Decision / DecisionKind
types via protocol-like duck typing (we do not import sibling types here).
Components are indexed by their string `listens` field — the Phase B
addition to the Component dataclass. The mount→event alias is handled in
each sibling's Component.__post_init__ (defaults `listens = mount.value`
when listens=""); the dispatcher itself only sees strings.

Per-sibling integration

  Each sibling constructs ONE Dispatcher per task with:

      Dispatcher(
          components,                   # iterable of the sibling's Component objects
          validate_decision=fn(comp, event_name, kind) -> None,
          apply_decision=fn(ctx, decision, comp) -> bool,   # True = stop
          trace_sink=fn(name, event, kind, extra) -> None | None,
      )

  The sibling's existing dispatch sites (e.g. gaia's `_dispatch(mount, ctx,
  by_mount)`, tau2's `_dispatch_post_llm_response`, sopbench's
  `Dispatcher.dispatch`) become one-line wrappers around
  `self._dispatcher.emit(mount.value, ctx)`.

`emit` is synchronous: it blocks until all subscribers + apply hooks
return, so component handlers see decisions resolve before the runtime
continues. Components may call `ctx.emit(custom_event, ...)` from within
a handler — this re-enters `emit` synchronously; the dispatcher caps
recursion depth at 10 to break runaway loops.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Callable, Iterable


class Dispatcher:
    """Sync event dispatch over a per-task component set."""

    def __init__(
        self,
        components: Iterable[Any],
        *,
        validate_decision: Callable[[Any, str, Any], None],
        apply_decision: Callable[[Any, Any, Any], bool],
        trace_sink: Callable[[str, str, str, dict], None] | None = None,
        max_recursion_depth: int = 10,
    ):
        """
        Parameters
        ----------
        components : iterable of sibling Component objects. Each must expose
            `.name : str`, `.priority : int`, `.matcher : Callable | None`,
            `.handler : Callable`, `.listens : str`. (Sibling Component
            dataclass adds `listens` via __post_init__ defaulting to
            `mount.value` for backward compat — see Phase B note in
            shared_types.py.)
        validate_decision : called as `(comp, event_name, decision.kind)`
            BEFORE apply. Should raise ComponentPolicyError on a non-admitted
            decision. The sibling supplies this so the per-sibling ALLOWED
            matrix stays the source of truth.
        apply_decision : called as `(ctx, decision, comp)`. Mutates ctx in
            place; returns True to stop firing remaining components at this
            event (BLOCK / BLOCK_TERMINATION semantics).
        trace_sink : optional. Called after each fire as
            `(comp.name, event_name, decision.kind.value, {})`. Used to
            populate `.component-state/<run_tag>/fired.jsonl`.
        max_recursion_depth : guard against runaway ctx.emit() loops.
        """
        self._by_event: dict[str, list[Any]] = defaultdict(list)
        for c in components:
            ev = self._event_key(c)
            self._by_event[ev].append(c)
        for ev in self._by_event:
            self._by_event[ev].sort(key=lambda c: c.priority)
        self._validate = validate_decision
        self._apply = apply_decision
        self._trace = trace_sink
        # Per-thread emit-depth counter. The Dispatcher is process-singleton
        # (cached at module level in each sibling's base.py) so the previous
        # `self._depth = 0` was SHARED across worker threads at parallel>1
        # — N concurrent emits at the same event tripped the recursion cap
        # without any real recursion. threading.local() keeps each worker's
        # depth isolated.
        self._depth_state = threading.local()
        self._max_depth = max_recursion_depth

    @staticmethod
    def _event_key(c: Any) -> str:
        """A Component's subscription key. Phase B adds `listens: str` as
        first-class; the field is populated via __post_init__ from
        `mount.value` when the caller did not pass it explicitly. We still
        guard against legacy callers by falling back to `mount.value`."""
        listens = getattr(c, "listens", "") or ""
        if listens:
            return listens
        mount = getattr(c, "mount", None)
        return getattr(mount, "value", str(mount)) if mount is not None else ""

    def subscribers(self, event_name: str) -> list[Any]:
        """Expose the per-event list for skill / debug introspection."""
        return list(self._by_event.get(event_name, ()))

    def known_events(self) -> list[str]:
        """All event names that have at least one subscriber."""
        return sorted(self._by_event.keys())

    def emit(self, event_name: str, ctx: Any) -> None:
        depth = getattr(self._depth_state, "depth", 0)
        if depth >= self._max_depth:
            raise RuntimeError(
                f"ctx.emit recursion depth {depth} exceeded "
                f"max_recursion_depth={self._max_depth} at event "
                f"{event_name!r}; check for emit cycles among components."
            )
        self._depth_state.depth = depth + 1
        # Tag the context with the firing event name so handlers / apply hooks
        # can route on `ctx.event` without the caller threading it. Restore on
        # exit so an in-handler `ctx.emit(custom)` does not leak back to the
        # outer caller's notion of "current event".
        prev_event = getattr(ctx, "event", None) if ctx is not None else None
        if ctx is not None:
            try:
                setattr(ctx, "event", event_name)
            except (AttributeError, TypeError):
                pass
        try:
            for comp in self._by_event.get(event_name, ()):
                matcher = getattr(comp, "matcher", None)
                if matcher is not None and not matcher(ctx):
                    continue
                decision = comp.handler(ctx)
                try:
                    self._validate(comp, event_name, decision.kind)
                except Exception as e:
                    if self._trace is not None:
                        self._trace(
                            comp.name, event_name, "policy_error",
                            {"error": repr(e)},
                        )
                    # Hard policy error: skip this component, continue others.
                    continue
                stop = bool(self._apply(ctx, decision, comp))
                if self._trace is not None:
                    kind_value = getattr(decision.kind, "value", str(decision.kind))
                    self._trace(comp.name, event_name, kind_value, {})
                if stop:
                    break
        finally:
            if ctx is not None:
                try:
                    setattr(ctx, "event", prev_event if prev_event is not None else "")
                except (AttributeError, TypeError):
                    pass
            self._depth_state.depth = getattr(self._depth_state, "depth", 1) - 1
