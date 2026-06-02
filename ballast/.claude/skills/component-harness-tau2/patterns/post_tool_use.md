# Pattern: `listens="post_tool_use"`

## When to choose this event

A tool just returned. The matcher inspects `ctx.incoming_message` (a `ToolMessage`) for an observable failure or signal — `tool.failed`, an error string, a structurally surprising payload — and the handler injects a context note for the next assistant turn.

## Which classes admit this event

| class             | admitted? | decisions permitted          |
|-------------------|-----------|------------------------------|
| `mechanism_layer` | yes       | inject_context               |
| `reactive_guard`  | yes       | inject_context               |
| `induced_rule`    | no        | (no advisory slot here)      |

`post_tool_use` is the canonical event for `reactive_guard`. The class fits because the matcher tests an observed failure event, not interpretation.

## Decision semantics

`inject_context(text)` — `text` is wrapped in a `<components_post_tool_use>` system note and pushed into `state.system_messages`, reaching the next LLM call once. The note does not persist across turns.

## Worked example: reactive_guard on tool.failed

```python
def _matches(ctx: ComponentContext) -> bool:
    tm = ctx.incoming_message
    return tm is not None and getattr(tm, "error", None) is not None


def _handler(ctx: ComponentContext) -> Decision:
    tm = ctx.incoming_message
    return Decision.inject_context(
        f"<tool_failure tool={getattr(tm, 'tool_name', '?')} "
        f"error={tm.error!r}>\n"
        f"The previous tool call failed with the error above. Re-examine the "
        f"arguments before retrying; do not retry identical args.\n"
        f"</tool_failure>"
    )


COMPONENT = Component(
    name="tool_failed_observer",
    cls=ComponentClass.REACTIVE_GUARD,
    listens="post_tool_use",
    matcher=_matches,
    handler=_handler,
    trust=Trust(
        evidence_anchor=(
            "ToolMessage.error is a framework-level field populated by the "
            "env when a tool raises. Independent of any task or KB doc."
        ),
        blast_radius="local",
        rollback_when="`tool.error` field renamed or no longer populated.",
        fallback="Tool returned successfully → matcher False → no note injected.",
    ),
)
```

## Common mistakes

- Matching on the *content* of a non-error ToolMessage and rewriting follow-up tool args based on interpretation. That's `induced_rule` shaped as `mechanism_layer`; the structure is unsafe even when admitted. Keep `post_tool_use` matchers focused on framework-level failure signals.
- Injecting verbose policy text in the note. The next-turn context window is limited; keep notes structural and short.
