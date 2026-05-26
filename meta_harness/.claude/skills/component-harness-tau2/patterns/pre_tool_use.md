# Pattern: `listens="pre_tool_use"`

## When to choose this event

The LLM has emitted a `ToolCall` and you want to inspect, rewrite, or block it before the env sees it. Canonical event for **wrap-tool** interventions: sanitise arguments, strip optional fields, enforce schema invariants, block malformed calls.

## Which classes admit this event

| class             | admitted? | decisions permitted                                       |
|-------------------|-----------|-----------------------------------------------------------|
| `mechanism_layer` | yes       | rewrite_tool_args, defer (v1: falls back to allow), block  |
| `reactive_guard`  | yes       | block, rewrite_tool_args (after observed prior failure)   |
| `induced_rule`    | no        | (no advisory-injection slot at tool dispatch)             |

## Decision semantics

- `rewrite_tool_args(new_args: dict)` — replaces `tool_call.arguments` wholesale.
- `block(reason: str)` — drops the tool_call from the assistant message. If the LLM made multiple tool calls in one turn, only the matched one is dropped.
- `defer(replay_when)` — v1 treats as `allow` and logs. A real replay queue is v2.5+; do not design a hook whose correctness depends on real deferral.

## Wrap-tool pattern (mechanism_layer)

See `agent_tau2/components/close_account_strip_optional_reason.py`. The matcher tests the tool's declared schema:

```python
def _matches(ctx: ComponentContext) -> bool:
    return ctx.tool_name == "close_bank_account" and "reason" in ctx.tool_args


def _handler(ctx: ComponentContext) -> Decision:
    return Decision.rewrite_tool_args(
        {k: v for k, v in ctx.tool_args.items() if k != "reason"}
    )
```

This is `mechanism_layer` because the matcher tests the tool catalog (a system-level schema fact), not a policy reading.

## Reactive-guard pattern at pre_tool_use

Only appropriate when the matcher reads a prior-turn signal of failure via `ctx.history`:

```python
def _matches(ctx: ComponentContext) -> bool:
    # The previous ToolMessage was an "InvalidArgument" error on this tool.
    return _last_tool_message_was_invalid_for(ctx.history, ctx.tool_name)


def _handler(ctx: ComponentContext) -> Decision:
    return Decision.block(reason="prior call returned InvalidArgument; LLM is looping")
```

Class is `reactive_guard`, decision is `block`, matrix admits the combination.

## Common mistakes

- Class = `mechanism_layer` with `decision = block` based on policy interpretation ("the policy says this tool should not be called on weekends"). Matrix admits the cell, but the matcher is interpretation-layer — the right class is `induced_rule`, which at this event is REJECTED. Redesign: do not block; instead inject an advisory note at `pre_context_build`.
- Matching on `ctx.task_id` (the field is hidden at fire time; the matcher always returns False).
- Mutating `ctx.tool_call.arguments` in place. The handler must return a fresh `Decision`; the runtime is the only mutator.
