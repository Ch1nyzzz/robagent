# Pattern: mount = `post_llm_response`

## When to choose this mount

The LLM has emitted an `AssistantMessage` — possibly with `tool_calls` — and you want to inspect it **before** the runtime dispatches those calls. Distinct from `pre_tool_use` because the full assistant turn is visible: the LLM's reasoning text, the choice of tool, and all arguments are in one record.

Use cases:
- **Sub-LLM verifier**: pass the assistant turn through a small isolated LLM call, get a verdict, rewrite or block.
- **Self-consistency check**: detect "I will close the account" followed by `close_bank_account` with a mismatched account id (the LLM contradicting itself within one turn).
- **Empty-turn detector**: assistant emitted neither text nor tool_calls.

## Which classes admit this mount

| class             | admitted? | decisions permitted                                       |
|-------------------|-----------|-----------------------------------------------------------|
| `mechanism_layer` | yes       | rewrite_tool_args, block, inject_context                  |
| `reactive_guard`  | yes       | rewrite_tool_args, block, inject_context                  |
| `channel`         | no        | (no task-structure trigger after generation)              |
| `induced_rule`    | no        | (no advisory slot post-generation; the LLM has already responded) |

## Decision semantics

- `rewrite_tool_args(new_args)` — if the assistant message has tool_calls, replaces the FIRST tool_call's arguments. With multiple tool_calls, gate by matcher (use `ctx.tool_call.name`).
- `block(reason)` — drops the entire `tool_calls` list; the assistant message becomes text-only.
- `inject_context(text)` — appended as a system note before the next assistant turn (the message itself still goes; the note reaches the next turn once).

## Sub-LLM verifier discussion

Sub-LLM verifiers are the highest-leverage use of `post_llm_response`. Pattern:

```python
def _matches(ctx: ComponentContext) -> bool:
    am = ctx.assistant_message
    return am is not None and bool(am.tool_calls)


def _handler(ctx: ComponentContext) -> Decision:
    am = ctx.assistant_message
    # capabilities=(LLM_CALL,) declared on the Component; handler may
    # invoke a sub-LLM. The verifier's system prompt is a *consistency check*
    # over (am.content, am.tool_calls), NOT a policy interpretation.
    verdict = ctx.shared["verify"](am.content, am.tool_calls)
    if not verdict["ok"]:
        return Decision.block(reason=f"sub_llm_verifier: {verdict['reason']}")
    return Decision.allow()
```

The `evidence_anchor` for a verifier is **the consistency-check algorithm itself**, not a policy reading: "an assistant turn whose text contradicts its own tool_call is structurally inconsistent" — holds across all domains, verifiable off-evidence.

Critical: the verifier's *system prompt* must not encode interpretation-layer rules ("block the call if reason field looks fishy"). It must encode the consistency check ("text says X, tool does Y; flag iff X and Y disagree"). The instant the verifier starts reading policy docs to decide, the component is `induced_rule` shaped as `mechanism_layer` — and `induced_rule` rejects this mount.

## Common mistakes

- Sub-LLM verifier whose system prompt is "follow these 12 rules from doc_015 and reject the call if it violates any." That's interpretation-layer mounted as mechanism_layer with block authority. The mechanism the matcher and verifier point at must be a *consistency check on the assistant turn itself*, not a policy reading.
- Blocking on the LLM's "thinking" text alone (without examining tool_calls). The block fires before the action would have happened, but the block reason is your interpretation of natural language — predictive_heuristic in disguise.
- Forgetting that `block` zeroes the entire `tool_calls` list. To selectively block one call, use `pre_tool_use` instead.
