# Pattern: mount = `post_llm_response`

## When to choose this mount

The LLM has returned its raw content; you want to **observe or rewrite it** before answer extraction. Distinct from `pre_answer_emit` because here you have the *raw response* (possibly with reasoning text, code blocks, `<thinking>` segments), not the extracted answer.

## Which classes admit this mount

| class             | admitted? | decisions permitted                  |
|-------------------|-----------|--------------------------------------|
| `mechanism_layer` | yes       | rewrite, block, inject_context       |
| `reactive_guard`  | yes       | rewrite, block, inject_context       |
| `channel`         | no        | (no task-structure trigger post-LLM) |
| `induced_rule`    | no        | (no advisory slot here)              |

## Decision semantics

- `rewrite(new_raw)` — replaces `ctx.raw_response`. The default extractor (`ctx.answer = ctx.raw_response.strip()`) then runs on the new value.
- `block(reason)` — `ctx.blocked=True`; task returns `answer=None`.
- `inject_context(text)` — appended to `ctx.shared["post_llm_inject"]`. v1 reactive_guards that perform a recovery LLM call read this list and incorporate it.

## Worked example: finish_reason=length recovery (reactive_guard)

```python
from agent.llm import chat


def _matches(ctx: ComponentContext) -> bool:
    return (
        ctx.shared.get("finish_reason") == "length"
        and not (ctx.raw_response or "").strip()
    )


def _handler(ctx: ComponentContext) -> Decision:
    # Recovery pass with a shorter prompt + sub-LLM call.
    recovery_msgs = [
        {"role": "system",
         "content": "Output ONLY the final answer value to the question — "
                    "no reasoning, no explanation. Just the raw answer."},
        {"role": "user", "content": ctx.prompt or ""},
    ]
    try:
        recovered = chat(messages=recovery_msgs, max_tokens=128)["content"]
    except Exception:
        return Decision.allow()
    return Decision.rewrite(recovered or "")


COMPONENT = Component(
    name="finish_reason_length_recovery",
    cls=ComponentClass.REACTIVE_GUARD,
    mount=Mount.POST_LLM_RESPONSE,
    matcher=_matches,
    handler=_handler,
    state_scope=StateScope.NONE,
    capabilities=(Capability.LLM_CALL,),
    trust=Trust(
        evidence_anchor=(
            "finish_reason is an OpenAI/DeepSeek API spec field with a closed "
            "value set including 'length'; this matcher's predicate is on a "
            "structural API contract, not a task-specific guess."
        ),
        blast_radius="local",
        rollback_when="finish_reason='length' rate goes to 0 across 30 train runs (model upgraded to handle longer responses inline).",
        fallback="If recovery LLM also fails, decision.allow → original empty response passes through and the default extractor returns empty.",
    ),
)
```

## Worked example: strip `<thinking>` blocks (mechanism_layer)

```python
import re
_THINK_RE = re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE)


def _matches(ctx: ComponentContext) -> bool:
    return bool(_THINK_RE.search(ctx.raw_response or ""))


def _handler(ctx: ComponentContext) -> Decision:
    return Decision.rewrite(_THINK_RE.sub("", ctx.raw_response or "").strip())


COMPONENT = Component(
    name="strip_thinking_blocks",
    cls=ComponentClass.MECHANISM_LAYER,
    mount=Mount.POST_LLM_RESPONSE,
    matcher=_matches,
    handler=_handler,
    trust=Trust(
        evidence_anchor=(
            "<thinking>...</thinking> is a literal XML-style tag pattern emitted "
            "by some reasoning-mode LLMs; the rewrite is a regex over a structural "
            "marker, not an interpretation of content."
        ),
        blast_radius="local",
        rollback_when="model stops emitting thinking tags (matcher → 0 fires).",
        fallback="No thinking tags → matcher False → raw_response passes through.",
    ),
)
```

## Common mistakes

- Class = `mechanism_layer` `rewrite` to "extract the answer" — that's `pre_answer_emit`'s job, not post_llm_response's. Use post_llm_response when the rewrite changes the entire raw response shape (recovery, thinking-strip); use pre_answer_emit when extracting the final answer from a wrapper.
- Forgetting that `block` here zeroes the entire response. To produce a different answer instead, use `rewrite`.
- Reactive_guard with `inject_context` but no downstream consumer. v1 inject_context at post_llm_response is consumed by *other* post_llm_response components; if no one reads it, the injection is dead weight.
