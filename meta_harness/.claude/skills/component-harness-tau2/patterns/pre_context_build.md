# Pattern: mount = `pre_context_build`

## When to choose this mount

You need to inject content into `system_prompt` whose **value depends on the session** (the current user's KB doc, a fresh retrieval, a domain-policy slice gated by the task structure). `session_start` doesn't work — its handler runs *after* `system_prompt` is frozen by `get_init_state`, so it can only append at session boundaries with no per-session conditioning.

`pre_context_build` fires earlier: the handler receives `ctx.domain_policy`, `ctx.tool_names`, and (for retrieval components) the raw user message. The returned `inject_context` payload is folded into `system_prompt` itself.

## Which classes admit this mount

| class             | admitted? | decision permitted          |
|-------------------|-----------|-----------------------------|
| `mechanism_layer` | yes       | inject_context              |
| `reactive_guard`  | no        | (nothing to react to yet)   |
| `channel`         | yes       | inject_context              |
| `induced_rule`    | yes       | inject_context (advisory)   |

## Decision semantics

`inject_context(text)` — `text` is appended to `system_prompt` inside a `<components_prompt_injection>` tagged block, before the LLM is initialised. Text is present on every turn of that session.

## Worked example: dynamic KB retrieval channel

```python
def _matches(ctx: ComponentContext) -> bool:
    # Fire only on sessions whose first user message mentions a KB doc id.
    msg = ctx.incoming_message
    if msg is None:
        return False
    return bool(_DOC_PATTERN.search(getattr(msg, "content", "") or ""))


def _handler(ctx: ComponentContext) -> Decision:
    doc_id = _DOC_PATTERN.search(ctx.incoming_message.content).group(1)
    # capabilities=(TOOL_CALL,) declared on the Component
    doc = ctx.shared["retriever"](doc_id)
    return Decision.inject_context(f"<kb_doc id={doc_id}>\n{doc}\n</kb_doc>")


COMPONENT = Component(
    name="kb_doc_retrieval_channel",
    cls=ComponentClass.CHANNEL,
    mount=Mount.PRE_CONTEXT_BUILD,
    matcher=_matches,
    handler=_handler,
    state_scope=StateScope.SESSION,
    capabilities=(Capability.TOOL_CALL,),
    trust=Trust(
        evidence_anchor=(
            "The KB tool's `read_kb_doc(doc_id)` schema returns the doc text "
            "verbatim — a tool-declared schema fact, not a policy reading."
        ),
        blast_radius="local",
        rollback_when=(
            "Matcher fires but tool returns error on >50% of sessions "
            "(KB schema changed)."
        ),
        out_of_evidence_probe=(
            "On a session whose user message mentions doc_999 (not in any "
            "evidence sim), the matcher fires, the tool returns the doc, and "
            "the LLM reads it. If doc_999 does not exist, the tool returns an "
            "error string and the LLM sees the error — no override, no risk."
        ),
        fallback="No doc id in the user message → matcher False → prompt unchanged.",
    ),
)
```

## Common mistakes

- Using `pre_context_build` with `decision=rewrite_tool_args`. The matrix rejects it — no tool call exists at this mount.
- Using `pre_context_build` with class `induced_rule` and a long compiled rule set as the payload. Even though the matrix admits this, you are smuggling interpretation-layer code via the injection. Keep advisory injections short and structural ("doc_021 governs closures; read it before proceeding") rather than compiled IF/THEN.
- Forgetting `state_scope=SESSION` when the handler caches retrieved docs. Without it, the retrieval re-fires every time the system_prompt is rebuilt.
