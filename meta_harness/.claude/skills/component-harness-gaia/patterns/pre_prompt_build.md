# Pattern: mount = `pre_prompt_build`

## When to choose this mount

You need to **modify the prompt sent to the LLM** before the LLM call — typically because the LLM cannot reach external content the task points at (a file the user attached via `extras.file_name`, a URL named in the prompt, a documented KB resource).

## Which classes admit this mount

| class             | admitted? | decisions permitted          |
|-------------------|-----------|------------------------------|
| `mechanism_layer` | yes       | inject_context, rewrite, block |
| `reactive_guard`  | no        | (no failure observed yet)    |
| `channel`         | yes       | inject_context               |
| `induced_rule`    | yes       | inject_context (advisory)    |

## Decision semantics

- `inject_context(text)` — appended to `ctx.system_prompt` (then prepended to the LLM messages as the system content).
- `rewrite(new_prompt)` — replaces `ctx.prompt` (the user content) wholesale. MECHANISM_LAYER only.
- `block(reason)` — marks `ctx.blocked=True`; the task returns `answer=None` with the reason logged. Use for `BLOCKED{reason=model_capability_gap}` cases (e.g., a vision question on a text-only model).

## Worked example: file_reader channel

```python
def _matches(ctx: ComponentContext) -> bool:
    fname = (ctx.extras or {}).get("file_name") or ""
    if not fname:
        return False
    # Channel scope: text-readable files only. Vision / binary →
    # a separate vision_blocked component handles those.
    return fname.lower().endswith((".txt", ".md", ".csv", ".json", ".py"))


def _handler(ctx: ComponentContext) -> Decision:
    fname = ctx.extras["file_name"]
    try:
        with open(fname, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError as e:
        # File missing / unreadable: pass through, no inject, no block.
        return Decision.allow()
    return Decision.inject_context(
        f"<attached_file path={fname!r}>\n{content}\n</attached_file>"
    )


COMPONENT = Component(
    name="file_reader_channel",
    cls=ComponentClass.CHANNEL,
    mount=Mount.PRE_PROMPT_BUILD,
    matcher=_matches,
    handler=_handler,
    state_scope=StateScope.NONE,
    capabilities=(Capability.READ_FILE,),
    trust=Trust(
        evidence_anchor=(
            "extras.file_name is a system field populated by bench/gaia/loader.py "
            "from the GAIA dataset row; reading it is a deterministic OS call."
        ),
        blast_radius="local",
        rollback_when="extras.file_name field renamed or 0 hits across 30 train tasks.",
        fallback="No file_name set → matcher False → prompt unchanged.",
    ),
)
```

## Common mistakes

- Class = `mechanism_layer` with `decision = rewrite` based on a regex over the prompt text ("if the question mentions a date, rewrite to add 'use current date'"). The matcher is interpretation-layer; the right class is `induced_rule` advisory inject_context, or no component at all.
- Class = `channel` with `decision = rewrite`. The matrix admits only `inject_context` for channel — rewriting the prompt wholesale is mechanism_layer's job.
- Forgetting `capabilities=(READ_FILE,)`. The runtime records it in the manifest; the durability audit cross-checks declared vs observed effects.
