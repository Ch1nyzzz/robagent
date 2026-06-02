# Pattern: mount = `session_start`

## When to choose this mount

You want to extend `system_prompt` with content that is **session-invariant** — the same text for every task. Format directives, framework reminders, structural notices. If the content depends on `extras` or the user prompt, use `pre_prompt_build` instead.

## Which classes admit this mount

| class             | admitted? | decisions permitted          |
|-------------------|-----------|------------------------------|
| `mechanism_layer` | yes       | inject_context               |
| `reactive_guard`  | no        | (no failure to react to)     |
| `induced_rule`    | no        | (no advisory slot before the task is visible) |

## Decision semantics

`inject_context(text)` — text is appended to `ctx.system_prompt` once at session start. Present in every LLM call for this task.

## Worked example: switch to FINAL ANSWER format

```python
_FORMAT_DIRECTIVE = (
    "Output the final answer on its own line in this exact format:\n"
    "FINAL ANSWER: <value>\n"
    "where <value> is the answer only, with no extra punctuation."
)


def _matches(ctx: ComponentContext) -> bool:
    return True  # always-on


def _handler(ctx: ComponentContext) -> Decision:
    return Decision.inject_context(_FORMAT_DIRECTIVE)


COMPONENT = Component(
    name="final_answer_format_directive",
    cls=ComponentClass.MECHANISM_LAYER,
    listens="session_start",
    matcher=_matches,
    handler=_handler,
    priority=100,
    trust=Trust(
        evidence_anchor=(
            "The FINAL ANSWER: <value> format is a regex-extractable canonical "
            "anchor; a paired pre_answer_emit mechanism_layer extractor will "
            "parse it. This is structural, not interpretive."
        ),
        blast_radius="global",
        rollback_when=(
            "Paired extractor disabled OR LLM stops respecting the format on "
            ">50% of tasks (observable in fired.jsonl: rewrite count drops to 0)."
        ),
        fallback="Always-on; if LLM ignores the directive, the extractor falls through and the raw response is returned as-is.",
    ),
)
```

## Common mistakes

- Inject long policy paraphrases. Most GAIA tasks don't need them; they add tokens without information.
- `induced_rule` at this mount. The matrix rejects it — induced rules need to see the task to be relevant; use `pre_prompt_build` advisory.
- Using session_start to inject content the agent could reach via its own tools (file_read / url_fetch / web_search) — that's a capability gap masquerading as stabilization. Either extend the tool or skip the hook.
