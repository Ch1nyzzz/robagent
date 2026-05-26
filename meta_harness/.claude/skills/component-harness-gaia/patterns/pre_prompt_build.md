# Pattern: event = `pre_prompt_build` (alias `pre_context_build`)

## When to choose this event

You need to **modify the initial task prompt** before the FC loop begins — typically to add a framework reminder ("output in FINAL ANSWER: format"), or to advisory-inject a policy reading the model would benefit from seeing upfront.

**Not for** giving the agent content it can't otherwise reach — that's a capability gap. The baseline FC loop already has `file_read` / `url_fetch` / `web_search` / `python_exec`. If the agent can call those, don't inject the content here.

## Which classes admit this event

| class             | admitted? | decisions permitted          |
|-------------------|-----------|------------------------------|
| `mechanism_layer` | yes       | inject_context, rewrite, block |
| `reactive_guard`  | no        | (no failure observed yet)    |
| `induced_rule`    | yes       | inject_context (advisory)    |

## Decision semantics

- `inject_context(text)` — appended to `ctx.system_prompt`, which becomes the system message in the first FC turn.
- `rewrite(new_prompt)` — replaces `ctx.prompt` (the user content) wholesale. `mechanism_layer` only.
- `block(reason)` — marks `ctx.blocked=True`; the task returns `answer=None` with the reason logged. Use for `BLOCKED{reason=model_capability_gap}` (e.g., the task is fundamentally outside the agent's tool reach AND adding a tool isn't on the table).

## Worked example: framework format reminder

```python
def _matches(ctx: ComponentContext) -> bool:
    # Fire on every task — this is a framework-invariant injection,
    # not a task-specific rule.
    return True


def _handler(ctx: ComponentContext) -> Decision:
    return Decision.inject_context(
        "Reminder: end your response with one line\n"
        "  FINAL ANSWER: <answer>\n"
        "where <answer> is exactly the value the question asks for "
        "(numbers as digits, no units; strings with no prefix; "
        "lists as comma-separated values)."
    )


COMPONENT = Component(
    name="final_answer_format_reminder",
    cls=ComponentClass.MECHANISM_LAYER,
    listens="pre_prompt_build",
    matcher=_matches,
    handler=_handler,
    trust=Trust(
        evidence_anchor=(
            "GAIA scorer.question_scorer normalises numeric/string answers "
            "against a specific shape; the FINAL ANSWER: line convention "
            "is documented in agent/base.py::SYSTEM_PROMPT and is independent "
            "of any individual evidence trace."
        ),
        blast_radius="local",
        rollback_when="train-30 accuracy drops vs frontier on tasks that "
                      "previously emitted a correct FINAL ANSWER without the reminder.",
        fallback="The base agent.base.SYSTEM_PROMPT already mentions FINAL ANSWER; "
                 "the reminder is redundant if the model never strays.",
    ),
)
```

## Common mistakes

- **Treating a capability gap as a hook problem.** "The agent should know about the attached file" — write a tool, not a hook. If the file format is unsupported by `file_read`, extend `agent/tools/file_read.py` (separate workflow, not this skill).
- Class = `mechanism_layer` with `decision = rewrite` based on a regex over the prompt text ("if the question mentions a date, rewrite to add 'use current date'"). The matcher is interpretation-layer; the right class is `induced_rule` advisory `inject_context`, or no component at all.
