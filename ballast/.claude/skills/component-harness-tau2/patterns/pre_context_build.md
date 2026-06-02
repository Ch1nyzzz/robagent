# Pattern: `listens="pre_context_build"`

## When to choose this event

You need to inject content into `system_prompt` whose **value depends on the session** (a per-task framework fact, a fresh look at `ctx.domain_policy`, an advisory note paraphrased from policy). `session_start` doesn't work — its handler runs *after* the LLM's system_prompt is initially frozen at session boundaries with no per-session conditioning.

`pre_context_build` fires earlier: the handler receives `ctx.domain_policy`, `ctx.tool_names`, and the raw incoming user message. The returned `inject_context` payload is folded into `system_prompt` itself for that session.

**Capability-vs-stabilization check.** If you find yourself reaching for `pre_context_build` to inject content the agent could reach via a tool (a KB doc body, a file body, an API result), that is a capability gap — register a tool / sub-agent, do not stuff the content into `system_prompt`. `pre_context_build` is for *framework / advisory* injection, not for substituting tool reach.

## Which classes admit this event

| class             | admitted? | decision permitted          |
|-------------------|-----------|-----------------------------|
| `mechanism_layer` | yes       | inject_context              |
| `reactive_guard`  | no        | (nothing to react to yet)   |
| `induced_rule`    | yes       | inject_context (advisory)   |

## Decision semantics

`inject_context(text)` — `text` is appended to `system_prompt` inside a `<components_prompt_injection>` tagged block, before the LLM is initialised. Text is present on every turn of that session.

## Worked example: advisory policy reminder (induced_rule)

```python
def _matches(ctx: ComponentContext) -> bool:
    # Fire only when the user message structure suggests a closure flow is
    # imminent; advisory note steers reading order, not the closure decision.
    msg = ctx.incoming_message
    if msg is None:
        return False
    text = (getattr(msg, "content", "") or "").lower()
    return "close" in text and "account" in text


def _handler(ctx: ComponentContext) -> Decision:
    return Decision.inject_context(
        "<advisory>\n"
        "Account closure has documented prerequisites in the domain policy "
        "(audit visibility, outstanding-balance checks). Consult the policy "
        "before initiating any close_* tool call.\n"
        "</advisory>"
    )


COMPONENT = Component(
    name="closure_prereq_advisory",
    cls=ComponentClass.INDUCED_RULE,
    listens="pre_context_build",
    matcher=_matches,
    handler=_handler,
    trust=Trust(
        evidence_anchor=(
            "Closure-prereq language paraphrases the policy doc's section on "
            "account-closure preconditions; the advisory does not override the "
            "LLM, it merely names the section."
        ),
        blast_radius="local",
        rollback_when=(
            "Policy doc removes the closure-prereqs section OR LLM closure "
            "success rate already at ceiling (advisory delta → 0)."
        ),
        out_of_evidence_probe=(
            "On a sim whose user message says 'cancel my checking account' "
            "(not in evidence sims), the matcher fires, the advisory is "
            "injected, the LLM still decides freely whether to close — no "
            "override, no risk."
        ),
        fallback="No 'close'+'account' tokens → matcher False → prompt unchanged.",
    ),
)
```

## Common mistakes

- Using `pre_context_build` with `decision=rewrite_tool_args`. The matrix rejects it — no tool call exists at this event.
- Using `pre_context_build` with class `induced_rule` and a long compiled rule set as the payload. Even when admitted, you are smuggling interpretation-layer code via the injection. Keep advisory injections short and structural ("doc_021 governs closures; read it before proceeding") rather than compiled IF/THEN.
- Using `pre_context_build` to inject content the agent should reach via a tool (KB doc bodies, API results, file contents). That's a capability gap — register a tool, do not write a hook.
