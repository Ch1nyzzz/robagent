# Pattern: `listens="session_start"`

## When to choose this event

You want to extend `system_prompt` with content that is **session-invariant**: the same text on every session, every task. Framework facts, structural notices. If the content depends on the user message or task structure, use `pre_context_build` instead.

## Which classes admit this event

| class             | admitted? | decisions permitted          |
|-------------------|-----------|------------------------------|
| `mechanism_layer` | yes       | inject_context               |
| `reactive_guard`  | no        | (no failure to react to)     |
| `induced_rule`    | no        | (no advisory slot at session start; the LLM has had no chance to interpret yet — induced_rule advisories belong at pre_context_build / user_prompt_submit where the task is visible) |

The matrix is asymmetric here: `induced_rule` advisories are NOT admitted at `session_start`. The rationale is that an advisory that hasn't seen the task is just a policy paraphrase; if it's worth saying, it belongs in `domain_policy` itself (which is outside our edit scope), not in a hook.

## Decision semantics

`inject_context(text)` — `text` is appended to `system_prompt` inside a `<components_prompt_injection>` block, before the LLM is initialised. Text is present on every turn of every session.

## Worked example: framework fact (mechanism_layer)

See `agent_tau2/components/discoverable_audit_channel.py` for the canonical case. Always-on matcher, framework-source-code anchor, no LLM judgement substituted.

```python
def _matches(ctx: ComponentContext) -> bool:
    return True  # always-on


def _handler(ctx: ComponentContext) -> Decision:
    return Decision.inject_context(_AUDIT_CONTEXT)


COMPONENT = Component(
    name="discoverable_audit_channel",
    cls=ComponentClass.MECHANISM_LAYER,
    listens="session_start",
    matcher=_matches,
    handler=_handler,
    trust=Trust(
        evidence_anchor=(
            "tau2's `call_discoverable_agent_tool` calls "
            "`add_to_db('agent_discoverable_tools', ...)` per "
            "tau2-bench-src/tools.py — a framework source-code fact."
        ),
        blast_radius="global",
        rollback_when="tau2 removes add_to_db from these tools.",
        fallback="Always-on; no fallback path.",
    ),
)
```

## Why this is mechanism_layer

A framework-constant notice is the same on every task — the matcher is the trivial `return True` and the injection is anchored on a framework source-code fact (a stable structure outside any evidence sim). If the inject_context payload depended on the user message ("this task's KB doc says X"), the right event would be `pre_context_build`. If the payload were "domain policy section 3.2 says X" — a reading of policy compiled from N evidence sims — the right class would be `induced_rule` and the event must shift to `pre_context_build` (the matrix rejects induced_rule at `session_start`).

If the content the agent needs is something it could reach via a tool (a KB doc body, a file body, an API result), that is a capability gap — register a tool / sub-agent, do not stuff the content into `system_prompt`.

## Common mistakes

- Putting per-session retrieval at `session_start`. The injection is computed once at agent init and frozen; the wrong content sticks for every session.
- Long policy paraphrases ("Domain policy says you must: 1) … 2) … 3) …"). The domain policy is already in `system_prompt`; restating it adds tokens without information. Inject *framework* facts (things outside the policy doc) here.
- `induced_rule` at this event. The matrix rejects it; redesign as `pre_context_build` + `inject_context` if you must surface a rule advisory.
- Using `session_start` to inject content the agent could reach via its own tools — that's a capability gap masquerading as stabilization.
