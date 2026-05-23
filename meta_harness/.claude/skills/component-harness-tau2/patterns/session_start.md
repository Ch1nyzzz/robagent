# Pattern: mount = `session_start`

## When to choose this mount

You want to extend `system_prompt` with content that is **session-invariant**: the same text on every session, every task. Framework facts, static channel material, structural notices. If the content depends on the user message or task structure, use `pre_context_build` instead.

## Which classes admit this mount

| class             | admitted? | decisions permitted          |
|-------------------|-----------|------------------------------|
| `mechanism_layer` | yes       | inject_context               |
| `reactive_guard`  | no        | (no failure to react to)     |
| `channel`         | yes       | inject_context               |
| `induced_rule`    | no        | (no advisory slot at session start; the LLM has had no chance to interpret yet — induced_rule advisories belong at pre_context_build / user_prompt_submit where the task is visible) |

The matrix is asymmetric here: `induced_rule` advisories are NOT admitted at `session_start`. The rationale is that an advisory that hasn't seen the task is just a policy paraphrase; if it's worth saying, it belongs in `domain_policy` itself (which is outside our edit scope), not in a component.

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
    mount=Mount.SESSION_START,
    matcher=_matches,
    handler=_handler,
    state_scope=StateScope.NONE,
    capabilities=(Capability.NONE,),
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

## Why this is mechanism_layer and not channel

`channel` is for content the agent cannot otherwise reach for *this particular task* (a file the user attached, a URL named in the prompt). A framework-constant notice is the same on every task — the right class is `mechanism_layer` (protocol-invariant injection) and the matcher is the trivial `return True`. If the inject_context payload were "this task's KB doc says X" (different per task), the class would be `channel` and the mount would be `pre_context_build`.

## Common mistakes

- Putting per-session retrieval at `session_start`. The injection is computed once at agent init and frozen; the wrong content sticks for every session.
- Long policy paraphrases ("Domain policy says you must: 1) … 2) … 3) …"). The domain policy is already in `system_prompt`; restating it adds tokens without information. Inject *framework* facts (things outside the policy doc) here.
- `induced_rule` at this mount. The matrix rejects it; redesign as `pre_context_build` + `inject_context` if you must surface a rule advisory.
