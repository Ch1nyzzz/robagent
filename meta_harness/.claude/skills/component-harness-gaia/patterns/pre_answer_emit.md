# Pattern: mount = `pre_answer_emit`

## When to choose this mount

The LLM's raw response has been extracted into `ctx.answer` (default: `raw_response.strip()`). You want to **normalise or canonicalise** the final answer before return — strip prefixes, extract from a `FINAL ANSWER:` line, normalise numbers, lowercase, etc.

## Which classes admit this mount

| class             | admitted? | decisions permitted     |
|-------------------|-----------|-------------------------|
| `mechanism_layer` | yes       | rewrite, block          |
| `reactive_guard`  | yes       | rewrite, block          |
| `channel`         | no        | (no content to inject)  |
| `induced_rule`    | no        | (no advisory slot here) |

## Decision semantics

- `rewrite(new_answer)` — replaces `ctx.answer`. If `new_answer` is `None`, the runtime treats it as blocked.
- `block(reason)` — marks blocked; task returns `answer=None`.

## Worked example: extract FINAL ANSWER line

```python
import re

_FINAL_RE = re.compile(r"^FINAL ANSWER:\s*(.+)$", re.MULTILINE | re.IGNORECASE)


def _matches(ctx: ComponentContext) -> bool:
    return bool(_FINAL_RE.search(ctx.raw_response or ""))


def _handler(ctx: ComponentContext) -> Decision:
    m = list(_FINAL_RE.finditer(ctx.raw_response or ""))
    # Take the LAST match (most LLMs put it at the end after reasoning).
    return Decision.rewrite(m[-1].group(1).strip())


COMPONENT = Component(
    name="final_answer_extractor",
    cls=ComponentClass.MECHANISM_LAYER,
    mount=Mount.PRE_ANSWER_EMIT,
    matcher=_matches,
    handler=_handler,
    trust=Trust(
        evidence_anchor=(
            "FINAL ANSWER: <value> is a literal regex-extractable anchor; the "
            "extraction is over a textual marker pattern, not over content."
        ),
        blast_radius="local",
        rollback_when="Paired session_start directive disabled → no LLM emits the marker → matcher 0 fires.",
        fallback="No marker → matcher False → default-stripped answer passes through.",
    ),
)
```

## Worked example: detect "I cannot answer" → block (reactive_guard)

```python
_REFUSAL_PATTERNS = (
    "i cannot answer",
    "i don't know",
    "unable to determine",
    "not enough information",
)


def _matches(ctx: ComponentContext) -> bool:
    low = (ctx.answer or "").lower().strip()
    return any(p in low for p in _REFUSAL_PATTERNS)


def _handler(ctx: ComponentContext) -> Decision:
    return Decision.block(reason=f"refusal_detected: {ctx.answer!r}")


COMPONENT = Component(
    name="refusal_to_blocked",
    cls=ComponentClass.REACTIVE_GUARD,
    mount=Mount.PRE_ANSWER_EMIT,
    matcher=_matches,
    handler=_handler,
    trust=Trust(
        evidence_anchor=(
            "GAIA scoring is exact-match; emitting 'I cannot answer' as the "
            "answer always scores 0. Marking blocked is honest-over-fabricated."
        ),
        blast_radius="local",
        rollback_when="Model upgrades to handle these queries (refusal rate → 0 in fired.jsonl).",
        fallback="No refusal phrase → matcher False → answer passes through.",
    ),
)
```

## Common mistakes

- Number normalisation with cultural assumptions (e.g., comma-as-decimal). The matcher should test the FORMAT (digit + comma + 3-digit groups), not guess what the question wanted.
- `block` based on answer length / "looks weird". Predictive_heuristic in disguise. Use only on structural failure signals.
- Chaining many small `pre_answer_emit` rewrites in fixed priority order without considering interactions. Each fires unconditionally; later ones see the earlier one's output. Add `state_scope=SESSION` if a component needs to know whether an earlier one already rewrote.
