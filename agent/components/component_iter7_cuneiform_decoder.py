from __future__ import annotations

import unicodedata

from agent.component_runtime.types import (
    Component, ComponentClass, ComponentContext,
    Decision, Trust,
)

_CUNEIFORM_NUMERIC_START = 0x12400
_CUNEIFORM_NUMERIC_END = 0x1247F


def _matches(ctx: ComponentContext) -> bool:
    prompt = ctx.prompt or ""
    return any(
        _CUNEIFORM_NUMERIC_START <= ord(ch) <= _CUNEIFORM_NUMERIC_END
        for ch in prompt
    )


def _handler(ctx: ComponentContext) -> Decision:
    prompt = ctx.prompt or ""
    seen: dict[str, None] = {}
    for ch in prompt:
        cp = ord(ch)
        if _CUNEIFORM_NUMERIC_START <= cp <= _CUNEIFORM_NUMERIC_END:
            seen[ch] = None

    if not seen:
        return Decision.allow()

    lines = []
    for ch in seen:
        cp = ord(ch)
        name = unicodedata.name(ch, f"UNKNOWN (U+{cp:04X})")
        lines.append(f"  U+{cp:04X}  '{ch}'  ->  {name}")

    context = (
        "Unicode character reference for the cuneiform numeric symbols appearing in"
        " this task (block U+12400-U+1247F, Cuneiform Numbers and Punctuation):\n"
        + "\n".join(lines)
        + "\n\n"
        "Naming conventions: 'ONE ASH' = 1, 'TWO ASH' = 2, …, 'NINE ASH' = 9. "
        "'ONE TEN' = 10, 'TWO TEN' = 20, 'THREE TEN' = 30, 'FOUR TEN' = 40, "
        "'FIVE TEN' = 50. "
        "In the Babylonian sexagesimal (base-60) system, groups of signs separated "
        "by whitespace represent successive powers of 60 (rightmost group = units, "
        "next group left = 60s, next = 3600s, etc.)."
    )
    return Decision.inject_context(context)


COMPONENT = Component(
    name="cuneiform_numeric_decoder",
    cls=ComponentClass.MECHANISM_LAYER,
    listens="pre_prompt_build",
    matcher=_matches,
    handler=_handler,
    priority=90,
    trust=Trust(
        evidence_anchor=(
            "Python standard library `unicodedata.name()` and the Unicode Standard "
            "Cuneiform Numbers and Punctuation block (U+12400-U+1247F) — a stable "
            "ISO/IEC 10646 code-point block with fixed, published name assignments "
            "for each cuneiform numeric sign, documented outside the evidence traces"
        ),
        blast_radius="local",
        rollback_when=(
            "injected character names are absent or misleading (unicodedata module "
            "missing names for characters in the block) such that train-30 accuracy "
            "decreases on tasks not containing cuneiform numeric characters"
        ),
        out_of_evidence_probe=(
            "A task with no characters in U+12400-U+1247F (e.g., a task mentioning "
            "ancient writing systems but containing only ASCII text): matcher returns "
            "False, handler never fires, prompt unchanged. "
            "A task with regular cuneiform script (U+12000-U+123FF syllabic signs, "
            "outside the numeric block): matcher returns False."
        ),
        fallback=(
            "matcher returns False for prompts with no cuneiform numeric code points; "
            "handler returns allow() when seen dict is empty after scanning"
        ),
    ),
)
