"""dangerous_goods iter1: recover the final XML when the assistant's final
turn drops the required <hazard_class> tag.

Failure mode (3/10 train failures, tasks 8/10/27 on v0):
  The locked SUT sometimes terminates its final assistant turn with a
  hallucinated Qwen-style pseudo tool-call block
  ('<｜DSML｜tool_calls><｜DSML｜invoke name="finish">...') instead of a clean
  OpenAI tool_call or a plain `<hazard_class>LABEL</hazard_class>` answer.
  The agent stores that prose as ctx.final_output. OutputParser then:
    - skips XML/JSON/dict parsers (no parseable structure),
    - falls into _parse_plain_format,
    - sees len > 100,
    - returns decision='unknown'  (parser.py:429-431)
  'unknown' is a null token in compare_decisions (parser.py:482), so it
  never matches the expected 'Hazard Class A/B/C/D' / 'Unable to Decide'
  even when the model's prose conclusively states the right label.
"""
from __future__ import annotations

import re

from agent.component_runtime_sopbench import (
    Component,
    ComponentClass,
    ComponentContext,
    Decision,
    Trust,
)

# SOP section 6.1 enumerates the *only* admissible output values verbatim:
#   "Hazard Class A", "Hazard Class B", "Hazard Class C", "Hazard Class D",
#   or "Unable to Decide".
_SOP_LABELS_CANONICAL = (
    "Hazard Class A",
    "Hazard Class B",
    "Hazard Class C",
    "Hazard Class D",
    "Unable to Decide",
)
_LABELS_BY_LOWER = {s.lower(): s for s in _SOP_LABELS_CANONICAL}

_LABEL_RX = re.compile(
    r"(?i)\b(?:hazard\s+class\s+[ABCD]|unable\s+to\s+decide)\b"
)
_HAZARD_CLASS_TAG_RX = re.compile(
    r"<\s*hazard_class\s*>\s*([^<]+?)\s*</\s*hazard_class\s*>",
    re.IGNORECASE,
)


def _canonicalise_label(token: str) -> str | None:
    collapsed = re.sub(r"\s+", " ", token.strip().lower())
    return _LABELS_BY_LOWER.get(collapsed)


def _has_clean_hazard_class_tag(text: str) -> bool:
    for inner in _HAZARD_CLASS_TAG_RX.findall(text):
        if _canonicalise_label(inner):
            return True
    return False


def _matches(ctx: ComponentContext) -> bool:
    text = ctx.final_output or ""
    if not text:
        return False
    # Already has a parseable <hazard_class>LABEL</hazard_class> tag.
    if _has_clean_hazard_class_tag(text):
        return False
    # Short plain answers like "hazard class c" are handled by the parser's
    # _parse_plain_format path (case-insensitive equality in
    # compare_decisions). Don't disturb them.
    plain = text.strip()
    if len(plain) <= 100 and _canonicalise_label(plain):
        return False
    # Fire only when at least one SOP-declared label appears in the prose.
    return _LABEL_RX.search(text) is not None


def _handler(ctx: ComponentContext) -> Decision:
    text = ctx.final_output or ""
    matches = list(_LABEL_RX.finditer(text))
    if not matches:
        return Decision.allow()
    canonical = _canonicalise_label(matches[-1].group(0))
    if canonical is None:
        return Decision.allow()
    return Decision.rewrite(f"<hazard_class>{canonical}</hazard_class>")


COMPONENT = Component(
    name="sopbench_dangerous_goods_final_xml_recovery",
    cls=ComponentClass.REACTIVE_GUARD,
    listens="pre_final_emit",
    matcher=_matches,
    handler=_handler,
    priority=100,
    trust=Trust(
        evidence_anchor=(
            "SOP section 6.1 enumerates the output vocabulary verbatim "
            "('Hazard Class A/B/C/D' or 'Unable to Decide'); section 6.5 "
            "mandates the <hazard_class> XML tag. OutputParser's "
            "_parse_plain_format fallback (third_party/SOP-Bench/.../"
            "evaluation/parser.py:429-431) coerces any final_output longer "
            "than 100 chars without a parseable structure to the null token "
            "'unknown', which compare_decisions (parser.py:482) treats as "
            "matching no real label."
        ),
        blast_radius="local",
        rollback_when=(
            "Disable if the rewrite ever changes a previously-correct answer "
            "(any task the prior frontier scored 1.0 that this iteration "
            "regresses), or if 'last in-prose mention' turns out not to "
            "reflect the model's intended final answer (e.g., a model that "
            "deliberates 'A vs B' and concludes with the earlier mention)."
        ),
        fallback=(
            "If no SOP-declared label appears anywhere in the final_output, "
            "return ALLOW so the existing OutputParser path (and its "
            "'unknown' result) stands — no fabricated guess."
        ),
    ),
)
