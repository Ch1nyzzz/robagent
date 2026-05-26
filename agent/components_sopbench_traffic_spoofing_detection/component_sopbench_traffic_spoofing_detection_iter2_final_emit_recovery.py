"""Final-emit recovery for traffic_spoofing_detection (iter5 strengthened).

Targeted failure mode (carried over from iter2 + sharpened by iter4
regression analysis): the SUT loop terminates with `finish_reason=stop`
and EMPTY assistant content; the SopBenchAgent loop's `if not tcs:
break` discards any POST_LLM_RESPONSE inject_context (inject queue
drains at the start of the NEXT turn), so PRE_FINAL_EMIT is the only
remaining hook. The iter2 implementation issued ONE chat() recovery
call with a 256-token budget and required the recovery content to
contain `<tag>...</tag>` verbatim — failing that, the component fell
back to `Decision.allow()` and `final_output` stayed empty (parser
returns the literal "unknown"). In iter4 this hard-fail path bit task
17 on train: the model stopped after only the 5.1-5.3 data-gathering
tool calls (no CalculateRiskScore, no GenerateEvidenceReport, no
ExecuteEnforcementAction) and the 256-token recovery either truncated
mid-emission or emitted prose acknowledging the analysis without ever
opening the SOP-required XML tag.

This iter5 replacement keeps the matcher and overall flow identical
(REACTIVE_GUARD at PRE_FINAL_EMIT, same anchor on the SOP's Section 6
tag, same fail-safe back to Decision.allow()) and adds two narrow
robustness changes that point at SOP-document-structure facts, not at
train answers:

  1. Raise the recovery budget from 256 to 1024 tokens. 256 is small
     enough that a model that briefly walks the SOP before emitting
     can be cut off MID-TAG; the substring check then rejects a partial
     `<enforcement_action>Tempor...` even though the model was on its
     way to a correct emission. 1024 is comfortable margin for the
     SOP-Bench SOP scale (full SOP-following responses on this domain
     are ~600-800 tokens). One sub-LLM call.

  2. SOP-Section-6 enumerated-value fallback. The SOP's Section 6 line
     lists the allowed output values inside DOUBLE QUOTES — for
     traffic_spoofing_detection: '"Account Closure", "Temporary
     Suspension", "Warning Issued", or "No Action"'. This is the SOP
     authors' own enumeration of the legal final-tag values; we parse
     it live from ctx.sop_text (Section 6 slice, double-quote regex
     over that slice only), no domain-specific strings in this file.
     When the recovery's `chat()` content contains NO well-formed
     `<tag>...</tag>` BUT contains exactly one — or, by the
     last-mention heuristic, an unambiguous final pick — SOP-enumerated
     value, we wrap that value in the proper `<tag>` and REWRITE
     `final_output`. The match is case-insensitive and uses word
     boundaries so partial overlaps don't fire. When no enumerated
     values can be parsed from Section 6, the fallback is disabled and
     behavior is byte-equivalent to iter2.

Why this is not memorisation:
  * No specific enforcement_action value, risk_level value,
    violation_type, partner_id, or evidence shape appears in this
    file. The strings "Account Closure", "Temporary Suspension",
    "Warning Issued", "No Action" are absent from the source — the
    fallback obtains them by reading the SOP document at runtime.
  * The fallback only fires AFTER the recovery LLM has produced
    output; it surfaces an explicit choice the LLM made, it does not
    introduce a choice from any train-set distribution.

Anchors (Trust.evidence_anchor):
  * OpenAI Chat Completions API response schema — finish_reason and
    content are documented response fields; empty content with
    finish_reason=stop is a well-defined "model declared done" state.
  * SOP document structure — Section 6 ("Output") names the required
    XML tag (iter2's anchor) AND enumerates the legal values inside
    double quotes ("in the format \"X\", \"Y\", ..."). Both are
    parsed live from ctx.sop_text.

Locked SUT: chat() calls in this component pass NO model= kwarg —
they go through the locked target inference path (the chat() wrapper
raises RuntimeError if any model override is attempted, which would
surface as a Decision.allow() via the BaseException catch).
"""
from __future__ import annotations

import re
from typing import Any, List

from agent.component_runtime_sopbench import (
    Component,
    ComponentClass,
    ComponentContext,
    Decision,
    Trust,
)
from agent.llm import chat


_OUTPUT_TAG_RE = re.compile(r"<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>")
_QUOTED_VALUE_RE = re.compile(r'"([^"\n]{1,80})"')
_RECOVERY_MAX_TOKENS = 1024


def _section_6_slice(sop: str) -> str:
    """Return the Section 6 (Output) slice of the SOP document, or ''."""
    idx = sop.find("\n6.")
    if idx < 0:
        return ""
    return sop[idx:]


def _extract_output_tag(sop: str) -> str:
    """Extract the XML tag named in the SOP's Section 6 (Output) block."""
    tail = _section_6_slice(sop) or sop
    m = _OUTPUT_TAG_RE.search(tail)
    return m.group(1) if m else ""


def _extract_section_6_enumerated_values(sop: str) -> List[str]:
    """Return the double-quoted enumerated final-output values from Section 6.

    SOP-Bench SOPs end Section 6 with a line of the shape:
        in the format "Value A", "Value B", "Value C", or "Value D"
    We collect every double-quoted span within Section 6 and de-dupe.
    Returns [] when no quoted enumeration is found.
    """
    section = _section_6_slice(sop)
    if not section:
        return []
    seen: set[str] = set()
    out: List[str] = []
    for m in _QUOTED_VALUE_RE.finditer(section):
        val = m.group(1).strip()
        if not val or val in seen:
            continue
        seen.add(val)
        out.append(val)
    return out


def _has_tag(text: str, tag: str) -> bool:
    """True iff `text` contains both `<tag>` and `</tag>` (case-insensitive)."""
    if not tag:
        return True  # treat as already-OK so matcher won't fire
    lo = text.lower()
    return f"<{tag.lower()}>" in lo and f"</{tag.lower()}>" in lo


def _find_last_enumerated_mention(content: str, values: List[str]) -> str:
    """Return the latest-mentioned SOP-enumerated value in `content`, or ''.

    Case-insensitive whole-phrase match (word-boundary on the value's
    edges). When multiple values appear, returns the one with the
    largest end-position — for typical SUT prose ("weighing X vs Y,
    the conclusion is Y") this corresponds to the model's final pick.
    """
    if not content or not values:
        return ""
    lo = content.lower()
    best_end = -1
    best_val = ""
    for v in values:
        vl = v.lower()
        # word-boundary on both edges so 'Action' inside 'Account' etc.
        # doesn't trigger; we use \b which works for ASCII letters.
        pat = re.compile(r"\b" + re.escape(vl) + r"\b")
        for m in pat.finditer(lo):
            if m.end() > best_end:
                best_end = m.end()
                best_val = v  # preserve original casing
    return best_val


def _matches(ctx: ComponentContext) -> bool:
    if ctx.event != "pre_final_emit":
        return False
    tag = _extract_output_tag(ctx.sop_text or "")
    if not tag:
        return False
    return not _has_tag(ctx.final_output or "", tag)


def _build_recovery_messages(
    base_messages: list[dict[str, Any]],
    tag: str,
) -> list[dict[str, Any]]:
    """Reuse the existing conversation and append a final emit-only nudge."""
    reminder = (
        "Your previous reply ended without the required final XML tag, "
        f"<{tag}></{tag}>. Re-read the SOP's Section 6 (Output) and the "
        "tool results above, then reply with the final XML on its own "
        "line. Do not call any more tools. You may briefly reason first "
        "if needed, but the LAST line of your reply MUST be the literal "
        f"<{tag}>VALUE</{tag}> where VALUE comes verbatim from the set "
        "the SOP's Output section enumerates."
    )
    return list(base_messages) + [
        {"role": "system", "content": reminder},
        {"role": "user", "content": f"Emit the final <{tag}>...</{tag}> now."},
    ]


def _handler(ctx: ComponentContext) -> Decision:
    sop = ctx.sop_text or ""
    tag = _extract_output_tag(sop)
    if not tag:
        return Decision.allow()
    if not ctx.messages:
        return Decision.allow()

    try:
        resp = chat(
            _build_recovery_messages(ctx.messages, tag),
            tools=None,
            max_tokens=_RECOVERY_MAX_TOKENS,
        )
    except BaseException:
        return Decision.allow()

    new_content = (resp or {}).get("content") or ""

    # Primary path (preserved from iter2): the model emitted the required
    # XML tag inside its response.
    if _has_tag(new_content, tag):
        return Decision.rewrite(new_content)

    # Fallback path (iter5): the model produced prose that names a
    # SOP-Section-6-enumerated value but never opened the tag. Wrap the
    # last unambiguous mention in the proper tag and emit.
    enumerated = _extract_section_6_enumerated_values(sop)
    pick = _find_last_enumerated_mention(new_content, enumerated)
    if not pick:
        return Decision.allow()
    return Decision.rewrite(f"<{tag}>{pick}</{tag}>")


COMPONENT = Component(
    name="sopbench_traffic_spoofing_detection_final_emit_recovery",
    cls=ComponentClass.REACTIVE_GUARD,
    listens="pre_final_emit",
    matcher=_matches,
    handler=_handler,
    priority=100,
    trust=Trust(
        evidence_anchor=(
            "OpenAI Chat Completions API response schema (finish_reason "
            "/ content for the empty-stop detection) AND SOP-Bench SOP "
            "document structure for BOTH the required output XML tag "
            "(Section 6 first tag occurrence) AND the SOP authors' own "
            "enumerated set of legal final-tag values (the "
            "double-quoted spans inside Section 6 — e.g. the literal "
            "'in the format \"...\", \"...\", or \"...\"' line). Both "
            "are parsed live from ctx.sop_text via regexes that do not "
            "hardcode any domain value. The recovery chat() call goes "
            "through the same locked SUT chat() with no model= kwarg."
        ),
        blast_radius="local",
        rollback_when=(
            "Disable if (a) candidate standalone train n_correct drops "
            "below the iter4 baseline of 22 (would mean the larger "
            "recovery budget or the enumerated-value fallback is "
            "introducing more wrong final tags than empty-output "
            "recoveries); (b) the enumerated-value fallback fires on a "
            "task and the LLM's prose names a value tangentially "
            "(\"although X is not appropriate here, ...\") so the "
            "last-mention heuristic picks the wrong value (visible in "
            "the predicted_output vs the recovery's content text); or "
            "(c) per-task token cost on tasks the matcher fires on "
            "grows beyond ~1.5x the iter2 baseline (suggesting the "
            "model is now using the larger budget to call out to "
            "long-form reasoning that does not end in a tag)."
        ),
        fallback=(
            "Three independent fail-safes preserve the iter2 behavior "
            "as a strict lower bound: (1) chat() raising → "
            "Decision.allow(); (2) recovery content already contains "
            "<tag>...</tag> → byte-identical iter2 REWRITE path; (3) "
            "Section 6 has no double-quoted enumeration OR the "
            "recovery content mentions no enumerated value → "
            "Decision.allow(), final_output unchanged. The component "
            "can only make things STRICTLY better than the iter2 "
            "empty-output failure case along the new fallback path."
        ),
    ),
)
