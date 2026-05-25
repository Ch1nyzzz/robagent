"""SOP completion checklist injector for traffic_spoofing_detection.

Failure mode targeted: the v0 baseline frequently terminates the FC loop
with `finish_reason=stop` and EMPTY assistant content BEFORE running SOP
Section 5.6 (Enforcement Action) and BEFORE emitting the mandatory
`<enforcement_action>...</enforcement_action>` XML tag. With an empty
final output, the SOP-Bench parser returns `"unknown"` (parser.py line
62-63), which the grader scores 0. 13 of 18 train failures match this
pattern (tasks 1, 2, 3, 6, 8, 10, 14, 15, 17, 23, 25, 28, 29).

Approach: SESSION_START CHANNEL that reads the live `ctx.sop_text`,
extracts (a) the Section 5.X subsection headers and (b) the required
output XML tag name from Section 6, then appends a "Process Completion
Checklist" block to the system prompt enumerating the steps the agent
MUST touch and the EXACT tag it MUST emit before stopping. This relies
on stable SOP document structure (every SOP-Bench SOP has a numbered
Section 5 Main Procedure and a Section 6 specifying the output tag);
nothing in this file is task-specific.
"""
from __future__ import annotations

import re
from typing import List

from agent.component_runtime_sopbench import (
    Capability,
    Component,
    ComponentClass,
    ComponentContext,
    Decision,
    StateScope,
    Trust,
)


_SECTION_HEADER_RE = re.compile(r"^\s*(5\.\d+)\s+(.+?)\s*$", re.MULTILINE)
_OUTPUT_TAG_RE = re.compile(r"<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>")


def _extract_subsections(sop: str) -> List[tuple[str, str]]:
    """Return ordered (id, title) pairs for Section 5.X headers."""
    seen: set[str] = set()
    out: List[tuple[str, str]] = []
    for m in _SECTION_HEADER_RE.finditer(sop):
        sid, title = m.group(1), m.group(2).strip()
        if sid in seen:
            continue
        seen.add(sid)
        out.append((sid, title))
    return out


def _extract_output_tag(sop: str) -> str:
    """Return the XML tag name the SOP's output spec mentions (e.g. 'enforcement_action').

    Scans the slice of the SOP starting at "6. Output" (or "6 ") so we do
    not pick up tag-shaped tokens in the procedure body. Returns empty
    string when no tag is detectable.
    """
    idx = sop.find("\n6.")
    tail = sop[idx:] if idx >= 0 else sop
    m = _OUTPUT_TAG_RE.search(tail)
    return m.group(1) if m else ""


def _matches(ctx: ComponentContext) -> bool:
    if ctx.event != "session_start":
        return False
    sop = ctx.sop_text or ""
    return bool(_extract_subsections(sop)) and bool(_extract_output_tag(sop))


def _handler(ctx: ComponentContext) -> Decision:
    sop = ctx.sop_text or ""
    subsections = _extract_subsections(sop)
    tag = _extract_output_tag(sop)

    bullets = "\n".join(f"  [ ] {sid} {title}" for sid, title in subsections)
    reminder = (
        "\n\n=== PROCESS COMPLETION CHECKLIST (machine-enforced) ===\n"
        "Your reply is parsed by a deterministic extractor. If you finish "
        "your turn (finish_reason=stop) without emitting the EXACT XML tag "
        f"below, your answer is recorded as the literal string \"unknown\" "
        "and the task is scored 0. An empty assistant reply at finish_reason="
        "stop is a SILENT FAILURE — there is no retry.\n\n"
        "Before you stop, verify ALL of the following:\n"
        f"{bullets}\n"
        f"  [ ] Final XML tag emitted: <{tag}>VALUE</{tag}>\n\n"
        "Operational rules:\n"
        "  1. Do NOT stop after only collecting data (subsections 5.1-5.3 "
        "or similar early data-gathering steps). Every Section 5 subsection "
        "above must be reasoned about, and any subsection whose action is "
        "implementable as one of the provided tools should trigger that "
        "tool call.\n"
        f"  2. The final user-visible turn MUST contain the literal "
        f"substring \"<{tag}>\" followed by your chosen value followed by "
        f"\"</{tag}>\". The value must come verbatim from the set the SOP's "
        "Output section enumerates.\n"
        "  3. If you have already produced the final XML tag, stop calling "
        "tools — do not loop.\n"
        "=== END CHECKLIST ===\n"
    )
    return Decision.inject_context(reminder)


COMPONENT = Component(
    name="sopbench_traffic_spoofing_detection_sop_completion_checklist",
    cls=ComponentClass.CHANNEL,
    listens="session_start",
    matcher=_matches,
    handler=_handler,
    state_scope=StateScope.NONE,
    capabilities=(Capability.NONE,),
    priority=50,
    trust=Trust(
        evidence_anchor=(
            "SOP document structure: every SOP-Bench SOP exposes a numbered "
            "Section 5 (Main Procedure) with `5.X Title` subsection headers, "
            "and a Section 6 (Output) line containing a literal XML tag "
            "`<tag_name>`. Both are parsed live from ctx.sop_text — no "
            "task-specific or domain-specific names appear in this file."
        ),
        blast_radius="local",
        rollback_when=(
            "Disable if train TSR drops vs v0 (would indicate the extra "
            "system text is degrading reasoning), or if the checklist "
            "header text starts appearing inside the model's <enforcement_"
            "action> emission (would indicate the model is leaking the "
            "reminder into the answer)."
        ),
        fallback=(
            "Component injects context only — if the SOP is missing Section "
            "5 or Section 6 the matcher returns False and the system prompt "
            "is unchanged."
        ),
    ),
)
