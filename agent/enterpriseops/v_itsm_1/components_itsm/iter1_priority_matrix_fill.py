"""Fill `priority` from the documented Impact x Urgency matrix.

The ITSM domain `system_prompt` (Sections 3.1 + 4.2, included verbatim in
every itsm task) states:

    "Priority (`critical`, `high`, `moderate`, `low`, `planning`) is
     determined automatically by the system based on a below pre-defined
     matrix of Impact x Urgency."

Empirically the gym MCP server does NOT do this server-side: the priority
field on `create_incident` / `update_incident` / `create_problem` /
`update_problem` is stored as whatever the agent passed (or remains at
the pre-existing value / default 'planning' if omitted). SQL verifiers
join on the `priority` column, so trusting the "automatically" wording
silently fails the priority check on tasks where the matrix-derived
priority differs from the stored value.

This component fires at `pre_tool_arg_validation` when:
  - the tool is one of the four priority-bearing ITSM CRUD tools, AND
  - the args carry BOTH `impact` and `urgency` (so the matrix lookup is
    well-defined without an extra MCP read of the existing record), AND
  - the args' `priority` doesn't already equal the matrix-derived value
    (skip when already correct).

It then rewrites the args to set `priority` to the matrix-derived value.
Cases where the agent updates only `impact` OR only `urgency` are left
untouched — those require reading the existing record's other axis, which
is outside this narrow guard.
"""
from __future__ import annotations

from ..runtime import (
    Component,
    ComponentClass,
    ComponentContext,
    Decision,
    Trust,
)


# Impact x Urgency -> Priority, transcribed from the ITSM domain
# system_prompt's Section 3.1 / 4.2 priority matrix.
_MATRIX: dict[tuple[str, str], str] = {
    ("high",   "high"):   "critical",
    ("high",   "medium"): "high",
    ("high",   "low"):    "moderate",
    ("medium", "high"):   "high",
    ("medium", "medium"): "moderate",
    ("medium", "low"):    "low",
    ("low",    "high"):   "moderate",
    ("low",    "medium"): "low",
    ("low",    "low"):    "planning",
}

# ITSM MCP tools that take impact/urgency/priority on create or update.
_PRIORITY_TOOLS = frozenset({
    "create_incident",
    "update_incident",
    "create_problem",
    "update_problem",
})


def _norm(value) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _matches(ctx: ComponentContext) -> bool:
    if ctx.current_tool_name not in _PRIORITY_TOOLS:
        return False
    args = ctx.current_tool_args or {}
    impact = _norm(args.get("impact"))
    urgency = _norm(args.get("urgency"))
    key = (impact, urgency)
    if key not in _MATRIX:
        return False
    expected = _MATRIX[key]
    return _norm(args.get("priority")) != expected


def _handler(ctx: ComponentContext) -> Decision:
    args = dict(ctx.current_tool_args or {})
    impact = _norm(args.get("impact"))
    urgency = _norm(args.get("urgency"))
    args["priority"] = _MATRIX[(impact, urgency)]
    return Decision.rewrite(args)


COMPONENT = Component(
    name="enterpriseops_itsm_priority_matrix_fill",
    cls=ComponentClass.MECHANISM_LAYER,
    listens="pre_tool_arg_validation",
    matcher=_matches,
    handler=_handler,
    priority=100,
    emits=(),
    trust=Trust(
        evidence_anchor=(
            "ITSM domain system_prompt Sections 3.1 (Incident Priority "
            "Calculation) and 4.2 (Problem Priority Calculation): the "
            "Impact x Urgency -> Priority matrix is documented verbatim in "
            "every itsm task's system_prompt as the authoritative mapping. "
            "Gym MCP server behaviour: priority is stored as-passed, not "
            "derived. Verifier SQL queries (e.g. "
            "'WHERE priority = ...' joins) compare against the stored value, "
            "so an unset/default priority on impact+urgency changes flunks."
        ),
        blast_radius="local",
        rollback_when=(
            "(a) Any of the 10 currently-passing v0 itsm tasks regresses "
            "after this component lands; (b) trace audit shows a verifier "
            "expecting priority that contradicts the matrix derivation when "
            "the agent supplied both impact and urgency (would invert the "
            "rule); (c) the gym server is observed to actually auto-derive "
            "priority server-side (then this component becomes a no-op "
            "since current_priority would already equal _MATRIX[key])."
        ),
        fallback=(
            "Matcher returns False unless tool name is in _PRIORITY_TOOLS "
            "and BOTH impact and urgency are present and valid enum values. "
            "Update calls that change only one axis are not touched. If the "
            "agent already set priority to the matrix-derived value, no "
            "rewrite occurs (matcher returns False). On any other tool / "
            "argument shape the call passes through unchanged."
        ),
    ),
)
