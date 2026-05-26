"""Phase F demo: on_explicit_terminate → artifact gate BLOCK (toolathlon).

PURPOSE
-------
Proves Phase D's `on_explicit_terminate` Tier-1 event composes with a
deterministic filesystem check to gate task termination. This component
is NOT part of the active toolathlon frontier — it is a demonstrator
that the new expressiveness shippable to proposers in Phase E reaches
end-to-end dispatch.

MECHANISM
---------
Toolathlon's `CrTaskAgent.run_interaction_loop` emits `on_explicit_terminate`
after `termination_checker(...)` returns True (the agent has produced a
final answer and the loop is about to break). A subscriber may BLOCK the
event, in which case the loop continues and the agent is given another
chance to satisfy whatever invariant the gate enforces.

This demo enforces the invariant: "if the user asked the agent to write a
report to the workspace, the workspace must contain a file named
`report.md` before the agent is allowed to terminate." The matcher only
fires if the user prompt contains the substring 'report.md' OR
'write a report' (so the gate is scoped to tasks that actually require it).
The handler checks the workspace; if the artifact is missing it BLOCKs.

NOT-A-RULE
----------
The matcher anchors on the user-instruction grammar (substring presence),
which is a structural cue distinct from policy-text reading. The handler
performs a deterministic filesystem check — no LLM call. Both surfaces are
stable across task instances, so this is REACTIVE_GUARD, not INDUCED_RULE.
"""
from __future__ import annotations

import os
from typing import Iterable

from agent_toolathlon.component_runtime.types import (
    Component, ComponentClass, ComponentContext,
    Decision, Trust,
)


_TRIGGER_NEEDLES: tuple[str, ...] = (
    "report.md",
    "write a report",
    "save a report",
    "produce a report",
)


def _iter_user_query_strings(ctx: ComponentContext) -> Iterable[str]:
    """Best-effort extraction of the user query text from shared state.

    Toolathlon shells the user query into multiple places depending on
    single_turn vs multi-turn mode; we sweep the obvious candidates and
    bail silently if none are populated (the matcher then returns False).
    """
    cand = ctx.shared.get("_user_query")
    if isinstance(cand, str):
        yield cand
    # The full agent log carries the user instruction as the first
    # role=user entry; sample defensively.
    for item in ctx.history or []:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if role == "user" and isinstance(content, str):
                yield content
                break


def _matches(ctx: ComponentContext) -> bool:
    for text in _iter_user_query_strings(ctx):
        low = text.lower()
        if any(needle in low for needle in _TRIGGER_NEEDLES):
            return True
    return False


def _handler(ctx: ComponentContext) -> Decision:
    workspace = ctx.shared.get("_agent_workspace")
    if not isinstance(workspace, str) or not workspace:
        # Workspace not exposed (no isolation mount); the gate cannot make
        # a meaningful check, fall through to ALLOW so we don't block a
        # task by accident.
        return Decision.allow()
    target = os.path.join(workspace, "report.md")
    if os.path.isfile(target) and os.path.getsize(target) > 0:
        return Decision.allow()
    return Decision.block(
        reason=(
            "artifact_gate: user instruction implied a workspace report, "
            f"but {target!r} does not exist or is empty. The agent must "
            "write the report before terminating."
        )
    )


COMPONENT = Component(
    name="component_demo_artifact_gate",
    cls=ComponentClass.REACTIVE_GUARD,
    listens="on_explicit_terminate",
    matcher=_matches,
    handler=_handler,
    priority=100,
    emits=(),
    trust=Trust(
        evidence_anchor=(
            "Anchored on (a) the user-instruction grammar — the literal "
            "substring 'report.md' or 'write a report' appearing in the "
            "first user message — and (b) POSIX filesystem semantics for "
            "`os.path.isfile` + `os.path.getsize`. Both are stable across "
            "any task corpus; the matcher does not read policy text."
        ),
        blast_radius="local",
        rollback_when=(
            "Disable when toolathlon's workspace contract changes (e.g. "
            "the artifact moves out of `_agent_workspace`), or when the "
            "matcher needles produce false positives on tasks that mention "
            "'report.md' for unrelated reasons (track the BLOCK fire count "
            "vs the genuine missing-artifact failure count)."
        ),
        out_of_evidence_probe=(
            "Out of evidence: a task with the user instruction 'please "
            "write a report describing X to report.md' but where the agent "
            "terminates without calling fs-write_file. The matcher detects "
            "'write a report' AND 'report.md'; the handler finds "
            "workspace/report.md absent and returns Decision.block(...). "
            "The agent loop continues; if the agent then writes the file "
            "and terminates again, the matcher still fires but the handler "
            "now sees the file and returns Decision.allow()."
        ),
        fallback=(
            "Matcher=False (no report needle in user instruction) → handler "
            "not invoked. Handler returns allow() when workspace is unset "
            "or artifact is present → termination proceeds."
        ),
    ),
)
