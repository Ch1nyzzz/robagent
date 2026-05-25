"""Enforce SOP 5.3.2's Pending->Processing transition by normalising the
input current_status on updateResolutionStatus -- but only when the
problem-driven transition is structurally engaged.

SOP section 5.3.2 says verbatim:
    "Update resolution_status based on problem_type:
       - Initialize as 'Pending'
       - Progress to 'Processing' upon problem confirmation"

The "Initialize as Pending" invariant exists to define the starting
state for the Pending->Processing transition. That transition is gated
by `problem_type` being non-empty (the SOP's "problem confirmation").
When `problem_type` is empty, there is no SOP-defined transition to
evaluate, and `updateResolutionStatus`'s third branch returns
`current_status` unchanged (see tools.py: the `else` after the Wrong
Item / non-empty problem checks). The agent's chosen current_status is
then a legitimate signal that the tool will echo back -- often the
agent has set it to the resolution_status field returned by
validateBarcode on the matched path (which the toolspec exposes), and
the model uses the echo to confirm a clean shipment before emitting
'Resolved'.

The earlier body of this component (iter2) fired on every
updateResolutionStatus call whose current_status != 'Pending'. That
worked for the problem-confirmation case but silently overrode the
agent on the empty-problem case. The mh_iter2 task 2 reasoning trace
shows this concretely: the model called updateResolutionStatus four
times trying current_status='Processing' / 'Resolved' to confirm a
clean shipment; this component silently rewrote each to 'Pending', the
tool returned 'Pending' each time, and the model -- with the iter1
ban on emitting 'Pending' -- improvised a final answer instead of
having tool-grounded confirmation.

This refinement narrows the matcher to fire only when `problem_type`
is non-empty (the SOP-defined transition is structurally engaged).
For empty problem_type, the agent retains authority over
current_status and gets honest echo feedback from the tool.

Evidence anchors:
  * SOP 5.3.2 (quoted above) -- the second clause is the gated
    transition; "Initialize as Pending" scopes to that transition.
  * tools.py :: updateResolutionStatus branching:
      - if 'Wrong Item' in problem_type -> 'Returned to Vendor'
      - elif len(problem_type) > 0 and current_status=='Pending'
            -> 'Processing'
      - else -> returns current_status unchanged
    The third branch is the one that lets the agent's choice through;
    the SOP's Pending invariant has no bearing on it.
  * toolspecs.json :: updateResolutionStatus.inputSchema declares
    current_status as a required string parameter accepting one of
    ['Pending','Processing','Resolved','Returned to Vendor'].

Component identity (name, class, mount) is preserved -- this is an
in-place replace_node refinement, not a new node.
"""
from __future__ import annotations

from agent.component_runtime_sopbench import (
    Capability,
    Component,
    ComponentClass,
    ComponentContext,
    Decision,
    StateScope,
    Trust,
)


_INITIAL_STATUS = "Pending"
_TARGET_TOOL = "updateResolutionStatus"


def _matches(ctx: ComponentContext) -> bool:
    if ctx.benchmark != "warehouse_package_inspection":
        return False
    if ctx.current_tool_name != _TARGET_TOOL:
        return False
    args = ctx.current_tool_args
    if not isinstance(args, dict):
        return False
    if args.get("current_status") == _INITIAL_STATUS:
        return False
    problem_type = args.get("problem_type")
    if not isinstance(problem_type, list) or len(problem_type) == 0:
        # SOP 5.3.2's Pending->Processing transition only triggers on
        # problem confirmation; with no problem_type the tool echoes
        # current_status unchanged and the agent's choice is honest signal.
        return False
    return True


def _handler(ctx: ComponentContext) -> Decision:
    new_args = dict(ctx.current_tool_args)
    new_args["current_status"] = _INITIAL_STATUS
    return Decision.rewrite(new_args)


COMPONENT = Component(
    name="sopbench_warehouse_package_inspection_update_status_pending_init",
    cls=ComponentClass.MECHANISM_LAYER,
    listens="pre_tool_use",
    matcher=_matches,
    handler=_handler,
    state_scope=StateScope.NONE,
    capabilities=(Capability.NONE,),
    priority=100,
    trust=Trust(
        evidence_anchor=(
            "SOP 5.3.2: 'Initialize as Pending -> Progress to Processing upon "
            "problem confirmation' (the Pending->Processing transition is "
            "gated by non-empty problem_type) AND "
            "third_party/SOP-Bench/.../warehouse_package_inspection/tools.py::"
            "updateResolutionStatus branching: when problem_type is empty and "
            "current_status is not 'Returned to Vendor', the tool returns "
            "current_status unchanged (the `else` branch after the Wrong Item "
            "and non-empty-problem branches) AND "
            "toolspecs.json::updateResolutionStatus.inputSchema declares "
            "current_status as a required string in "
            "['Pending','Processing','Resolved','Returned to Vendor']."
        ),
        blast_radius="local",
        rollback_when=(
            "SOP 5.3.2 changes the initial state away from 'Pending' or moves "
            "the transition gate away from problem confirmation, or "
            "updateResolutionStatus drops/renames the current_status or "
            "problem_type parameter, or the tool changes its behaviour on the "
            "empty-problem_type branch to no longer echo current_status, or "
            "evolution_summary shows train accuracy drops after this refinement "
            "is admitted."
        ),
        fallback=(
            "iter2's unconditional rewrite -- always force current_status to "
            "'Pending', which trapped the agent in a Pending-echo loop on the "
            "empty-problem_type branch."
        ),
    ),
)
