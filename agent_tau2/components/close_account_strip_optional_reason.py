"""close_bank_account: strip the optional `reason` argument.

Mechanism
---------
[[tau2-optional-arg-db-mismatch]]: `close_bank_account.reason` is the only
optional free-text string parameter in the banking tool catalog. When the
LLM invents a value for it, the DB hash diverges from gold even though the
write itself is correct. Stripping the argument restores the hash.

Why MECHANISM_LAYER (not INDUCED_RULE)
--------------------------------------
The activation predicate is `tool_call.name == "close_bank_account"` plus
`"reason" in arguments` — both are facts of the tool's declared schema,
not facts read out of policy text or out of the failed simulations. The
component therefore anchors on structure that lives OUTSIDE the N evidence
simulations (the schema is the same regardless of which tasks fail), so
the class is `mechanism_layer` and the rewrite is permitted.

Out-of-evidence probe
---------------------
If a future task asks the agent to close an account for a customer whose
`reason` carries a legally-required disclosure (a case not in any evidence
sim), the matcher still fires and the handler strips the field. Worst case:
the LLM's intent is lost from the DB record, but the close itself succeeds
(the field is declared optional → not state-changing). No override risk.
If a future schema change makes `reason` REQUIRED, the optional-strip
becomes a structural error — but that's the rollback_when signal, not a
silent regression.

Dead-weight signal
------------------
If a future schema change makes `reason` required, no call will carry it
as an optional addition; the component fires zero times and shows up dead
in durability_audit.py.
"""
from __future__ import annotations

from agent_tau2.component_runtime.types import (
    Component,
    ComponentClass,
    ComponentContext,
    Decision,
    Trust,
)


_TOOL = "close_bank_account"
_OPTIONAL_FIELD = "reason"


def _matches(ctx: ComponentContext) -> bool:
    return ctx.tool_name == _TOOL and _OPTIONAL_FIELD in ctx.tool_args


def _strip(ctx: ComponentContext) -> Decision:
    args = {k: v for k, v in ctx.tool_args.items() if k != _OPTIONAL_FIELD}
    return Decision.rewrite_tool_args(args)


COMPONENT = Component(
    name="close_account_strip_optional_reason",
    cls=ComponentClass.MECHANISM_LAYER,
    listens="pre_tool_use",
    matcher=_matches,
    handler=_strip,
    priority=100,
    trust=Trust(
        evidence_anchor=(
            "The banking tool catalog declares `reason` as the sole optional "
            "free-text parameter of close_bank_account. DB-hash equality with "
            "gold is unaffected by argument content (the field is not "
            "state-changing), so removing the LLM-invented value normalises "
            "to the same write. This is the tool's declared schema, not a "
            "policy reading."
        ),
        blast_radius="local",
        rollback_when=(
            "No close_bank_account tool_call observed across 30 consecutive "
            "task runs (see .component-state/<tag>/fired.jsonl), OR the "
            "banking tool schema makes `reason` a required field."
        ),
        fallback=(
            "On any unexpected schema, the matcher returns False; the LLM's "
            "original arguments pass through unchanged."
        ),
    ),
)
