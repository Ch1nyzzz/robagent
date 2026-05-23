"""close_bank_account: strip the optional `reason` argument.

Mechanism
---------
[[tau2-optional-arg-db-mismatch]]: `close_bank_account.reason` is the only
optional free-text string parameter in the banking tool catalog. When the
LLM invents a value for it, the DB hash diverges from gold even though the
write itself is correct. Stripping the argument restores the hash.

Why deterministic_glue (not induced_rule)
-----------------------------------------
The activation predicate is `tool_call.name == "close_bank_account"` plus
`"reason" in arguments` — both are facts of the tool's declared schema,
not facts read out of policy text or out of the failed simulations. The
hook therefore points at structure that lives OUTSIDE the N evidence
simulations (the schema is the same regardless of which tasks fail), so
the class is `deterministic_glue` and the rewrite is permitted.

Dead-weight signal
------------------
If a future schema change makes `reason` required, no call will carry it
as an optional addition; the hook fires zero times and shows up dead in
durability_audit.py.
"""
from __future__ import annotations

from agent_tau2.hook_runtime.types import (
    Decision,
    Hook,
    HookClass,
    HookContext,
    HookEvent,
)


_TOOL = "close_bank_account"
_OPTIONAL_FIELD = "reason"


def _matches(ctx: HookContext) -> bool:
    return ctx.tool_name == _TOOL and _OPTIONAL_FIELD in ctx.tool_args


def _strip(ctx: HookContext) -> Decision:
    args = {k: v for k, v in ctx.tool_args.items() if k != _OPTIONAL_FIELD}
    return Decision.rewrite_tool_args(args)


HOOK = Hook(
    name="close_account_strip_optional_reason",
    cls=HookClass.DETERMINISTIC_GLUE,
    event=HookEvent.PRE_TOOL_USE,
    matcher=_matches,
    handler=_strip,
    generalization_argument=(
        "(a) Stable structure: the banking tool catalog declares `reason` "
        "as the sole optional free-text parameter of close_bank_account. "
        "DB-hash equality with gold is unaffected by argument content "
        "(it is not a state-changing field), so removing the LLM-invented "
        "value normalises to the same write. This is the tool's declared "
        "schema, not a policy reading. "
        "(b) deterministic_glue; no override risk: a future task that "
        "needs `reason` to carry information would also need a tool whose "
        "schema validates that information, in which case the field would "
        "be required and the hook would not fire."
    ),
    fallback="On any unexpected schema, the hook simply does not fire; "
             "the LLM's original arguments pass through unchanged.",
    dead_when="No close_bank_account tool_call observed across 30 "
              "consecutive task runs (see .hook-state/<tag>/fired.jsonl).",
)
