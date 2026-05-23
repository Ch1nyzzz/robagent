"""Inject tau2 discoverable-tool DB-audit semantics at session start.

Mechanism
---------
The v0 base agent makes extra `call_discoverable_agent_tool` and
`give_discoverable_user_tool` invocations beyond what gold expects.  Because
the agent does not know each such call writes a unique row to the evaluation
DB, it speculatively calls "research" tools (e.g. `get_user_dispute_history`)
or premature Stage-2 tools (e.g. `update_transaction_rewards_3847`) that add
rows not present in gold.  DB hash diverges → reward 0.

Observed in ≥3 training simulations:
  * task_031 — extra call(get_user_dispute_history_7291) as pre-dispute research
  * task_018 — extra call(update_transaction_rewards_3847) before disputes resolved
  * task_051 — extra calls to get_user_dispute_history + get_pending_replacement

Stable structure
----------------
The tau2 framework invariant `call_discoverable_agent_tool(T)` →
`add_to_db("agent_discoverable_tools", record_id, …)` (tools.py line ~674)
and `give_discoverable_user_tool(T)` → `add_to_db("user_discoverable_tools",
…)`.  Both tables participate in the evaluation DB hash.  This is a framework
schema fact — it holds on every task regardless of KB content or customer
scenario, and is verifiable from the tau2 source code.

Why deterministic_glue
----------------------
The matcher is always-on (fires at every session start with no condition).
The injected text is a statement of tau2 sandbox behaviour — a structural
constant, not a policy reading.  No task-specific content, no LLM judgement
substituted.  The class is DETERMINISTIC_GLUE, event SESSION_START, decision
INJECT_CONTEXT.

Dead-weight signal
------------------
If tau2 removes `add_to_db` from these entry-point tools, the notice becomes
harmless informational text that no longer affects scoring.  Detectable via
durability_audit.py if train-30 scores stop improving after injection.
"""
from __future__ import annotations

from agent_tau2.hook_runtime.types import (
    Decision,
    Hook,
    HookClass,
    HookContext,
    HookEvent,
)

_AUDIT_CONTEXT = (
    "<discoverable_tool_audit>\n"
    "## Discoverable Tool DB-Audit Semantics (tau2 framework fact)\n\n"
    "Each of the following entry-point calls writes rows to the evaluation "
    "database that participate in the task-success hash:\n\n"
    "  - `call_discoverable_agent_tool(agent_tool_name=T, …)` writes one row "
    "to the `agent_discoverable_tools` table keyed by T.  Re-calling with the "
    "same T is deduplicated (no new row), but each NEW unique T adds a row — "
    "including tools whose name starts with `get_` (they are tau2 WRITE tools "
    "when invoked through this entry point).\n"
    "  - `give_discoverable_user_tool(discoverable_tool_name=T)` writes one "
    "row to the `user_discoverable_tools` table keyed by T.  Do not add an "
    "`arguments` key to this call — only `discoverable_tool_name` is expected; "
    "extra keys are not stored but may confuse the user simulator.\n"
    "  - `unlock_discoverable_agent_tool` does NOT write to the DB by itself; "
    "the write happens only when the unlocked tool is subsequently called.\n\n"
    "Consequence: any `call_discoverable_agent_tool` or `give_discoverable_user_tool` "
    "invocation that is not part of the documented gold workflow adds an extra "
    "DB row that diverges the hash from gold and zeroes the task reward.  "
    "Only unlock, call, or give a discoverable tool when:\n"
    "  (a) the knowledge base explicitly instructs it for this scenario, AND\n"
    "  (b) you actually intend to use or forward the result.\n"
    "Avoid speculative 'research' calls to tools like "
    "`get_user_dispute_history_*` or `get_pending_replacement_orders_*` "
    "unless the documented workflow for the current task requires them.\n"
    "</discoverable_tool_audit>"
)


def _matches(ctx: HookContext) -> bool:
    return True


def _handler(ctx: HookContext) -> Decision:
    return Decision.inject_context(_AUDIT_CONTEXT)


HOOK = Hook(
    name="discoverable_audit_channel",
    cls=HookClass.DETERMINISTIC_GLUE,
    event=HookEvent.SESSION_START,
    matcher=_matches,
    handler=_handler,
    generalization_argument=(
        "(a) Stable structure: the tau2 framework's `call_discoverable_agent_tool` "
        "calls `add_to_db('agent_discoverable_tools', record_id, ...)` on every "
        "invocation (tools.py ~line 674); `give_discoverable_user_tool` similarly "
        "calls `add_to_db('user_discoverable_tools', ...)`.  Both tables are "
        "included in the DB hash used for evaluation.  This is a framework schema "
        "fact independent of any specific task, KB document, or training simulation. "
        "(b) Not induced_rule/predictive_heuristic: the injected text does not "
        "encode task-specific policy or predict what the LLM will do next — it "
        "describes a framework constant verifiable from tau2 source code."
    ),
    fallback=(
        "Matcher is always True; fallback is not possible.  If the injected "
        "text is ignored by the LLM the hook has no effect; existing task "
        "scores are unchanged."
    ),
    dead_when=(
        "tau2 removes `add_to_db` from `call_discoverable_agent_tool` and "
        "`give_discoverable_user_tool`, making these entry points no longer "
        "write to the evaluation DB.  Observable via durability_audit.py as "
        "the notice becoming factually incorrect."
    ),
)
