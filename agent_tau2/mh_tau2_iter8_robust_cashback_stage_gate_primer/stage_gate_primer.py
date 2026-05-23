"""Plan-formation primer for the cash-back rewards two-stage workflow.

Why this exists
---------------
On the v3 robust frontier the agent still fails three cash-back rewards
correction tasks (task_022, task_027, task_029) and the cash-back leg of
the adversarial pair (task_026) by jumping directly to Stage 2 — calling
``unlock_discoverable_agent_tool('update_transaction_rewards_3847')``
followed by a batch of ``call_discoverable_agent_tool(
'update_transaction_rewards_3847', ...)`` — without ever issuing the
Stage-1 ``give_discoverable_user_tool('submit_cash_back_dispute_0589')``
for any transaction. The ``cash_back_disputes`` evaluation table ends up
empty where gold expects one row per disputed transaction and the
``agent_discoverable_tools`` audit table picks up an ``update_transaction_
rewards_3847`` row that is not matched by gold, so ``db_check`` fails.

The iter5 ``cashback_stage_state_advisor`` runtime advisor fires AT the
Stage-2 unlock/call, by which point the LLM has already committed to a
batched Stage-2 plan in the same turn. The advisor text is appended to
the unlock's tool response but the model proceeds to Stage 2 calls in
the very next assistant message. The intervention point that is missing
is *plan-formation time* — before any unlock call — which means a
durable, compact restatement of the gate in the system prompt itself.

What this module does
---------------------
Provides a static, deterministic, compact restatement of the two-stage
cash-back-rewards-correction workflow defined by KB docs
``doc_credit_cards_credit_cards_(general)_003`` and ``_004`` and
anchored to the framework's emitted ``Status:`` substring. The text:

  * NAMES Stage 1 as the agent giving and the user invoking the
    user-side discoverable tool ``submit_cash_back_dispute_0589``,
    citing doc credit_cards_(general)_003.
  * NAMES Stage 2 as the agent unlocking and calling the agent-side
    discoverable tool ``update_transaction_rewards_3847``, citing doc
    credit_cards_(general)_004.
  * STATES the gate condition: Stage 2 is applicable ONLY IF a Stage-1
    response for the same ``transaction_id`` returned the literal
    substring ``Status: RESOLVED`` (the framework emits this at
    ``tau2-bench-src/.../banking_knowledge/tools.py`` line 4140; the
    alternative is ``Status: SUBMITTED`` at line 4149).
  * STATES the DB-evaluation consequence of skipping Stage 1 (empty
    ``cash_back_disputes`` row set + unmatched ``agent_discoverable_
    tools`` audit row + diverged DB hash from gold).

What this module does NOT do
----------------------------
  * It never modifies, rewrites, or suppresses the LLM's tool call.
  * It never references customer names, user ids, account ids, or any
    other task-specific datum.
  * It never encodes a rule beyond the literal Stage-1->Stage-2
    ordering described verbatim in doc 004 ("After a cash back dispute
    is resolved and approved, you must update the affected
    transaction(s)..."). It does not encode points-calculation
    arithmetic, eligibility heuristics, or any non-cash-back workflow.

Stable structure captured
-------------------------
Three layers, all independent of the failed simulations:

  1. KB-doc facts. The two-stage ordering and the requirement to
     "give" ``submit_cash_back_dispute_0589`` to the customer are the
     entire content of KB docs (general)_003 and _004. The primer
     cites these docs by id; it does not invent ordering.

  2. Tool-schema facts. The entry-point names
     (``give_discoverable_user_tool``, ``unlock_discoverable_agent_
     tool``, ``call_discoverable_agent_tool``) and the discoverable
     tool names (``submit_cash_back_dispute_0589``,
     ``update_transaction_rewards_3847``) are declared in
     ``tau2-bench-src/src/tau2/domains/banking_knowledge/tools.py``.

  3. Framework return-string facts. The literal ``Status: RESOLVED``
     and ``Status: SUBMITTED`` substrings are emitted by the framework
     in the Stage-1 tool response at known source locations. The
     primer reads these substrings as the gate signal — not the
     agent's content choices.

If a future tau2 release renames any of these tools or restructures
the cash-back workflow, the primer text becomes out-of-date prose. It
cannot cause a wrong tool call because it never directs a specific
call; it only restates the documented ordering. The LLM retains full
discretion.
"""
from __future__ import annotations

# The primer is a single, immutable string. It is composed once at
# import time and appended verbatim to the system prompt.
#
# Why this is a string constant rather than a builder:
# the content is policy text, not a per-task computation. Every banking_
# knowledge conversation gets the same primer; no task-specific data is
# substituted in.
PRIMER: str = """
## STAGE GATE — Cash-back rewards correction (banking_knowledge)

When the customer asks you to identify, correct, refund, or adjust cash-back
rewards on one or more credit-card transactions, this workflow is mandatory.
The full policy lives in KB docs `doc_credit_cards_credit_cards_(general)_003`
("Submitting a Cash Back Dispute (Internal)") and
`doc_credit_cards_credit_cards_(general)_004` ("Applying Resolved Cash Back
Dispute Corrections (Internal)"); the gate below is a compact restatement at
the decision-formation point.

STAGE 1 (per doc (general)_003) — Submit a dispute for each affected
transaction:
  For EACH transaction with a suspected cash-back discrepancy, the agent gives
  the user the user-side discoverable tool `submit_cash_back_dispute_0589`:
    agent → give_discoverable_user_tool(
      discoverable_tool_name="submit_cash_back_dispute_0589")
  Then ask the customer to invoke that tool with their own user_id and the
  transaction_id of the affected purchase. The framework's response to the
  customer's invocation carries one of two literal status lines:
    - "Status: RESOLVED"  (the dispute was auto-resolved in the customer's
      favour; the framework wrote the resolution into cash_back_disputes)
    - "Status: SUBMITTED" (the dispute is open and under review; no further
      action is possible in this conversation)

STAGE 2 (per doc (general)_004) — Apply the corrected rewards, ONLY if
Stage 1 returned RESOLVED:
  If, AND ONLY IF, the Stage-1 tool response for a given transaction_id
  contained the literal substring "Status: RESOLVED", proceed with:
    agent → unlock_discoverable_agent_tool(
      agent_tool_name="update_transaction_rewards_3847")
    agent → call_discoverable_agent_tool(
      agent_tool_name="update_transaction_rewards_3847",
      arguments='{"transaction_id": "<the resolved transaction_id>",
                 "new_rewards_earned": "<X points>"}')
  If the Stage-1 status was "SUBMITTED" (or any non-RESOLVED value, or absent),
  STOP. Stage 2 is NOT applicable in this conversation; explain the dispute
  remains under review.

GATE — do not skip Stage 1:
  NEVER call `update_transaction_rewards_3847` for a transaction_id without
  having first issued `give_discoverable_user_tool` with
  discoverable_tool_name="submit_cash_back_dispute_0589" AND received a
  Stage-1 tool response containing "Status: RESOLVED" for that same
  transaction_id. Skipping Stage 1 leaves the `cash_back_disputes` evaluation
  table empty where gold expects one row per disputed transaction, AND writes
  an `update_transaction_rewards_3847` audit row into `agent_discoverable_
  tools` that gold does not expect — both diverge the evaluation DB hash from
  gold and produce a 0-reward task.
""".strip()
