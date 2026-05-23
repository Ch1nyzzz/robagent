"""tau2 candidate mh_tau2_iter8_robust_cashback_stage_gate_primer.

Hypothesis
----------
On the v3 robust frontier (19/30) the three remaining cash-back rewards
correction tasks task_022, task_027, task_029 — and the cash-back leg of
the adversarial pair task_026 — all fail by the same mechanism: the
agent jumps directly to Stage 2 (``unlock_discoverable_agent_tool``
followed by a batch of ``call_discoverable_agent_tool('update_
transaction_rewards_3847', ...)``) without ever issuing the Stage-1
``give_discoverable_user_tool('submit_cash_back_dispute_0589')``. The
``cash_back_disputes`` evaluation table ends up empty where gold expects
one row per disputed transaction, the ``agent_discoverable_tools`` audit
table picks up an unmatched Stage-2 row, and ``db_check`` is False.

The iter5 ``cashback_stage_state_advisor`` runtime advisor fires at the
Stage-2 unlock and call moments, by which point the LLM has already
committed to a single-turn batched Stage-2 plan in the very next
assistant message — the advisor cannot land on the previous assistant
message because it is appended to the unlock tool response after the
plan was formed. The intervention point that is missing is *plan-
formation time*, which means a durable, compact gate inside the system
prompt itself.

This candidate adds that primer: a static, deterministic STAGE GATE
block restating the two-stage cash-back workflow defined by KB docs
``credit_cards_(general)_003`` and ``_004`` and anchored to the
framework-emitted ``Status: RESOLVED`` / ``Status: SUBMITTED``
substrings declared at tools.py lines 4140 / 4149. The block is
appended verbatim to the system prompt on every conversation. The LLM
remains the sole decision-maker about which tool to call next; the
primer never makes, modifies, or suppresses any tool call.

This candidate extends iter5's ``CashbackStageStateAdvisorAgent`` so
that the runtime decision-point advisor still fires. The combination
gives the LLM:

  * Plan-formation guidance — the STAGE GATE primer in the system
    prompt, named, structured, and anchored to KB doc ids.
  * Decision-point guidance — iter5's Stage State advisor at the
    Stage-2 unlock / call (inherited unchanged).

Mechanism (banking sims the v3 frontier still fails on cash-back)
-----------------------------------------------------------------
  * task_022 (iter7 sim, db_check=False): 11 gold actions all MISS.
    Agent batches 10 ``update_transaction_rewards_3847`` calls in a
    single assistant turn after the Stage-2 unlock; zero Stage-1
    gives anywhere in the conversation. ``cash_back_disputes`` is
    empty in the agent's DB where gold has 10 rows.
  * task_027 (iter7 sim, db_check=False): 5 gold actions MISS.
    Agent goes straight to Stage 2; zero Stage-1 gives.
  * task_029 (iter7 sim, db_check=False): 7 gold actions MISS.
    Same pattern.
  * task_026 (iter7 sim, db_check=False): 6 gold actions MISS.
    The adversarial-pair partner of task_027: identical agent-visible
    input but gold has BOTH Stage 1 AND Stage 2 (auto-resolve case).
    Agent again skips Stage 1, then Stage 2 writes a row that gold
    expects only if Stage 1 actually fired.

Decomposition
-------------
  * Deterministic code (``stage_gate_primer``): an immutable static
    string restating the two-stage workflow with explicit
    preconditions, citing the KB doc ids and the framework's
    ``Status:`` substring contract. Appended to the system prompt.
  * LLM judgement: which transactions have suspected cash-back
    discrepancies, the correct points calculation per transaction,
    when the customer's status confirmation is sufficient, every
    customer-facing message, and whether to follow the primer's
    sequencing or override it. The primer never selects, modifies, or
    suppresses a tool call.

Why this captures stable structure (not training-set induction)
---------------------------------------------------------------
Three layers, all independent of the failed simulations:

  1. KB-doc facts. The two-stage ordering ("After a cash back dispute
     is resolved and approved, you must update the affected
     transaction(s) with the correct rewards value") is the entire
     content of doc credit_cards_(general)_004. The user-tool give of
     submit_cash_back_dispute_0589 is the entire content of doc
     credit_cards_(general)_003 ("Tool to provide to the user:
     submit_cash_back_dispute_0589"). The primer cites both docs by
     id; it does not invent ordering.

  2. Tool-schema facts. The entry-point names
     (``give_discoverable_user_tool``, ``unlock_discoverable_agent_
     tool``, ``call_discoverable_agent_tool``) and the discoverable
     tool names (``submit_cash_back_dispute_0589``,
     ``update_transaction_rewards_3847``) are declared in
     ``tau2-bench-src/src/tau2/domains/banking_knowledge/tools.py``.
     These are tau2-system facts; they would hold on a fresh
     banking_knowledge task suite written against the same tools.py.

  3. Framework return-string facts. The literal ``Status: RESOLVED``
     and ``Status: SUBMITTED`` substrings are emitted by the framework
     into the Stage-1 tool response at tools.py lines 4140 / 4149.
     The primer reads these as the gate signal — not the agent's
     content choices, not the user's natural-language reply.

Deployment is non-destructive. The primer is text appended to the
system prompt; the LLM has full discretion to follow or override. No
tool call is suppressed, rewritten, or fabricated by this candidate.

The HIGH-risk gate
------------------
This candidate is classified ``induced_rule`` because the IF-THEN
gate structure ("Stage 2 only if Stage 1 returned RESOLVED") is the
builder's compilation of policy doc text into a directive form. Per
the SKILL HIGH-risk gate:

  (1) >=3 evidence simulations share the failure mechanism:
      task_022, task_027, task_029 (all three on the v3 frontier),
      plus the structurally identical task_026.

  (2) No low-risk fix is available: the iter1/iter2 channel loads
      the full docs (low-risk, channel) — iter1 reached frontier
      9/30 but does NOT fix tasks 022/027/029. The iter5 runtime
      advisor (low-risk, deterministic_glue) reached frontier 18/30
      and fixed task_004 but does NOT fix tasks 022/027/029.
      iter6's extension regressed. The shared cause is that both
      the channel and the runtime advisor land too late — the LLM
      has already committed to a Stage-2 plan by the time it reads
      either signal. A compact directive at plan-formation time is
      a different intervention point that the lower-risk classes
      cannot reach without compromising their structural anchor.

  (3) Non-destructive deployment: the primer is system-prompt text
      only. The LLM remains the sole decision-maker on every tool
      call. The primer cannot silently override the LLM's output.

Out-of-evidence considerations
------------------------------
A future task may legitimately invoke ``update_transaction_rewards_
3847`` for a non-cash-back reason (e.g., a service-recovery one-off
credit). The primer's framing is scoped to "cash-back rewards
correction": "When the customer asks you to identify, correct,
refund, or adjust cash-back rewards on one or more credit-card
transactions, this workflow is mandatory." Non-cash-back rewards
adjustments are outside the primer's stated scope; the LLM should
recognise this and bypass the gate. If the LLM misapplies the gate to
a non-cash-back rewards adjustment, the worst case is an unneeded
``give_discoverable_user_tool('submit_cash_back_dispute_0589')`` call
— a single audit row in ``user_discoverable_tools``. Per the
tau2-bench banking_knowledge convention only unique tool names are
recorded, so an unneeded give would be at most one extra row, not a
fan-out.

If the tau2 framework renames any of {give_discoverable_user_tool,
unlock_discoverable_agent_tool, call_discoverable_agent_tool,
submit_cash_back_dispute_0589, update_transaction_rewards_3847} the
primer text becomes stale prose; the LLM has the renamed tools in its
tool list and can override the primer's specific tool names.

build_agent(tools, domain_policy, **kwargs) -> HalfDuplexAgent
"""
from __future__ import annotations

from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT

from agent_tau2.mh_tau2_iter5_robust_cashback_stage_state_advisor.agent import (
    CashbackStageStateAdvisorAgent,
)

from . import stage_gate_primer


class CashbackStageGatePrimerAgent(CashbackStageStateAdvisorAgent):
    """LLMAgent that appends the cash-back STAGE GATE primer to the system
    prompt, in addition to inheriting iter5's Stage State runtime advisor.

    Both interventions point at the same failure mechanism — the agent
    jumping straight to Stage 2 of the cash-back-rewards-correction
    workflow without first doing Stage 1 — at different points in the
    conversation:

      * the primer lands at plan-formation time (system prompt, seen
        before any user message);
      * the inherited Stage State advisor lands at decision-point
        time (appended to the tool response of the Stage-2 unlock /
        call).

    Neither modifies or suppresses any tool call. The LLM remains the
    sole decision-maker about which tool to call next.
    """

    @property
    def system_prompt(self) -> str:
        base = SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        )
        return base + "\n\n" + stage_gate_primer.PRIMER


def build_agent(tools, domain_policy, **kwargs):
    """Return a HalfDuplexAgent that surfaces the cash-back STAGE GATE primer
    in the system prompt while inheriting iter5's runtime Stage State
    advisor for the Stage-2 entry-point calls."""
    return CashbackStageGatePrimerAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
