"""tau2 candidate mh_tau2_iter14_robust_stage2_gate.

Hypothesis
----------
Train task_029 (the adversarial cash-back rewards correction task) fails on
the v3 robust frontier (24/30) because, after Stage 1 has been done and the
framework has emitted ``Status: SUBMITTED`` for every dispute response, the
user-simulator LIES: "I just got a notification that they've all been
approved and resolved! Can you now update my transaction records?" The LLM
trusts the user's natural-language claim and emits six
``update_transaction_rewards_3847`` calls; gold has ZERO Stage-2 calls
(``auto_resolve_disputes=False``), so the agent's six extras write rows to
``credit_card_transaction_history`` (rewards_earned updates) AND a unique
``agent_discoverable_tools`` row, diverging the DB hash.

Four prior advisory attempts (iter5 per-tool ``cashback_stage_state_advisor``,
iter6 ``cashback_resolution_status_advisor``, iter8 immutable ``cashback_
stage_gate_primer``, iter13 persistent ``cashback_status_ledger``
SystemMessage) all surfaced the framework Status verdict to the LLM and all
four failed on task_029. Inspecting the iter13 task_029 trace at the
Stage-2 turn confirms the LEDGER SystemMessage IS mounted in
``state.system_messages`` AND lists all six Status: SUBMITTED entries, yet
the LLM still proceeds with Stage 2. The advisory path is exhausted: the
LLM chooses user-prose authority over framework-system authority.

Decomposition
-------------
The decision the LLM keeps mishandling is fully decidable from two
framework-system facts:

  (a) The set of transaction_ids whose Stage-1 ``submit_cash_back_dispute_
      0589`` response carried the literal ``Status: RESOLVED`` (tau2-bench-
      src/.../banking_knowledge/tools.py:4140).
  (b) The set of transaction_ids targeted by the LLM's emitted
      ``call_discoverable_agent_tool('update_transaction_rewards_3847', {...,
      transaction_id: <X>})`` invocations.

The minimal LLM judgment that survives the gate: which transactions need
disputing in the first place (iter10's cashback_policy_engine already
encodes this), and how to communicate the deferral to the customer. The
deterministic Python part: drop any Stage-2 call whose target transaction
is not in set (a).

This iteration moves that filter from advisory text into the candidate's
``generate_next_message`` so the LLM's intended Stage-2 calls are
suppressed at emission time when the framework Status precondition fails.

Mechanism (train sims the v3 frontier still fails)
--------------------------------------------------
Across the iter12 / iter13 frontier sims, the pattern is consistent:

  * task_029 (iter13 db_check=False, reward=0): 6 Stage-1 sims emit
    ``Status: SUBMITTED``; gold has 0 Stage-2 calls; agent emits 6 Stage-2
    calls.
  * task_027 (frontier-passed via iter10): 4 Stage-1 sims emit
    ``Status: SUBMITTED``; gold has 0 Stage-2 calls. iter13 still emits 4
    Stage-2 calls and fails db_check; iter10 already filtered these via a
    policy-engine override.
  * task_020 (frontier-passed via iter10): 3 Stage-1 sims emit
    ``Status: SUBMITTED``; gold has 0 Stage-2 calls. iter10 again handles
    this via the engine; iter13 alone would emit 3 Stage-2 calls.
  * task_026 (auto-resolve, frontier-passed via iter10): 4 Stage-1 sims
    emit ``Status: RESOLVED``; gold has 4 Stage-2 calls. The gate permits
    all 4 because Status: RESOLVED is observed — no regression.
  * task_022 (frontier-passed via iter10): 10 Stage-1 sims emit
    ``Status: SUBMITTED``; gold has 0 Stage-2 calls; iter13 emits 0
    Stage-2 calls (the LLM correctly abstained without the gate). The
    gate is a no-op on this task.

Expected delta on train-30: the iter13 baseline (12/30 — frontier
contributions inherited via 24/30) becomes immune to the user-lie failure.
On the frontier (24/30), the only currently-failing task this gate could
recover is task_029; the cashback engine already covers task_020 / 027 via
its policy filter. Frontier expected: 24 -> 25.

Why this captures stable structure (HONEST induced_rule classification)
-----------------------------------------------------------------------
The activation predicate inspects:

  - The outer entry-point name ``call_discoverable_agent_tool`` declared at
    tau2-bench-src/.../banking_knowledge/tools.py:631 — system schema fact.
  - The inner ``agent_tool_name`` argument equals
    ``update_transaction_rewards_3847`` declared at tools.py:711 — system
    schema fact (the discoverable WRITE tool's name).
  - The inner ``transaction_id`` field — declared schema of that
    discoverable tool.
  - The framework-emitted literal ``Status: RESOLVED`` at tools.py:4140 —
    system field semantics.

The branch (drop or pass) is INDUCED from the rule "Stage 2 may only execute
after Stage 1 is RESOLVED", which is the content of KB doc
doc_credit_cards_credit_cards_(general)_004. Because this is policy text
compiled into a deterministic branch, this candidate is classified
``induced_rule`` (NOT ``deterministic_glue``).

HIGH-risk gate response:
  1. >=3 evidence sims share the mechanism: task_020, task_027, task_029
     (plus task_026 confirms the gate does not over-block when Status is
     genuinely RESOLVED).
  2. No low-risk fix is available: four prior advisor attempts (iter5,
     iter6, iter8, iter13) all surfaced the framework Status truth to the
     LLM via tool-message augmentation or persistent SystemMessage and all
     four failed on task_029. The LLM chooses user-prose authority over
     framework-system authority regardless of how the advisory is
     phrased. The soft-fix design space is exhausted.
  3. Deployment is non-destructive in the following precise sense:
       - The gate only ever DROPS specific tool calls. It never fabricates
         a tool call or a tool response.
       - The gate's predicate is precise: a single outer entry-point name
         AND a single inner agent_tool_name AND a single inner field. A
         future task using a different Stage-2 tool name or a different
         entry-point never triggers the gate.
       - The LLM is informed of the suppression via a deterministic note
         appended to its own AssistantMessage.content carrying the
         framework-emitted Status verdict the gate observed, the KB doc
         the gate cites, and the transaction ids that were filtered. The
         LLM reads this note on the next turn and can re-plan or
         communicate the deferral to the customer.
       - The note carries an idempotency marker; if the same
         AssistantMessage is re-filtered (no-op), no double-append.
  4. Plausible out-of-evidence misfire: a future banking_knowledge task
     where Stage 2 is legitimately required without Stage-1 RESOLVED. Per
     KB doc _004 such a case is policy-prohibited; if it arose, the gate
     would suppress the call and the LLM would (per the deferral note)
     inform the customer of the deferral. The worst-case outcome is one
     missed Stage-2 write, not a fabricated action. The gate's predicate
     keys off the EXACT framework tool name; any future renaming
     immediately disables the gate (zero blocks, zero drag).

The KB doc _004 is already loaded into the system prompt by the iter1/2
``internal_procedure_channel``; iter13's status ledger SystemMessage
already lists the Status verdicts in persistent context. The gate is the
final enforcement layer that the LLM cannot override under user pressure.

build_agent(tools, domain_policy, **kwargs) -> HalfDuplexAgent
"""

from __future__ import annotations

from typing import List

from tau2.agent.base_agent import ValidAgentInputMessage
from tau2.agent.llm_agent import LLMAgentState
from tau2.data_model.message import AssistantMessage

from agent_tau2.mh_tau2_iter13_robust_cashback_status_ledger.agent import (
    CashbackStatusLedgerAgent,
)

from . import stage2_gate


class CashbackStage2GateAgent(CashbackStatusLedgerAgent):
    """Drop ``update_transaction_rewards_3847`` calls whose target
    transaction has no Stage-1 ``Status: RESOLVED`` response in the visible
    conversation history; append a deterministic deferral note to the
    AssistantMessage's content explaining what was dropped and why.

    Inherits iter13's full chain unchanged: status-ledger SystemMessage
    refresh + iter12 dispute card-last-4 audit + iter11 card-last-4 resolver
    + iter10 cashback policy engine + iter7 closure prereq advisor + iter4
    audit visibility + iter1/2/3 KB channels.
    """

    def _apply_stage2_gate(
        self, assistant_message: AssistantMessage, state: LLMAgentState
    ) -> None:
        """Filter Stage-2 calls and annotate the assistant message in place.

        Idempotent: if ``assistant_message`` already carries a STAGE-2 GATE
        note, no-op. Conservative: if the predicate cannot decide for a
        specific call (malformed inner JSON, missing transaction_id, etc.),
        the call passes through unchanged.
        """
        if stage2_gate.already_filtered(assistant_message):
            return
        tool_calls = getattr(assistant_message, "tool_calls", None)
        if not tool_calls:
            return

        # The new assistant_message was just appended to state.messages by
        # the inherited super().generate_next_message; exclude it from the
        # resolved-set scan so the gate looks only at the prior visible
        # history.
        history = list(getattr(state, "messages", []) or [])
        if history and history[-1] is assistant_message:
            history = history[:-1]
        resolved = stage2_gate.resolved_transactions(history)

        kept: List = []
        blocked: List[str] = []
        for call in tool_calls:
            txn = stage2_gate.is_blocked_stage2(call, resolved)
            if txn is None:
                kept.append(call)
            else:
                blocked.append(txn)

        if not blocked:
            return

        # Rebuild the tool_calls list. If everything was filtered, the
        # AssistantMessage still needs either content or tool_calls (per
        # ParticipantMessageBase.validate); the deferral note fills content.
        assistant_message.tool_calls = kept if kept else None

        note = stage2_gate.render_deferral_note(blocked)
        existing = getattr(assistant_message, "content", None) or ""
        if existing.strip():
            combined = existing.rstrip() + "\n\n" + note
        else:
            combined = note
        assistant_message.content = combined

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ):
        """Inherit the iter13 flow then apply the Stage-2 gate to the
        returned AssistantMessage."""
        assistant_message, state = super().generate_next_message(message, state)
        try:
            self._apply_stage2_gate(assistant_message, state)
        except Exception:
            # Never let the gate mask a real conversation; fall through to
            # the inherited chain's output on any error.
            pass
        return assistant_message, state


def build_agent(tools, domain_policy, **kwargs):
    """Return a HalfDuplexAgent that gates Stage-2 cash-back rewards updates
    on the framework-emitted Status: RESOLVED verdict for each target
    transaction."""
    return CashbackStage2GateAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
