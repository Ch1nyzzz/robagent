"""tau2 candidate mh_tau2_iter5_robust_cashback_stage_state_advisor.

Hypothesis
----------
Cash-back-rewards correction tasks (train task_020 and task_022, with
related patterns in task_026, task_027, task_029) fail because the agent
skips Stage 1 of the two-stage workflow described in KB doc
credit_cards_(general)_003 / _004. Specifically the agent invokes
``unlock_discoverable_agent_tool('update_transaction_rewards_3847')`` and
immediately follows with ``call_discoverable_agent_tool(
'update_transaction_rewards_3847', ...)`` for each transaction, without
ever issuing ``give_discoverable_user_tool('submit_cash_back_dispute_0589')``.

Mechanism (banking sims the v3 frontier still fails)
----------------------------------------------------
  * task_020 (iter4 sim, db_check=False): agent diagnoses cash-back
    discrepancies on the user's Business Silver Rewards Card transactions
    and proceeds straight to ``unlock_discoverable_agent_tool(
    'update_transaction_rewards_3847')`` + five ``call_discoverable_agent_tool``
    invocations. Zero ``give_discoverable_user_tool('submit_cash_back_dispute_0589')``
    calls. The ``cash_back_disputes`` table has 0 rows where gold has 5.
  * task_022 (iter4 sim, db_check=False): same pattern, 9 corrections,
    zero Stage-1 gives. ``cash_back_disputes`` is empty in the agent's DB
    where gold has 9 rows.

The internal_procedure_channel (iter1 and iter2) already loads KB docs
003 and 004 into the system prompt — the policy text *is* in the LLM's
context — but the LLM does not apply the two-stage rule at the particular
decision point of "I am about to invoke Stage 2." A just-in-time, decision-
point state report — appended to the tool response for the Stage-2 unlock
and for the Stage-2 call — gives the LLM concrete, mechanical visibility
into Stage-1 progress.

Decomposition
-------------
  * Deterministic code (``stage_state``): a tool-response post-processor
    that, when the producing assistant tool_call is
    ``unlock_discoverable_agent_tool`` or ``call_discoverable_agent_tool``
    with inner ``agent_tool_name='update_transaction_rewards_3847'``,
    appends a Stage State advisory naming the two-stage workflow (citing
    the existing KB doc ids) and reporting the mechanical count of
    Stage-1 give calls observed in the agent's message history.
  * LLM judgement: which transactions are eligible for correction, the
    correct points calculation per transaction, whether and when to do
    Stage 1, when to consider Stage 1 "resolved" by the user's reply,
    every customer-facing message. The advisor never makes or substitutes
    a tool call.

Why this captures stable structure (not training-set induction)
---------------------------------------------------------------
Two layers of structure, both independent of the failed simulations:

  1. Tool-schema facts: the entry-point names
     (``unlock_discoverable_agent_tool``, ``call_discoverable_agent_tool``,
     ``give_discoverable_user_tool``) are declared in
     ``tau2/domains/banking_knowledge/tools.py``. The Stage-2 tool name
     (``update_transaction_rewards_3847``) is declared in tools.py too
     (line ~4151 area) and is a tau2-system fact. The Stage-1 user tool
     name (``submit_cash_back_dispute_0589``) is declared at tools.py
     line 4106 area.

  2. KB doc facts: the two-stage workflow itself is the entire content
     of doc_credit_cards_credit_cards_(general)_003 ("Submitting a Cash
     Back Dispute (Internal)") and doc_credit_cards_credit_cards_
     (general)_004 ("Applying Resolved Cash Back Dispute Corrections
     (Internal)"). The advisor cites these docs by id; it does not
     reproduce or branch on their text and it does not encode any rule
     beyond "Stage 2 follows Stage 1."

The Stage-1-give-count is a mechanical reduction over the agent's
observable tool_call history — no policy text, no keyword guessing on
user content. The advisor never modifies the LLM's tool call: it only
appends an advisory string to a tool response. The LLM is free to
proceed to Stage 2 regardless of the observed count, or to back out and
do Stage 1 first.

build_agent(tools, domain_policy, **kwargs) -> HalfDuplexAgent
"""

from __future__ import annotations

from tau2.agent.base_agent import ValidAgentInputMessage
from tau2.agent.llm_agent import LLMAgent, LLMAgentState
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    ToolMessage,
)

from . import stage_state


_ADVISORY_TAG = "[STAGE STATE]"


class CashbackStageStateAdvisorAgent(LLMAgent):
    """LLMAgent that appends a Stage-1/Stage-2 advisory to the tool response
    for the Stage-2 cash-back-rewards entry-point calls.

    The advisor activates only when the producing assistant tool_call is
    ``unlock_discoverable_agent_tool`` or ``call_discoverable_agent_tool``
    targeting the Stage-2 discoverable tool
    ``update_transaction_rewards_3847``. In every other case the agent
    behaves identically to the stock ``LLMAgent``.
    """

    def _augment_tool_message(
        self, tool_msg: ToolMessage, history
    ) -> ToolMessage:
        producing = stage_state.lookup_producing_tool_call(tool_msg.id, history)
        if producing is None:
            return tool_msg
        if not stage_state.is_stage_2_entry_point_call(producing):
            return tool_msg
        give_count = stage_state.count_stage_1_give_calls(history)
        advisory = stage_state.render_state_advisory(give_count)
        current = tool_msg.content or ""
        # Idempotent: never double-append the same advisory tag in the same
        # tool response, even if augmentation is invoked twice somehow.
        if _ADVISORY_TAG in current:
            return tool_msg
        new_content = (current + "\n\n" + advisory) if current else advisory
        return tool_msg.model_copy(update={"content": new_content})

    def _generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> AssistantMessage:
        history = state.messages
        if isinstance(message, MultiToolMessage):
            augmented = [self._augment_tool_message(tm, history) for tm in message.tool_messages]
            message = MultiToolMessage(role=message.role, tool_messages=augmented)
        elif isinstance(message, ToolMessage):
            message = self._augment_tool_message(message, history)
        return super()._generate_next_message(message, state)


def build_agent(tools, domain_policy, **kwargs):
    """Return a HalfDuplexAgent that surfaces cash-back Stage-1 state at the
    Stage-2 entry-point decision."""
    return CashbackStageStateAdvisorAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
