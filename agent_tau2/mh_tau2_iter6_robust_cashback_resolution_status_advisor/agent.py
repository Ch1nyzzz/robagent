"""tau2 candidate mh_tau2_iter6_robust_cashback_resolution_status_advisor.

Hypothesis
----------
Cash-back-rewards correction tasks where the dispute is NOT auto-resolved
fail because, even with iter5's Stage-1-give-count advisory, the agent
still proceeds to Stage 2 (``update_transaction_rewards_3847``) after
Stage 1. The tau2 framework writes an explicit ``Status: SUBMITTED`` line
in every ``submit_cash_back_dispute_0589`` user-tool return when
``task_config.dispute_settings.auto_resolve_disputes`` is false (tools.py
line 4149), and an explicit ``Status: RESOLVED`` line when it is true
(tools.py line 4140). This Status line is a tau2-framework signal — not
a KB policy interpretation — that distinguishes the two workflows:

  * All RESOLVED   -> Stage 2 follows (e.g. task_026).
  * Any SUBMITTED  -> Stage 2 is premature for that transaction (e.g.
                      task_020, task_022, task_027, task_029).

iter5's advisory surfaced ``give_count`` (how many times the agent
extended Stage 1 to the user). It did NOT surface the actual Status
field of the Stage-1 returns. This candidate extends the advisor to
include a deterministic partition of observed Stage-1 returns by
``Status: RESOLVED`` / ``Status: SUBMITTED``. The LLM sees, at the
Stage-2 entry point, exactly how many disputes are RESOLVED vs.
SUBMITTED, and can decline Stage 2 when none are RESOLVED.

Mechanism (banking sims the v3 frontier still fails)
----------------------------------------------------
Across iter5 sims:

  * task_020: 1 give, 13 update_transaction_rewards_3847 calls.
    Every Stage-1 return after the first carries ``Status: SUBMITTED``.
    Gold has zero update_transaction_rewards calls.
  * task_022: same pattern, 9 unwarranted Stage-2 calls.
  * task_027: same pattern, 3+ SUBMITTED returns and unwarranted Stage 2.
  * task_029: same pattern.
  * task_026 (Stage 2 IS warranted): 2+ Stage-1 returns showing
    ``Status: RESOLVED`` — the advisory will report this and not impede.

Decomposition
-------------
  * Deterministic code (``resolution_state``): post-processor over the
    incoming ToolMessage at a Stage-2 entry-point response. Reads only
    tool-call ``name`` / ``arguments`` and tool-message ``content``.
    Computes give_count and (RESOLVED, SUBMITTED, other) partition off
    literal Status substrings declared in tools.py.
  * LLM judgement: which transactions to dispute, which to correct,
    when to give Stage 1, how to phrase the customer message, when to
    end the turn.

Why this captures stable structure (not training-set induction)
---------------------------------------------------------------
Three layers of structure, ALL independent of the failed simulations:

  1. Tool-name facts: ``unlock_discoverable_agent_tool``,
     ``call_discoverable_agent_tool``, ``give_discoverable_user_tool``,
     ``call_discoverable_user_tool``, ``update_transaction_rewards_3847``
     (tools.py line 711), ``submit_cash_back_dispute_0589`` (tools.py
     line 4106) are all framework-declared identifiers in
     ``tau2-bench-src/src/tau2/domains/banking_knowledge/tools.py``.
     They are tau2-system facts, not properties of any task.

  2. Return-string format facts: the framework emits the literal
     substrings ``Status: RESOLVED`` (tools.py line 4140) and
     ``Status: SUBMITTED`` (tools.py line 4149) into the Stage-1
     return body, plus the ``Executed: submit_cash_back_dispute_0589``
     marker (tools.py line 4155). The advisor reads those substrings;
     it does not interpret KB policy text and does not branch on the
     agent's content choices. On a freshly authored task suite using
     the same tools.py, the same literal lines would appear.

  3. KB-doc citations: the advisor names KB doc ids _003 and _004 as
     pointers; it does not encode their text and does not branch on
     them. The two-stage relationship is the entire content of those
     two docs, already preloaded by iter1/iter2's
     ``internal_procedure_channel``.

The advisor never modifies, suppresses, or rewrites the LLM's tool
call; it only appends an advisory string to the tool response. The
LLM is free to proceed to Stage 2 regardless of the observed status
partition.

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

from . import resolution_state


_ADVISORY_TAG = "[STAGE STATE]"


class CashbackResolutionStatusAdvisorAgent(LLMAgent):
    """LLMAgent that appends a Stage-state + resolution-status advisory
    to the tool response for the Stage-2 cash-back-rewards entry-point
    calls.

    Activates only when the producing assistant tool_call is
    ``unlock_discoverable_agent_tool`` or ``call_discoverable_agent_tool``
    targeting ``update_transaction_rewards_3847``. In every other case
    the agent behaves identically to the stock ``LLMAgent``.
    """

    def _augment_tool_message(
        self, tool_msg: ToolMessage, history
    ) -> ToolMessage:
        producing = resolution_state.lookup_producing_tool_call(
            tool_msg.id, history
        )
        if producing is None:
            return tool_msg
        if not resolution_state.is_stage_2_entry_point_call(producing):
            return tool_msg
        give_count = resolution_state.count_stage_1_give_calls(history)
        resolved, submitted, other = (
            resolution_state.count_stage_1_resolution_status(history)
        )
        advisory = resolution_state.render_state_advisory(
            give_count, resolved, submitted, other
        )
        current = tool_msg.content or ""
        # Idempotent: never double-append the same advisory tag.
        if _ADVISORY_TAG in current:
            return tool_msg
        new_content = (current + "\n\n" + advisory) if current else advisory
        return tool_msg.model_copy(update={"content": new_content})

    def _generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> AssistantMessage:
        history = state.messages
        if isinstance(message, MultiToolMessage):
            augmented = [
                self._augment_tool_message(tm, history)
                for tm in message.tool_messages
            ]
            message = MultiToolMessage(role=message.role, tool_messages=augmented)
        elif isinstance(message, ToolMessage):
            message = self._augment_tool_message(message, history)
        return super()._generate_next_message(message, state)


def build_agent(tools, domain_policy, **kwargs):
    """Return a HalfDuplexAgent that surfaces both Stage-1 give count and
    the framework-emitted RESOLVED/SUBMITTED Status partition at the
    Stage-2 cash-back-rewards entry point.
    """
    return CashbackResolutionStatusAdvisorAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
