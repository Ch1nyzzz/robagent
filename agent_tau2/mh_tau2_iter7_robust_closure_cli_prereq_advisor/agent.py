"""tau2 candidate mh_tau2_iter7_robust_closure_cli_prereq_advisor.

Hypothesis
----------
The frontier v3 (18/30) still fails closure / CLI / retention tasks
(task_045, task_048, task_053; related patterns in task_038) because the
agent reaches the workflow-write entry-point (close, log-closure-reason,
apply-statement-credit, submit/approve/deny CLI) but skips the documented
prerequisite read calls — get_user_dispute_history_7291,
get_pending_replacement_orders_5765, get_closure_reason_history_8293,
get_credit_limit_increase_history_4829, get_payment_history_6183 — that
each workflow's Internal KB doc (logistics_002 / _003 / _007) mandates as
Step 1 / Step 2 audit reads. iter4's audit-visibility plugin warns
*against* discoverable extras in general; for these six workflow entry-
points specifically, the listed reads are NOT extras — they are required.
A just-in-time decision-point advisory at workflow unlock / call surfaces
the documented prereqs for THIS workflow (citing the KB doc id) and
reports the observed count of each prereq read in the agent's tool-call
history. The LLM remains in charge of which tool to call next.

Mechanism (banking sims the v3 frontier still fails)
----------------------------------------------------
  * task_053 (CLI + dispute) — agent reaches
    submit_credit_limit_increase_request_7392 and
    approve_credit_limit_increase_5847 after only
    get_credit_limit_increase_history_4829 and get_payment_history_6183.
    The audit shows 0 invocations of get_user_dispute_history_7291 and
    get_pending_replacement_orders_5765; logistics_007 Step 2 mandates
    both. The evaluation DB hash diverges from gold.
  * task_048 (4-card closure) — agent calls close_credit_card_account_7834
    for the first card but never invokes get_user_dispute_history_7291 /
    get_pending_replacement_orders_5765 / get_closure_reason_history_8293
    that logistics_002 / _003 mandate. 15 of 24 expected actions missing.
  * task_045 (retention + statement credit) — gold expects
    get_user_dispute_history_7291, get_pending_replacement_orders_5765,
    get_closure_reason_history_8293 before
    log_credit_card_closure_reason_4521 and apply_statement_credit_8472.
    The agent in iter6's trace abandons the workflow before reaching any
    workflow entry-point; the advisor will only help once the agent does
    reach unlock/call.

Across the full training set, every task whose gold action set uses any
of these six workflow tools also includes the relevant prereq reads
(audited across 23 tasks — 100% presence per workflow tool). This is the
expected outcome from logistics_002 / _003 / _007 being authoritative
procedure docs.

Decomposition
-------------
  * Deterministic code (``prereq_advisor``): a tool-response post-
    processor that, when the producing assistant tool_call is
    ``unlock_discoverable_agent_tool`` or ``call_discoverable_agent_tool``
    AND the inner ``agent_tool_name`` matches one of six workflow tool
    names declared in tau2/domains/banking_knowledge/tools.py, appends a
    deterministic advisory citing the KB doc id and reporting the
    mechanical count of each documented prereq read observed in the
    agent's prior tool calls.
  * LLM judgement: whether the workflow applies to the user's request,
    which account to operate on, what arguments to pass, every customer-
    facing message. The advisor never makes or substitutes a tool call;
    it appends informational text to a tool response and is idempotent
    via the ``[WORKFLOW PREREQS]`` tag.

Why this captures stable structure (not training-set induction)
---------------------------------------------------------------
Two layers of structure, both independent of the failed simulations:

  1. Tool-schema facts: the framework entry-point names
     (``unlock_discoverable_agent_tool``,
     ``call_discoverable_agent_tool``) are declared in
     ``tau2-bench-src/src/tau2/domains/banking_knowledge/tools.py``.
     The six inner workflow-write discoverable tool names
     (``close_credit_card_account_7834``,
     ``log_credit_card_closure_reason_4521``,
     ``apply_statement_credit_8472``,
     ``submit_credit_limit_increase_request_7392``,
     ``approve_credit_limit_increase_5847``,
     ``deny_credit_limit_increase_5848``) are all declared as banking
     discoverable WRITE tools in the same file. These are tau2 system
     facts; on a fresh banking_knowledge task suite written against the
     same tools.py and KB layout, the predicate would activate on the
     same set of entry-points.

  2. KB-doc facts: the workflow → prereq-reads mapping transcribes the
     Internal procedure docs ``logistics_002`` (closure),
     ``logistics_003`` (retention protocol), and ``logistics_007`` (CLI).
     The advisor *cites* these docs by id; it does not reproduce or
     branch on their text, and it does not encode any rule beyond
     "the doc says these reads must precede this write." The docs
     themselves are already in the system prompt via the iter1/iter2
     internal_procedure_channel.

The prereq-count is a mechanical reduction over the agent's observable
tool_call history — no policy interpretation, no keyword guessing on
user content. The advisor never modifies, suppresses, or rewrites the
LLM's tool call: it only appends an advisory string to the tool response.
The LLM is free to proceed without doing any prereq, or to back out and
make the prereq reads first.

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

from . import prereq_advisor


_ADVISORY_TAG = "[WORKFLOW PREREQS]"


class ClosureCliPrereqAdvisorAgent(LLMAgent):
    """LLMAgent that appends a workflow-prereq advisory to the tool response
    for closure / retention / CLI entry-point calls.

    Activation: the producing assistant tool_call is
    ``unlock_discoverable_agent_tool`` or ``call_discoverable_agent_tool``
    AND its inner ``agent_tool_name`` argument is one of six workflow-write
    discoverable tools (see ``prereq_advisor.WORKFLOW_PREREQS``). In every
    other case the agent behaves identically to the stock ``LLMAgent``.
    """

    def _augment_tool_message(
        self, tool_msg: ToolMessage, history
    ) -> ToolMessage:
        producing = prereq_advisor.lookup_producing_tool_call(tool_msg.id, history)
        if producing is None:
            return tool_msg
        workflow_tool = prereq_advisor.workflow_for_tool_call(producing)
        if workflow_tool is None:
            return tool_msg
        current = tool_msg.content or ""
        # Idempotent: never double-append the same advisory tag in the same
        # tool response even if augmentation is invoked twice somehow.
        if _ADVISORY_TAG in current:
            return tool_msg
        advisory = prereq_advisor.render_advisory(workflow_tool, history)
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
    """Return a HalfDuplexAgent that surfaces documented prereq reads at
    closure / retention / CLI workflow entry-point decisions."""
    return ClosureCliPrereqAdvisorAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
