"""tau2 candidate mh_tau2_iter9_robust_provisional_credit_checklist.

Hypothesis
----------
The v3 frontier (19/30) still fails task_038 (and would mis-handle any
future credit-card transaction-dispute task hitting the same edge) because
the LLM mis-applies KB doc credit_cards_(general)_015's provisional-credit
eligibility rule -- specifically criterion 5, which requires
``contacted_merchant=true`` for any non-fraud dispute_reason
(``duplicate_charge``, ``goods_services_not_received``, etc.). The doc is
already loaded into the system prompt by the iter1/iter2
``internal_procedure_channel`` plugin, yet across the visible cross-track
evidence the LLM keeps reaching the same wrong value at filing time:

  * iter7 robust ``closure_cli_prereq_advisor`` task_038 dispute #3
    (duplicate_charge + contacted_merchant=false): agent eligible=true,
    gold eligible=false.
  * iter16 baseline ``card_application_facilitator`` task_038 dispute #3
    (duplicate_charge + contacted_merchant=false): agent eligible=true,
    gold eligible=false.
  * iter18 baseline ``closure_procedure_reference`` task_038 dispute #3
    (same pattern, same direction).
  * iter20 baseline ``closure_apply_dual_workflow`` task_038 has the same
    LLM mis-application showing up on its filed disputes.

Across these ≥3 sims the failure mechanism is identical: the LLM sees the
doc text but does not apply criterion 5 at the decision point of filling
``eligible_for_provisional_credit``. The eligibility value is recorded
verbatim into the ``transaction_disputes`` row and participates in the
DB hash, so a single mis-set field can break the task on a ``DB`` reward
basis (which task_038 is, per the simulation record).

The fix is to re-surface the doc's eligibility criteria as a decision-
point checklist at the moment the agent unlocks or calls
``file_credit_card_transaction_dispute_4829`` -- the same JIT advisory
pattern that the iter7 prereq advisor uses for closure / CLI workflow
tools.

Mechanism
---------
A dedicated post-processor on incoming ToolMessages: when the producing
assistant tool_call is ``unlock_discoverable_agent_tool`` or
``call_discoverable_agent_tool`` AND its inner ``agent_tool_name`` equals
``file_credit_card_transaction_dispute_4829``, append a deterministic
``[PROVISIONAL CREDIT CHECKLIST]`` advisory to the tool response. The
advisory cites KB doc ``credit_cards_(general)_015`` by id and lists the
five eligibility criteria literally as written, plus the doc's own
"NOT Eligible Scenarios" worked-example bullets, plus a single literal
note that criteria 2 (reason category) and 5 (non-fraud merchant contact)
are decidable from the call's OWN arguments alone.

Decomposition
-------------
  * Deterministic code (``provisional_credit_advisor``): a tool-response
    post-processor keyed solely on the tau2-framework entry-point name
    plus the inner ``agent_tool_name`` argument. The advisor text is
    static (no branch on call args, no rule branching on policy beyond
    the literal doc transcription) and idempotent via the
    ``[PROVISIONAL CREDIT CHECKLIST]`` tag.
  * LLM judgement: every value in the dispute call (including
    ``eligible_for_provisional_credit``). The advisor never makes,
    substitutes, suppresses, or rewrites a tool call; it appends
    informational text and lets the LLM decide.

Why this captures stable structure (not training-set induction)
---------------------------------------------------------------
Two layers of structure, both independent of the failed simulations:

  1. Tool-schema facts: the framework entry-point names
     ``unlock_discoverable_agent_tool`` (declared at
     ``tau2-bench-src/src/tau2/domains/banking_knowledge/tools.py`` line
     ~590) and ``call_discoverable_agent_tool`` (line ~631) plus the
     inner ``file_credit_card_transaction_dispute_4829`` discoverable
     WRITE tool (declared at line ~787 with
     ``@is_discoverable_tool(ToolType.WRITE)``) are tau2 system facts.
     On a fresh banking_knowledge task suite written against the same
     tools.py, the predicate would activate on the same set of tool
     names; nothing about the predicate references task ids, customer
     names, transaction ids, or evidence-sim values.

  2. KB-doc facts: the eligibility criteria are a verbatim transcription
     of doc credit_cards_(general)_015's "Eligibility Criteria" and
     "NOT Eligible Scenarios" sections. The advisor cites the doc by id
     and re-surfaces the policy text at the decision moment; it does not
     interpret, summarize, or compile policy into branches that override
     the LLM.

This is the same JIT advisory pattern that the iter7
``closure_cli_prereq_advisor`` plugin uses (citing logistics_002 / _003 /
_007 at closure / retention / CLI workflow tools). The plugin class is
``deterministic_glue`` for the same reason: the advisor mechanically
surfaces stable system-level structure (tool schema + KB doc id) at the
decision point, never branches on doc text beyond literal citation, and
never overrides the LLM.

build_agent(tools, domain_policy, **kwargs) -> HalfDuplexAgent
"""

from __future__ import annotations

from tau2.data_model.message import ToolMessage

from agent_tau2.mh_tau2_iter7_robust_closure_cli_prereq_advisor.agent import (
    ClosureCliPrereqAdvisorAgent,
)

from . import provisional_credit_advisor


_ADVISORY_TAG = "[PROVISIONAL CREDIT CHECKLIST]"


class ProvisionalCreditChecklistAgent(ClosureCliPrereqAdvisorAgent):
    """LLMAgent that, in addition to inheriting iter7's closure/retention/CLI
    prereq advisor, appends a provisional-credit eligibility checklist to
    the tool response for any ``file_credit_card_transaction_dispute_4829``
    entry-point call.

    Activation: the producing assistant tool_call is
    ``unlock_discoverable_agent_tool`` or ``call_discoverable_agent_tool``
    AND its inner ``agent_tool_name`` argument equals
    ``file_credit_card_transaction_dispute_4829``. In every other case the
    agent behaves identically to its parent
    ``ClosureCliPrereqAdvisorAgent`` (which in turn behaves identically to
    the stock ``LLMAgent`` outside its own workflow-prereq activation).

    The parent's ``_generate_next_message`` already iterates over incoming
    tool messages and calls ``self._augment_tool_message`` on each one;
    Python's method resolution picks this subclass's override, which first
    delegates to the parent (so the prereq advisory is appended when it
    applies) and then appends the provisional-credit checklist when this
    plugin's predicate fires.
    """

    def _augment_tool_message(
        self, tool_msg: ToolMessage, history
    ) -> ToolMessage:
        # First let the parent's workflow-prereq advisor have its say.
        tool_msg = super()._augment_tool_message(tool_msg, history)

        producing = provisional_credit_advisor.lookup_producing_tool_call(
            tool_msg.id, history
        )
        if producing is None:
            return tool_msg
        if not provisional_credit_advisor.is_dispute_entry_point(producing):
            return tool_msg

        current = tool_msg.content or ""
        # Idempotent: never double-append the same advisory tag in the
        # same tool response even if augmentation is invoked twice.
        if _ADVISORY_TAG in current:
            return tool_msg
        advisory = provisional_credit_advisor.render_advisory()
        new_content = (current + "\n\n" + advisory) if current else advisory
        return tool_msg.model_copy(update={"content": new_content})


def build_agent(tools, domain_policy, **kwargs):
    """Return a HalfDuplexAgent that surfaces the provisional-credit
    eligibility checklist at every
    ``file_credit_card_transaction_dispute_4829`` entry-point call, in
    addition to the inherited closure/retention/CLI prereq advisor."""
    return ProvisionalCreditChecklistAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
