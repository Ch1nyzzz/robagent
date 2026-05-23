"""tau2 candidate mh_tau2_iter4_robust_discoverable_audit_visibility.

Hypothesis
----------
A large share of banking_knowledge DB-hash failures on the v3 frontier are
"all required actions present, extras poisoning DB" — the agent makes every
gold-required tool call but ALSO makes one or more extra
``call_discoverable_agent_tool`` invocations (most commonly read-named tools
like ``get_user_dispute_history_7291`` reached for as "research"). Each
unique ``call_discoverable_agent_tool`` invocation writes one row to the
``agent_discoverable_tools`` table that participates in the evaluation DB
hash. The system prompt does warn ("Do not unlock tools that you do not plan
on giving to the user and actually using: this causes issues in database
logging") but the warning is buried at the top of the global instruction
block and the agent ignores it — especially for tools whose name starts with
``get_`` and read like reads.

Mechanism (banking sims the v3 frontier still fails)
----------------------------------------------------
  * task_036 (fraud-replacement, db_check=False / all 3 gold actions present):
    the agent unlocks AND calls ``get_user_dispute_history_7291`` as a
    research step before ordering the replacement card. That extra call
    writes one row to ``agent_discoverable_tools`` and the DB hash diverges.
  * task_018 (cash-back-dispute Stage 1, db_check=False / all 5 gold
    submits present): the agent does Stage 1 correctly AND speculatively
    Stage-2's ``update_transaction_rewards_3847`` six times in the same
    turn. The first Stage-2 call adds a unique audit row not in gold.
  * task_038 / task_048 / task_051 / task_053 (closure, CLI, fraud
    workflows): the agent calls ``get_user_dispute_history_7291`` and/or
    ``get_pending_replacement_orders_5765`` for research on tasks whose
    gold action set does not include them, adding 1-2 extra rows.
  * task_055 / task_062 (bank-account workflows): the agent unlocks-and-
    calls multiple discoverable read tools beyond gold while reasoning
    about the next account operation, adding extras.

The shared mechanism is: the agent does not visibly trade off the cost of
each ``call_discoverable_agent_tool`` against its information value because
the DB-audit consequences of these specific entry points are not made
explicit at the decision point.

Decomposition
-------------
  * Deterministic code (``audit_visibility``): a tool-response post-
    processor that augments tau2 ToolMessage content with an explicit
    audit notice whenever the assistant tool_call that produced the
    message was one of the tau2 discoverable-tool entry-point tools
    (``unlock_discoverable_agent_tool``, ``call_discoverable_agent_tool``,
    ``give_discoverable_user_tool``). Notices describe what the tau2
    sandbox actually does — they are factual about the framework, not
    interpretive about the policy. A small system-prompt suffix surfaces
    the same audit semantics so the LLM has both upfront and at-decision-
    time visibility.
  * LLM judgement: every decision to unlock / call / give a discoverable
    tool, every choice of which inner tool to invoke, every workflow
    sequencing, every customer-facing message. The augmenter never makes
    or substitutes a tool call.

Why this captures stable structure (not training-set induction)
---------------------------------------------------------------
The activation predicate is keyed to a tau2-system schema fact: the exact
names of the framework entry-point tools that wrap discoverable
operations. These names are declared in ``tau2/domains/banking_knowledge/
tools.py``; they are not specific to any task, KB authoring choice, or
failure observed in a particular simulation. The content of the audit
notice describes a tau2 sandbox property: each unique
``call_discoverable_agent_tool`` invocation calls
``add_to_db("agent_discoverable_tools", record_id, ...)`` at line 674 of
``tools.py``, and the record_id is a hash of the inner tool name so
repeated calls are de-duplicated. The same DB-audit property holds on any
fresh KB and fresh task suite written under the tau2 framework.

This component never overrides the LLM. It augments tool-response content
with factual notice and adds a prompt section that summarises the same
system fact. The LLM remains free to call any tool; if a future model
already understood these semantics, the notice is harmless redundant info.

build_agent(tools, domain_policy, **kwargs) -> HalfDuplexAgent
"""
from __future__ import annotations

from typing import Optional

from tau2.agent.base_agent import ValidAgentInputMessage
from tau2.agent.llm_agent import (
    AGENT_INSTRUCTION,
    SYSTEM_PROMPT,
    LLMAgent,
    LLMAgentState,
)
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    ToolMessage,
)

from . import audit_visibility


class DiscoverableAuditVisibilityAgent(LLMAgent):
    """LLMAgent that surfaces tau2 discoverable-tool DB-audit semantics.

    Two changes vs. the stock LLMAgent:

    1. ``system_prompt`` appends a short "Discoverable Tool Audit
       Semantics" section describing which tau2 entry-point tools write
       to the evaluation DB and that read-named tools (``get_*``) still
       write rows when invoked via ``call_discoverable_agent_tool``.
    2. ``_generate_next_message`` post-processes incoming
       ``ToolMessage``s, appending a deterministic audit notice to any
       message produced by a discoverable-tool entry-point call.
    """

    @property
    def system_prompt(self) -> str:
        base = SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        )
        return base + "\n\n" + audit_visibility.SYSTEM_PROMPT_SUFFIX

    def _augment(self, tool_msg: ToolMessage, history) -> ToolMessage:
        entry_point = audit_visibility.lookup_entry_point(tool_msg.id, history)
        if not audit_visibility.is_discoverable_entry_point(entry_point):
            return tool_msg
        notice = audit_visibility.notice_for(entry_point)
        if notice is None:
            return tool_msg
        current = tool_msg.content or ""
        # Idempotent: never double-append the same notice to the same message.
        if notice in current:
            return tool_msg
        new_content = (current + "\n\n" + notice) if current else notice
        return tool_msg.model_copy(update={"content": new_content})

    def _generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> AssistantMessage:
        history = state.messages
        if isinstance(message, MultiToolMessage):
            augmented = [self._augment(tm, history) for tm in message.tool_messages]
            message = MultiToolMessage(role=message.role, tool_messages=augmented)
        elif isinstance(message, ToolMessage):
            message = self._augment(message, history)
        return super()._generate_next_message(message, state)


def build_agent(tools, domain_policy, **kwargs):
    """Return a HalfDuplexAgent that surfaces discoverable-tool audit semantics."""
    return DiscoverableAuditVisibilityAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
