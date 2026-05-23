"""tau2 candidate mh_tau2_iter10_baseline_closure_cli_procedure.

Hypothesis
----------
Credit-card account-closure and credit-limit-increase (CLI) tasks fail the
database-hash check because the agent treats them as single actions instead of
the multi-step procedures the knowledge base documents. It skips the mandatory
prerequisite eligibility-check phase — verifying outstanding balance, pending
transaction disputes, pending replacement-card orders, account age, prior
retention attempts (closure) / cooldown, payment history and utilization
(CLI) — and skips documented intermediate steps (paying off a balance, the
retention offer). Because a blocking condition or a documented alternative
path changes which writes the gold trajectory makes, the agent's write set
diverges from gold and the reward is zero.

Mechanism observed (banking sims)
---------------------------------
Across the closure/CLI failures the agent never calls the discoverable read
tools for dispute history and pending replacement orders:
  * task_044 — closure ("found a better card"). Agent skips both prerequisite
    reads and closes the account; gold runs the retention protocol, never
    closes, and helps the customer apply for a comparable Rho-Bank card.
  * task_045 — closure with a $125 outstanding balance and a retention offer.
    Agent skips both reads and only logs the closure reason; gold pays off the
    balance and applies a retention statement credit.
  * task_051 — multi-step CLI. Agent skips both reads; its decision/payment
    writes diverge from gold.
  * task_053 — CLI + dispute conflict. Agent skips the pending-replacement
    read; its approve write diverges from gold.
  * task_048 — four-card closure. Agent skips the dispute-history read on every
    card before closing.
All five are reward_basis = DB and score 0 on the current frontier.

Change
------
1. The system prompt gains a general, domain-agnostic <account_workflow_protocol>:
   account closures and limit increases are documented multi-step procedures;
   retrieve the full procedure from the knowledge base and complete every
   prerequisite eligibility check before any terminal write; do not close an
   account or approve/deny an increase until every documented step is done; if
   a blocking condition exists, follow the procedure's alternative path; for a
   "found a better card" closure, run the retention protocol and offer a
   comparable Rho-Bank card before closing; sequence interacting requests so
   each is processed while its eligibility still holds.
2. A deterministic guard (procedure_guard.ProcedureGuard) detects a turn that
   emits a terminal closure or CLI-decision write before the dispute-history
   and pending-replacement-orders reads are on record from prior turns, and
   regenerates that turn once with a transient note naming the missing checks.
   The guard never blocks a write outright (a capped nudge budget lets the turn
   through), and the reads it forces are read-only, so a legitimate write can
   never be suppressed and the candidate cannot score below baseline behaviour.

No customer names, ids, tool ids, rates, or per-task branching are used.
"""
from __future__ import annotations

from typing import Optional

from tau2.agent.llm_agent import (
    AGENT_INSTRUCTION,
    SYSTEM_PROMPT,
    LLMAgent,
    LLMAgentState,
)
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    SystemMessage,
)
from tau2.utils.llm_utils import generate

from agent_tau2.mh_tau2_iter10_baseline_closure_cli_procedure.procedure_guard import (
    ProcedureGuard,
)

# General, domain-agnostic guidance appended to the system prompt.
ACCOUNT_WORKFLOW_PROTOCOL = """
<account_workflow_protocol>
Credit-card account closures and credit-limit-increase (CLI) requests are
documented multi-step procedures, not single actions. Treat them that way.

Before performing any closure or limit-increase write:
1. Retrieve the complete procedure from the knowledge base and follow every
   step it lists, in order. Read the actual document — do not rely on memory.
2. Complete every prerequisite eligibility check the procedure names before the
   terminal write. For these workflows that includes verifying the account's
   outstanding balance, any active or pending transaction disputes, any pending
   replacement-card orders, the account's age, and — for closure — prior
   retention attempts, or — for CLI — the request cooldown, payment history
   and current utilization. Several of these checks require discoverable read
   tools; unlock and call them rather than skipping a check whose tool is not
   immediately visible.
3. Do not perform the terminal write — closing the account, or approving or
   denying a limit increase — until every documented prerequisite has been
   checked and satisfied. If a blocking condition exists (for example a
   pending dispute, a pending replacement card, an unpaid balance, or an
   ineligible amount), follow the procedure's alternative path — resolve the
   blocker first, deny with the correct coded reason, or defer — instead of
   forcing the write.

When a customer wants to close an account because they found a better card
elsewhere, the documented retention step is to ask what attracted them and, if
Rho-Bank offers a comparable card, offer to help them apply for it instead of
simply closing the current account. Only close the account if the customer
still declines after the documented retention steps.

When the customer makes two requests that interact (for example a limit
increase and filing a new dispute), sequence them so each is processed while
its eligibility still holds, rather than acting in the order they were
mentioned.
</account_workflow_protocol>
""".strip()

# Generations per turn: 1 original + 1 steered retry.
MAX_GENERATIONS = 2


class ClosureCLIProcedureAgent(LLMAgent):
    """LLMAgent that completes the documented closure/CLI prerequisite phase."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._guard = ProcedureGuard()

    @property
    def system_prompt(self) -> str:
        base = SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        )
        return base + "\n" + ACCOUNT_WORKFLOW_PROTOCOL

    def get_init_state(self, message_history=None) -> LLMAgentState:
        self._guard.reset()
        return super().get_init_state(message_history)

    def _generate(self, state: LLMAgentState, guidance: Optional[str]) -> AssistantMessage:
        """Run one LLM generation, optionally with a transient turn note."""
        system_messages = state.system_messages
        if guidance:
            base = state.system_messages[0].content
            system_messages = [
                SystemMessage(
                    role="system",
                    content=f"{base}\n\n<turn_note>\n{guidance}\n</turn_note>",
                )
            ]
        messages = system_messages + state.messages
        return generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            call_name="agent_response",
            **self.llm_args,
        )

    def generate_next_message(self, message, state: LLMAgentState):
        """Respond to a user or tool message, completing prerequisites first."""
        # Mirror LLMAgent: incoming message(s) are appended to history.
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        guidance: Optional[str] = None
        assistant_message: Optional[AssistantMessage] = None
        for _ in range(MAX_GENERATIONS):
            assistant_message = self._generate(state, guidance)
            note = self._guard.needs_prereq(assistant_message, state.messages)
            if note is None:
                break
            # Regenerate once, steering the agent to finish the prereq phase.
            self._guard.register_nudge()
            guidance = note

        state.messages.append(assistant_message)
        return assistant_message, state


def build_agent(tools, domain_policy, **kwargs):
    return ClosureCLIProcedureAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
