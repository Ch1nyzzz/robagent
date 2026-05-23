"""tau2 candidate mh_tau2_iter7_baseline_cashback_dispute_workflow.

Hypothesis
----------
On cash-back / rewards discrepancy tasks the agent loses the database-hash
reward because it uses the WRONG remediation tool. The gold workflow is the
cash-back DISPUTE process: the agent gives the customer a cash-back-dispute
tool and a dispute is filed for each genuinely wrong transaction. The agent
instead calls the tool that DIRECTLY overwrites a transaction's stored rewards
value (`update_transaction_rewards`). It does this either in place of filing
disputes at all, or layered on top of the disputes — and both break the DB
hash, because the gold action set for these tasks is disputes-only (or
disputes followed by a small, procedure-authorised correction). Steering the
agent back to the dispute workflow removes a definite db-breaking class of
extra/incorrect writes and raises train-30 reward.

Mechanism observed (iter6 banking sims, frontier still fails all of these)
--------------------------------------------------------------------------
Every cash-back task fails with db_match=false; in each, the agent calls the
direct rewards-edit tool when gold uses (or only uses) the dispute tool:
  * task_017 — agent called update_transaction_rewards x3 and NEVER gave the
    customer the cash-back dispute tool; gold = give dispute tool + 2 disputes.
  * task_027 — agent called update_transaction_rewards x13 and NEVER filed a
    dispute; gold = give dispute tool + 4 disputes.
  * task_022 — agent gave the dispute tool but also called
    update_transaction_rewards x9; gold = disputes only, no edits.
  * task_029 — agent filed disputes AND called update_transaction_rewards x5;
    gold = disputes only, no edits.
  * task_026 — agent filed disputes then called update_transaction_rewards x6;
    gold pairs disputes with a small set of post-resolution corrections.
The direct-edit calls are wrong-tool / extra writes in N>=4 disputes-only
tasks, so they alone zero the DB hash.

Change
------
1. The system prompt gains a general, domain-agnostic <remediation_protocol>:
   when a customer reports an incorrect value on their account record, follow
   the knowledge-base remediation procedure for that issue; for a cash-back
   discrepancy the remediation is to file a cash-back dispute, and a tool that
   directly overwrites the stored rewards value is only a post-resolution
   correction the procedure must explicitly authorise. It also reminds the
   agent to retrieve each card's earning-rate policy before auditing and to
   treat ~1-point differences as rounding, so it flags only materially wrong
   transactions. No identifiers, card names, or task hints are hardcoded.
2. A deterministic guard (cashback_guard.CashBackRemediationGuard) detects a
   turn that calls a tool which directly overwrites a transaction's stored
   rewards value (matched structurally: a tool naming a transaction + rewards
   with an edit verb, never the dispute tool or a read tool). The turn is
   regenerated once with a transient note steering the agent to the dispute
   workflow. The nudge fires at most once per conversation and never blocks
   the call, so a legitimate procedure-authorised correction still goes
   through.

The LLM still decides which transactions are wrong and what to file; the guard
only guarantees the agent reconsiders the dispute workflow before it reaches
for the direct-edit shortcut.
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

from agent_tau2.mh_tau2_iter7_baseline_cashback_dispute_workflow.cashback_guard import (
    CashBackRemediationGuard,
)

# General, domain-agnostic guidance appended to the system prompt.
REMEDIATION_PROTOCOL = """
<remediation_protocol>
When a customer reports that a value on their account record is wrong (for
example, an incorrect cash-back / rewards amount on a transaction), follow the
bank's defined remediation procedure for that issue. Do not invent your own
shortcut.

Before taking any remediation write action:
1. Retrieve the knowledge-base procedure that governs the customer's specific
   issue. For a rewards / cash-back discrepancy, search for the cash-back
   dispute procedure. Use the exact tool and workflow that procedure
   prescribes.
2. For a customer-reported incorrect cash-back amount, the remediation is to
   file a cash-back dispute through the customer-facing dispute tool. Filing
   the dispute IS the resolution step. A tool that directly overwrites the
   stored rewards value on a transaction is a post-resolution correction:
   use it only when the dispute procedure explicitly directs you to apply a
   correction after a dispute has been resolved. Never use it as a first-line
   fix in place of filing a dispute, and never layer it on top of a dispute
   you just filed.
3. Act only on transactions you are certain are materially wrong. Before
   auditing rewards, retrieve each card's earning-rate policy from the
   knowledge base so your expected-rewards math uses the correct rates. A
   difference of about one point between expected and recorded rewards is
   ordinary rounding, not an error — do not dispute or correct it. When in
   doubt, leave a transaction alone rather than flagging a correct one.
</remediation_protocol>
""".strip()

# Maximum number of generations per turn (1 original + 1 steered retry).
MAX_GENERATIONS = 2


class CashBackDisputeWorkflowAgent(LLMAgent):
    """LLMAgent that steers cash-back remediation toward the dispute workflow."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._guard = CashBackRemediationGuard()

    @property
    def system_prompt(self) -> str:
        base = SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        )
        return base + "\n" + REMEDIATION_PROTOCOL

    def get_init_state(self, message_history=None) -> LLMAgentState:
        self._guard.reset()
        return super().get_init_state(message_history)

    def _generate(
        self, state: LLMAgentState, guidance: Optional[str]
    ) -> AssistantMessage:
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
        """Respond to a user or tool message, steering away from direct edits."""
        # Mirror LLMAgent: incoming message(s) are appended to history.
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        guidance: Optional[str] = None
        assistant_message: Optional[AssistantMessage] = None
        for _ in range(MAX_GENERATIONS):
            assistant_message = self._generate(state, guidance)
            if not self._guard.needs_redirect(assistant_message):
                break
            # Regenerate once, steering the agent to the dispute workflow.
            self._guard.register_nudge()
            guidance = self._guard.guidance()

        state.messages.append(assistant_message)
        return assistant_message, state


def build_agent(tools, domain_policy, **kwargs):
    return CashBackDisputeWorkflowAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
