"""tau2 candidate mh_tau2_iter14_baseline_procedure_sequencing.

Hypothesis
----------
Multi-step banking procedures fail the database-hash check when the agent
executes them INCOMPLETELY or OUT OF ORDER. The sharpest, most deterministically
fixable case is ordering: when a customer makes two requests in one
conversation, the agent processes them in the order they were mentioned, and an
action it takes for the first request creates an account condition that blocks
the second. The gold trajectory orders the same two requests so neither blocks
the other. Steering the agent to (a) complete every documented step of a
procedure and (b) sequence interacting requests so an earlier action does not
block a later one will move the agent's write set onto gold and raise train-30
reward.

Mechanism observed (banking sims, frontier still fails all, reward_basis = DB)
-----------------------------------------------------------------------------
* task_053 — the customer wants both a transaction dispute and a credit-limit
  increase (CLI). The agent files the transaction dispute first; that places a
  PENDING DISPUTE on the account, and a pending dispute is a documented
  blocking condition for a CLI. The agent's own eligibility analysis shows the
  customer qualifies, yet it DENIES the CLI with reason ``pending_disputes`` —
  a blocker it created itself. Gold APPROVES the CLI: it completes the
  credit-limit-increase decision before filing the dispute. Exactly one gold
  action (the CLI approve) is missing from the agent's run.
* task_044 — closure + new-card application. The agent skips the documented
  dispute-history prerequisite read and hands the application off ("I don't
  have the ability to process a new credit card application") by transferring
  to a human, so two gold steps never happen.
* task_045 — closure with an outstanding balance the customer explicitly asks
  to pay off ("Yes, please pay that off from my checking account"). The agent
  acknowledges but never executes the payoff, never runs the dispute-history
  read, and the conversation ends with the procedure half-finished.
* task_048 — four-card closure. The agent skips the balance-payoff step (it
  even calls a guessed, nonexistent tool name) and skips a closure-reason log,
  leaving the write set short of gold.

Change
------
1. The system prompt gains a general, domain-agnostic <procedure_sequencing>
   section: treat closures, credit-limit increases, disputes and applications
   as documented multi-step procedures; carry out EVERY step including
   prerequisite reads; execute an action the customer explicitly authorizes
   rather than merely acknowledging it; when a customer makes several requests,
   finish one before starting the next and order them so completing one does
   not create a condition that blocks another; do not transfer to a human or
   redirect to an external channel for a request the available tools can
   complete.
2. A deterministic guard (sequencing_guard.SequencingGuard) detects a turn that
   files a transaction dispute while a CLI workflow is in progress but not yet
   decided, and regenerates that turn once with a transient note steering the
   agent to complete the CLI decision first. Detection is structural (substring
   tests on the discoverable inner tool name, never a hardcoded id), the guard
   only fires when BOTH a CLI workflow and a dispute filing are live in the
   same conversation, and the nudge is capped — so it can never perturb a task
   that involves only one of the two, and can never permanently suppress a turn.

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

from agent_tau2.mh_tau2_iter14_baseline_procedure_sequencing.sequencing_guard import (
    SequencingGuard,
)

# General, domain-agnostic guidance appended to the system prompt.
PROCEDURE_SEQUENCING = """
<procedure_sequencing>
Many banking requests — account closures, credit-limit increases, transaction
disputes, card applications — are documented multi-step procedures, not single
actions. Retrieve the full procedure from the knowledge base and carry out
EVERY step it lists, in the order it specifies. Do not stop after the most
obvious action.

Completeness:
- Perform every prerequisite check the procedure names (for example verifying
  an outstanding balance, pending transaction disputes, pending replacement-card
  orders, account age, payment history). If the tool for a check is not yet
  visible, search the knowledge base to unlock it — never skip a documented
  check because its tool is not immediately available, and never call a guessed
  tool name.
- When the customer explicitly authorizes an action, carry it out with your
  tools in the same conversation; do not merely acknowledge it and move on.

Ordering — when a customer makes several requests in one conversation, finish
one fully before starting the next, and order them so that completing one does
not create an account condition that blocks another:
- Filing a transaction dispute places a PENDING DISPUTE on the account, and a
  pending dispute is a blocking condition for a credit-limit increase. So when
  a customer wants both a credit-limit increase and a new transaction dispute,
  complete the credit-limit-increase decision FIRST, then file the dispute.
- More generally, before taking an action that opens a pending case or changes
  account state, check whether another pending request depends on the current
  state, and handle that request first.

Do not transfer to a human agent, or send the customer to an external website,
app, or branch, for a request that your tools — or the customer's own tools —
can complete. Transfer only when the policy explicitly requires it.
</procedure_sequencing>
""".strip()

# Generations per turn: 1 original + 1 steered retry.
MAX_GENERATIONS = 2


class ProcedureSequencingAgent(LLMAgent):
    """LLMAgent that completes procedures fully and orders interacting requests."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._guard = SequencingGuard()

    @property
    def system_prompt(self) -> str:
        base = SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        )
        return base + "\n" + PROCEDURE_SEQUENCING

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
        """Respond to a user or tool message, reordering interacting requests."""
        # Mirror LLMAgent: incoming message(s) are appended to history.
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        guidance: Optional[str] = None
        assistant_message: Optional[AssistantMessage] = None
        for _ in range(MAX_GENERATIONS):
            assistant_message = self._generate(state, guidance)
            note = self._guard.needs_reorder(assistant_message, state.messages)
            if note is None:
                break
            # Regenerate once, steering the agent to reorder the two requests.
            self._guard.register_nudge()
            guidance = note

        state.messages.append(assistant_message)
        return assistant_message, state


def build_agent(tools, domain_policy, **kwargs):
    return ProcedureSequencingAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
