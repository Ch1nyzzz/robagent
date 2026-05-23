"""tau2 candidate mh_tau2_iter13_baseline_call_set_sanitizer.

Hypothesis
----------
Closure / credit-limit-increase (CLI) DB-hash tasks fail because the agent's
SET of discoverable-tool calls diverges from gold by adding calls that no gold
trajectory ever makes. banking_knowledge DB tasks are graded on the resulting
database state, which records one ``agent_discoverable_tools`` row per unique
discoverable tool the agent calls; the reward is zero unless that set exactly
equals gold. Two never-gold extras recur:

(A) The agent hands the customer a website / app navigation redirect tool
    (``navigate_to_section``, ``open_webpage``) instead of performing the
    action itself — a guaranteed extra DB row.
(B) On a CLI task the agent hedges and calls BOTH the approval and the denial
    branch of the decision. Gold has exactly one outcome, so the second,
    contradictory branch is always an extra row.

Deterministically stripping these never-gold calls — and, with iter10's proven
prerequisite-read gate, reliably driving the two universal closure/CLI reads —
brings the discoverable-call set into line with gold and raises train-30
reward.

Mechanism observed
------------------
Surveying the 30 train-set gold trajectories: zero use a navigation redirect
tool; every CLI gold uses exactly one of approve/deny, never both. Surveying
890 prior simulations: redirect tools were emitted on eight distinct tasks
(task_001/002/003/014/024/025/044/048) and every such simulation scored 0;
both decision branches were emitted together 36 times, always on a CLI task
(task_050/051/053/054). Among the seven tasks the frontier still fails, the
never-gold-extra mechanism is present in:
  * task_024 — agent gives the customer ``open_webpage`` (extra row).
  * task_044 — agent gives the customer ``navigate_to_section`` (extra row).
  * task_048 — agent gives ``navigate_to_section`` (extra row).
  * task_053 — agent calls ``approve_credit_limit_increase`` and then the
    contradictory ``deny_credit_limit_increase`` (extra row); its only other
    divergence is a skipped ``get_pending_replacement_orders`` read.

Change
------
1. call_sanitizer.CallSanitizer deterministically strips, from every assistant
   turn before it runs, (a) any redirect-tool call and (b) any decision-tool
   call whose polarity contradicts a decision already committed for that
   subject (in a prior turn or earlier in the same turn). It can never remove
   a gold action: a redirect tool is never gold, and the opposite polarity of
   a committed decision is never gold either. When stripping empties a turn,
   the turn is regenerated once, steered to act through real tools.
2. procedure_guard.ProcedureGuard (iter10's proven gate): when a terminal
   closure / CLI-decision write precedes the universal dispute-history and
   pending-replacement-orders reads, the turn is regenerated once with a note
   naming the missing checks. Never blocks a write outright.
3. The system prompt gains a general <account_workflow_protocol> (complete the
   documented prerequisite checks before a closure/CLI write) and a general
   <discoverable_tool_discipline> (act through your own tools, never hand the
   customer a website-navigation tool, never reverse a decision you have
   already submitted).

No customer names, ids, card ids, rates, or per-task branching are used.
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

from agent_tau2.mh_tau2_iter13_baseline_call_set_sanitizer.call_sanitizer import (
    CallSanitizer,
)
from agent_tau2.mh_tau2_iter13_baseline_call_set_sanitizer.procedure_guard import (
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
   ineligible amount), follow the procedure's alternative path instead of
   forcing the write.

When a customer wants to close an account because they found a better card
elsewhere, the documented retention step is to ask what attracted them and, if
Rho-Bank offers a comparable card, offer to help them apply for it instead of
simply closing the current account.
</account_workflow_protocol>
""".strip()

DISCOVERABLE_TOOL_DISCIPLINE = """
<discoverable_tool_discipline>
You complete the customer's request yourself, through your tools.

- Act, do not redirect. Never hand the customer a website, app, or section
  navigation tool, and never tell them to finish the task on a website or in
  an app, when a tool of yours can perform the action. If a tool needs a
  detail you do not have, ask the customer for it and then make the call.
- One decision per request. A request that is decided by an approval-or-denial
  step has exactly one outcome. Once you have submitted an approval or a
  denial for a request, do not also submit the opposite decision for the same
  request — decide once, deliberately.
- Call only the tools the task needs. Every discoverable tool you call is
  recorded; do not call a tool that does not advance the customer's actual
  request.
</discoverable_tool_discipline>
""".strip()

# Maximum LLM generations per turn (original + up to two steered retries).
MAX_GENERATIONS = 3

# Steering note used when deterministic sanitization empties a turn.
SANITIZER_GUIDANCE = (
    "Your previous draft was discarded because every tool call in it was a "
    "call that cannot help complete this task: handing the customer a "
    "website/app navigation tool instead of doing the work yourself, or "
    "submitting a decision that reverses one you have already made. Produce a "
    "new turn that either calls a tool which genuinely advances the customer's "
    "request, or sends the customer a helpful message. Perform the requested "
    "action directly with your own tools; if you are missing a required "
    "detail, ask the customer for it."
)

# Last-resort content when a turn is still empty after every retry.
FALLBACK_MESSAGE = (
    "Let me help you with that directly. Could you confirm the details you'd "
    "like me to use so I can proceed?"
)


class CallSetSanitizerAgent(LLMAgent):
    """LLMAgent that strips never-gold discoverable calls and gates workflows."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sanitizer = CallSanitizer()
        self._guard = ProcedureGuard()

    @property
    def system_prompt(self) -> str:
        base = SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        )
        return base + "\n" + ACCOUNT_WORKFLOW_PROTOCOL + "\n" + DISCOVERABLE_TOOL_DISCIPLINE

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

    @staticmethod
    def _is_empty(message: AssistantMessage) -> bool:
        """True when a message carries neither text content nor a tool call."""
        content = getattr(message, "content", None)
        has_content = content is not None and bool(str(content).strip())
        return not has_content and not getattr(message, "tool_calls", None)

    def generate_next_message(self, message, state: LLMAgentState):
        """Respond to a user or tool message, sanitizing the turn first."""
        # Mirror LLMAgent: incoming message(s) are appended to history.
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        guidance: Optional[str] = None
        assistant_message: Optional[AssistantMessage] = None
        for attempt in range(MAX_GENERATIONS):
            last_attempt = attempt == MAX_GENERATIONS - 1
            assistant_message = self._generate(state, guidance)

            # 1. Deterministically strip never-gold discoverable calls.
            self._sanitizer.sanitize(assistant_message, state.messages)

            # 2. If stripping emptied the turn, steer toward acting directly.
            if self._is_empty(assistant_message):
                if last_attempt:
                    assistant_message.tool_calls = None
                    assistant_message.content = FALLBACK_MESSAGE
                    break
                guidance = SANITIZER_GUIDANCE
                continue

            # 3. Closure/CLI prerequisite-read gate (iter10's proven logic).
            note = self._guard.needs_prereq(assistant_message, state.messages)
            if note is not None and not last_attempt:
                self._guard.register_nudge()
                guidance = note
                continue
            break

        state.messages.append(assistant_message)
        return assistant_message, state


def build_agent(tools, domain_policy, **kwargs):
    return CallSetSanitizerAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
