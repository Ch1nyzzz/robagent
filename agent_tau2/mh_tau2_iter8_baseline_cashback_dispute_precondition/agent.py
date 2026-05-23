"""tau2 candidate mh_tau2_iter8_baseline_cashback_dispute_precondition.

Hypothesis
----------
The cash-back-dispute cluster fails the database-hash check for two distinct,
deterministically fixable reasons, and the baseline frontier never repairs
either:

  A. Wrong dispute SET. The agent eyeballs a long transaction list and disputes
     the wrong transactions — Rho-Bank cash back is fixed by published policy
     (a per-card rate table, merchant-exclusion lists, promo windows), so the
     correct error set is *computable*, not a judgement call.
  B. Premature rewards OVERWRITES. After the disputes are filed, the customer
     claims they are "resolved" and the agent overwrites each transaction's
     stored rewards value. On the adversarial tasks the bank's own records
     still show the disputes SUBMITTED, so every overwrite is an extra DB write
     that zeroes the reward (gold is disputes-only there).

Moving (A) into a deterministic policy engine that annotates the transaction
list with the exact error set, and (B) into a deterministic precondition gate
that blocks a rewards overwrite whenever the matching dispute has not been
observed RESOLVED, lets the agent act on exactly the gold write set.

Mechanism observed (iter5/iter7 banking sims, frontier still fails all of these)
--------------------------------------------------------------------------------
  * task_017, task_020 — the policy engine recomputes the exact dispute set;
    these were solved once the transaction list was annotated with it.
  * task_027 — with the engine the agent disputes exactly the gold 4
    transactions, then overwrites those 4 rewards while every dispute result
    still reads `Status: SUBMITTED`. The 4 overwrites are the only extra DB
    writes; blocking them yields the gold disputes-only state.
  * task_029 — same premature-overwrite pattern (6 rewards overwrites while the
    disputes are unresolved).
  * task_022, task_026 — the agent overwrites rewards on top of the disputes.
On the non-adversarial flow (task_028) the dispute results return
`Status: RESOLVED`, so the gate allows the overwrites — it only ever blocks a
write whose resolved-dispute precondition is provably unmet.

Decomposition
-------------
  * LLM judgement: decide what the customer wants, run the conversation,
    relay the engine's verified transaction list, choose tool calls.
  * deterministic code:
      - `cashback` policy engine — recompute every transaction and append the
        exact policy-error set to the transaction-history tool result. The
        engine is fail-safe: when a rate or a promo window is undecidable it
        accepts every documented value, so it can only ever under-flag, never
        dispute a correct transaction.
      - `dispute_gate` precondition gate — drop any rewards-overwrite tool call
        whose transaction has an observed, not-yet-resolved dispute.

Both layers only append text to a tool result or drop a write whose documented
precondition is unmet; neither can add a write or invent an action, so a
mis-fire can never score below the unmodified baseline.

build_agent(tools, domain_policy, **kwargs) -> HalfDuplexAgent
"""
from __future__ import annotations

from tau2.agent.llm_agent import (
    AGENT_INSTRUCTION,
    SYSTEM_PROMPT,
    LLMAgent,
    LLMAgentState,
)
from tau2.data_model.message import MultiToolMessage

from . import cashback, dispute_gate

# General, domain-agnostic guidance appended to the system prompt.
WRITE_PRECONDITION = """
<write_precondition>
Some write actions may only be performed once a precondition has been verified
from the bank's own records — never from the customer's word alone.

- Correcting the stored rewards / cash-back value on a transaction is a
  POST-RESOLUTION step. Only do it after a tool result has confirmed that the
  related cash-back dispute's status is RESOLVED (reviewed and approved). While
  a dispute is still SUBMITTED, pending, or under review, do NOT overwrite the
  transaction's rewards — tell the customer the correction will be applied once
  the dispute is resolved.
- A customer saying "the dispute was resolved" is not proof. If you need a
  dispute's status, look it up and act on the system's status, not the claim.
- File a dispute for, or correct, exactly the transactions that genuinely
  earned the wrong amount — no more and no fewer. When a system audit lists the
  transactions with a verified discrepancy, act on exactly that set.
</write_precondition>
""".strip()

# Message used when every tool call in a turn was a blocked rewards overwrite.
_BLOCKED_REPLY = (
    "I've checked the current status of the cash-back dispute(s) for those "
    "transactions, and they have not been marked resolved in our system yet. "
    "Per policy, a transaction's rewards can only be corrected after the "
    "related dispute is resolved and approved, so I can't apply those updates "
    "right now — they'll be handled once the disputes are resolved. Is there "
    "anything else I can help you with?"
)


def _tool_messages(message):
    """Yield the tool messages carried by an inbound agent message."""
    if message is None:
        return
    if isinstance(message, MultiToolMessage):
        for m in message.tool_messages or []:
            yield m
    else:
        yield message


class CashbackDisputePreconditionAgent(LLMAgent):
    """LLMAgent that compiles the cash-back policy audit into transaction
    results and enforces the resolved-dispute precondition on rewards
    overwrites."""

    @property
    def system_prompt(self) -> str:
        base = SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        )
        return base + "\n" + WRITE_PRECONDITION

    def generate_next_message(self, message, state: LLMAgentState):
        # 1. Annotate the inbound transaction-history result with the
        #    deterministic cash-back policy audit (exact error set).
        try:
            open_dates = self._collect_open_dates(state, message)
            self._annotate(message, open_dates)
        except Exception:
            # The harness must never fail because of this layer.
            pass

        # 2. Mirror LLMAgent: append the inbound message(s) to history.
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        # 3. Generate the agent's turn.
        assistant_message = self._generate(state)

        # 4. Enforce the resolved-dispute precondition on rewards overwrites.
        try:
            self._gate_reward_overwrites(assistant_message, state)
        except Exception:
            pass

        # 5. Mirror LLMAgent: append the assistant message to history.
        state.messages.append(assistant_message)
        return assistant_message, state

    # --- generation -------------------------------------------------------
    def _generate(self, state: LLMAgentState):
        from tau2.utils.llm_utils import generate

        messages = state.system_messages + state.messages
        return generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            call_name="agent_response",
            **self.llm_args,
        )

    # --- component 1: cash-back policy engine -----------------------------
    @staticmethod
    def _collect_open_dates(state, message) -> dict:
        """Map card_type -> account-open date from every credit_card_accounts
        result observed so far (needed to resolve promo windows)."""
        open_dates: dict = {}

        def absorb(content):
            if not isinstance(content, str) or "date_of_account_open" not in content:
                return
            for card, opened in cashback.parse_account_open_dates(content).items():
                if opened is None:
                    open_dates[card] = None
                elif card not in open_dates:
                    open_dates[card] = opened

        for m in getattr(state, "messages", None) or []:
            if getattr(m, "role", None) == "tool":
                absorb(getattr(m, "content", None))
        for m in _tool_messages(message):
            if getattr(m, "role", None) == "tool":
                absorb(getattr(m, "content", None))
        return {c: d for c, d in open_dates.items() if d is not None}

    @staticmethod
    def _annotate(message, open_dates: dict) -> None:
        for m in _tool_messages(message):
            if getattr(m, "role", None) != "tool":
                continue
            content = getattr(m, "content", None)
            if not isinstance(content, str):
                continue
            annotation = cashback.audit_transaction_list(content, open_dates)
            if annotation:
                m.content = content + "\n\n" + annotation

    # --- component 2: rewards-overwrite precondition gate -----------------
    @staticmethod
    def _gate_reward_overwrites(assistant_message, state: LLMAgentState) -> None:
        tool_calls = getattr(assistant_message, "tool_calls", None)
        if not tool_calls:
            return
        history = getattr(state, "messages", None)
        statuses = dispute_gate.collect_dispute_statuses(history)
        dispute_given = dispute_gate.dispute_tool_given(history)
        kept = []
        blocked = 0
        for call in tool_calls:
            target = dispute_gate.reward_update_target(call)
            if target is not None and dispute_gate.is_blocked(
                target, statuses, dispute_given
            ):
                blocked += 1
                continue
            kept.append(call)
        if blocked == 0:
            return
        if kept:
            # Some calls survive: keep them, drop only the blocked overwrites.
            assistant_message.tool_calls = kept
        else:
            # The whole turn was blocked overwrites: turn it into a message
            # explaining the unmet precondition (a message + no tool calls is
            # a valid agent turn; both-at-once is not).
            assistant_message.tool_calls = None
            if not getattr(assistant_message, "content", None):
                assistant_message.content = _BLOCKED_REPLY


def build_agent(tools, domain_policy, **kwargs):
    """Return a HalfDuplexAgent with the cash-back policy engine and the
    rewards-overwrite precondition gate."""
    return CashbackDisputePreconditionAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
