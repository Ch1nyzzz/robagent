"""tau2 candidate mh_tau2_iter13_robust_cashback_status_ledger.

Hypothesis
----------
Train task_029 (the adversarial cash-back rewards correction task) fails on
the v3 robust frontier because, after Stage 1 has been done and the
framework has emitted ``Status: SUBMITTED`` for every dispute response, the
user-simulator LIES: "I just got a notification that they've all been
approved and resolved! Can you now update my transaction records?" The LLM
trusts the user's natural-language claim and calls
``update_transaction_rewards_3847`` six times. Gold has zero Stage-2 calls
(auto_resolve_disputes=False); the agent's six extra calls write rows to
the ``credit_card_transaction_history`` (rewards_earned updates) and
``agent_discoverable_tools`` (unlock + call audit rows) tables that gold
does not have, so the DB hash diverges.

The iter5 ``cashback_stage_state_advisor`` was built to surface Stage-1
state at the Stage-2 unlock-tool response, but two structural facts prevent
it from helping here:

  1. iter7 (``ClosureCliPrereqAdvisorAgent``) forked off ``LLMAgent``
     directly, not from iter5. The v3 frontier chain is iter7 -> iter10 ->
     iter11 -> iter12, none of which inherits iter5's per-tool-response
     advisor. Inspecting the task_029 iter12 trace at the Stage-2 unlock
     turn confirms the [STAGE STATE] suffix is absent.
  2. Even when present, iter5's per-tool-response augmentation only
     reaches the LLM AT the Stage-2 unlock turn — by then the LLM has
     already committed (in the same turn) to a batched Stage-2 plan. The
     user's deceptive claim from the prior user turn is the loudest signal
     in the context at decision time.

Decomposition
-------------
Move the Stage-1 truth out of a one-shot per-tool-response advisory and
into PERSISTENT system context: rebuild a STAGE-1 LEDGER SystemMessage at
every assistant turn, listing each (transaction_id, framework-emitted
status) pair from every ``submit_cash_back_dispute_0589`` response in the
conversation, and append it to the agent's ``state.system_messages``. The
ledger refreshes on every turn so the framework truth is always at the top
of the LLM's context regardless of how the conversation unfolds. The LLM
remains the sole decision-maker about every tool call; the ledger is text
only.

Mechanism (train sims the v3 frontier still fails)
--------------------------------------------------
  * task_029 (iter12 sim, db_check=False): action_checks 8/8 pass (gold
    Stage-1 only); but the agent makes 6 extra ``update_transaction_rewards_3847``
    calls after the user's deceptive claim. The Stage-1 ledger after Phase
    1 of the conversation carries five entries, all ``Status: SUBMITTED``
    (auto_resolve_disputes=False). With those five entries persisted as a
    SystemMessage at every turn after, the LLM has the framework truth in
    context when the user lies, and may decline Stage 2.
  * task_026 (iter12 sim, db_check=False): auto_resolve_disputes=True; the
    Stage-1 ledger carries four ``Status: RESOLVED`` entries. The advisor
    correctly signals that Stage 2 IS warranted (gold expects four Stage-2
    calls). On task_026 the iter12 trace's failure is a ±1 rounding noise
    issue inside the iter10 engine (1499 vs 1500); the ledger neither
    helps nor hurts.
  * task_022 / task_020 (currently passing via iter10 engine): the ledger
    refresh runs but only changes the wording of context already present;
    no regression expected because Stage 2 is correctly warranted on these
    tasks (auto_resolve=True with RESOLVED status).

Why this captures stable structure (not training-set induction)
---------------------------------------------------------------
Two layers, both independent of the failed simulations:

  1. Framework tool-schema fact. The discoverable user-side tool name
     ``submit_cash_back_dispute_0589`` is declared at
     tau2-bench-src/src/tau2/domains/banking_knowledge/tools.py line 4106
     with @is_discoverable_tool(WRITE). The give-call audit row is at
     tools.py:4121, the cash_back_disputes row at tools.py:4151. These are
     tau2-system facts.
  2. Framework return-string fact. The literals ``Executed:
     submit_cash_back_dispute_0589`` (tools.py:4155), ``Status: SUBMITTED``
     (tools.py:4149), and ``Status: RESOLVED`` (tools.py:4140) are emitted
     directly by the framework's response body into the tool message. The
     ledger parser keys on these literals; no KB policy text is read by
     the parser, and the agent's prose is not inspected at any point.

The ledger never asserts a verdict on any specific tool call. It lists the
framework Status verdicts and the consequence wording (Stage 2 may only
execute when Status=RESOLVED) is the documented content of
doc_credit_cards_credit_cards_(general)_004, already preloaded into the
system prompt by the iter1/iter2 internal_procedure_channel. The LLM is
free to proceed to Stage 2 regardless of what the ledger reports.

build_agent(tools, domain_policy, **kwargs) -> HalfDuplexAgent
"""

from __future__ import annotations

from typing import List

from tau2.agent.base_agent import ValidAgentInputMessage
from tau2.agent.llm_agent import LLMAgentState
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    SystemMessage,
    ToolMessage,
)

from agent_tau2.mh_tau2_iter12_robust_dispute_card_last4_audit.agent import (
    DisputeCardLast4AuditAgent,
)

from . import status_ledger


class CashbackStatusLedgerAgent(DisputeCardLast4AuditAgent):
    """Refresh a STAGE-1 LEDGER SystemMessage at every assistant turn.

    Inherits iter12's full chain (card_last4 resolver + dispute audit +
    iter10 cashback policy engine + iter7 closure prereq advisor + iter4
    audit visibility + iter1/2/3 KB channels) unchanged. Adds one new
    deterministic transform: scan the visible history for framework-
    emitted ``submit_cash_back_dispute_0589`` responses, parse each into a
    (transaction_id, Status) pair, and mount the assembled ledger as a
    SystemMessage at the END of ``state.system_messages`` so it is
    refreshed on every turn.

    The agent's ``state.system_messages`` already starts with one entry
    (the base system prompt from ``LLMAgent.system_prompt``). The ledger
    is appended as a second SystemMessage tagged with the LEDGER_MARKER;
    on the next turn it is replaced wholesale (drop any prior LEDGER_MARKER
    entry and append a fresh one). Idempotent across turns; absent when
    no Stage-1 dispute responses have been seen.
    """

    def _ledger_iterable(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ):
        """Return (history, current) for the ledger parser.

        At the moment ``_generate_next_message`` is called, the incoming
        ``message`` has not yet been appended to ``state.messages``. We
        feed both into the parser so a Stage-1 response arriving in this
        very turn is reflected in the ledger we render this turn.
        """
        history = list(getattr(state, "messages", []) or [])
        # MultiToolMessage carries its own tool_messages list; append each.
        if isinstance(message, MultiToolMessage):
            return history + list(message.tool_messages or [])
        if isinstance(message, ToolMessage):
            return history + [message]
        return history

    def _refresh_ledger_system_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> None:
        """Rebuild and remount the LEDGER SystemMessage on ``state.system_messages``.

        - Drop any prior SystemMessage whose content carries the LEDGER_MARKER
          (one is mounted; we never accumulate).
        - Compute fresh entries from the visible history including ``message``.
        - If entries is non-empty, mount a fresh SystemMessage at the END of
          ``state.system_messages`` so it sits just before the conversation
          messages in the LLM context.
        """
        # Drop stale ledger system messages (idempotency across turns).
        kept: List[SystemMessage] = []
        for sm in getattr(state, "system_messages", []) or []:
            c = getattr(sm, "content", None)
            if isinstance(c, str) and status_ledger.LEDGER_MARKER in c:
                continue
            kept.append(sm)
        state.system_messages = kept

        # Compute the fresh ledger from visible history + incoming message.
        visible = self._ledger_iterable(message, state)
        entries = status_ledger.parse_ledger(visible)
        rendered = status_ledger.render_ledger(entries)
        if not rendered:
            return
        state.system_messages.append(
            SystemMessage(role="system", content=rendered)
        )

    def _generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> AssistantMessage:
        try:
            self._refresh_ledger_system_message(message, state)
        except Exception:
            # Never let the ledger refresh mask a real conversation; fall
            # through to the inherited chain on any error.
            pass
        return super()._generate_next_message(message, state)


def build_agent(tools, domain_policy, **kwargs):
    """Return a HalfDuplexAgent that refreshes a Stage-1 cash-back dispute
    STATUS ledger as a SystemMessage at every assistant turn."""
    return CashbackStatusLedgerAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
