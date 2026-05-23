"""Provisional-credit eligibility checklist advisor for
file_credit_card_transaction_dispute_4829 entry-point calls in the
banking_knowledge domain.

What this module is for
-----------------------
KB doc ``doc_credit_cards_credit_cards_(general)_015`` ("Provisional Credit
Eligibility Guidelines (Internal)") specifies a strict, multi-part rule for
the ``eligible_for_provisional_credit`` field of
``file_credit_card_transaction_dispute_4829``. The doc is already loaded into
the system prompt by the iter1/iter2 ``internal_procedure_channel`` plugin,
yet across the v3 evolution chain and the parallel baseline chain the LLM
still mis-applies criterion 2 (eligible reason category) and criterion 5
(non-fraud reasons require ``contacted_merchant=true``) on the dispute that
fires on tasks like task_038 dispute #3 (duplicate_charge with
``contacted_merchant=false`` — the agent sets eligible=true, gold expects
false).

This module surfaces the doc's eligibility checklist as a decision-point
advisory at the moment the agent unlocks or calls
``file_credit_card_transaction_dispute_4829``. The advisor:

  * cites the KB doc id (``doc_credit_cards_credit_cards_(general)_015``);
  * lists the five eligibility criteria literally as written in the doc,
    including the "NOT Eligible Scenarios" worked-example;
  * notes that criteria 2 (reason category) and 5 (non-fraud merchant
    contact) are decidable from the dispute call's OWN arguments alone, so
    the agent does not need any extra read tool to apply them.

What this module does NOT do
----------------------------
  * It does not modify, rewrite, suppress, or fabricate any tool call.
  * It does not branch on user content, customer name, account id,
    transaction id, or any other task-specific datum.
  * It does not compute a verdict ("eligible should be false for this
    call"); it presents the criteria from the doc and lets the LLM apply
    them. The LLM remains the sole decision-maker about every tool call's
    argument values.

Stable structure captured
-------------------------
  * Activation predicate is keyed on the tau2-framework tool names declared
    in ``tau2-bench-src/src/tau2/domains/banking_knowledge/tools.py`` —
    ``unlock_discoverable_agent_tool`` and ``call_discoverable_agent_tool``
    — plus the discoverable workflow tool name
    ``file_credit_card_transaction_dispute_4829`` (the
    ``@is_discoverable_tool(ToolType.WRITE)`` decorated method around
    line 787 of the same file). These are tau2 system facts independent of
    any specific task or simulation.
  * The eligibility criteria are a verbatim transcription of doc _015's
    "Eligibility Criteria" and "NOT Eligible Scenarios" sections — the
    same authoritative policy text already in the system prompt. The
    advisor cites the doc by id and re-surfaces the criteria at the
    decision moment; it does not interpret, summarize, or override the
    policy.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional


# tau2 framework entry-point tool names. These wrap discoverable tools and
# carry the inner ``agent_tool_name`` we want to inspect.
_AGENT_ENTRY_POINTS = {
    "unlock_discoverable_agent_tool",
    "call_discoverable_agent_tool",
}


# The discoverable WRITE tool whose ``eligible_for_provisional_credit`` field
# is governed by KB doc _015. Declared at
# tau2-bench-src/src/tau2/domains/banking_knowledge/tools.py:787.
DISPUTE_TOOL = "file_credit_card_transaction_dispute_4829"


# KB document id (already injected into the system prompt by the
# iter1/iter2 internal_procedure_channel).
KB_DOC_ID = "doc_credit_cards_credit_cards_(general)_015"


def _parse_args(raw: Any) -> dict:
    """Best-effort parse of a tool_call ``arguments`` field. Accepts either
    a dict (current tau2 convention) or a JSON string (legacy serialized
    traces). Returns an empty dict on any malformed input.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def _inner_agent_tool(args: dict) -> Optional[str]:
    name = args.get("agent_tool_name")
    return name if isinstance(name, str) else None


def is_dispute_entry_point(tool_call) -> bool:
    """Return True iff ``tool_call`` is an
    ``unlock_discoverable_agent_tool`` or ``call_discoverable_agent_tool``
    invocation whose inner ``agent_tool_name`` equals
    ``file_credit_card_transaction_dispute_4829``.

    Activation is purely structural: the predicate inspects only tool-call
    metadata (the outer entry-point name and the inner ``agent_tool_name``
    argument). It never branches on message content, task structure, or
    observed failure events.
    """
    name = getattr(tool_call, "name", None)
    if name not in _AGENT_ENTRY_POINTS:
        return False
    args = _parse_args(getattr(tool_call, "arguments", None))
    return _inner_agent_tool(args) == DISPUTE_TOOL


def lookup_producing_tool_call(tool_call_id: str, history_messages: Iterable):
    """Return the assistant ToolCall that produced ``tool_call_id``.

    Walks history in reverse looking for an AssistantMessage whose
    ``tool_calls`` contains a matching ``id``. Returns None if not found.
    """
    if not tool_call_id:
        return None
    for prev in reversed(list(history_messages)):
        tool_calls = getattr(prev, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in tool_calls:
            if getattr(tc, "id", None) == tool_call_id:
                return tc
    return None


def render_advisory() -> str:
    """Render the provisional-credit eligibility checklist for the
    ``file_credit_card_transaction_dispute_4829`` entry-point.

    The text:
      - Cites KB doc id _015 (already loaded in the system prompt).
      - Lists the five eligibility criteria from the doc's "Eligibility
        Criteria" section as written.
      - Notes that criteria 2 (reason category) and 5 (non-fraud merchant
        contact) are decidable from the call's own arguments alone.
      - Includes the doc's own "NOT Eligible Scenarios" worked-example
        bullets so the LLM has the policy text re-surfaced at the
        decision point.
      - Does NOT direct a specific value for any field; the LLM remains
        the sole decision-maker about every argument.
    """
    lines = [
        "[PROVISIONAL CREDIT CHECKLIST] "
        f"{DISPUTE_TOOL}.eligible_for_provisional_credit is a strict policy "
        "field (also mirrored into provisional_credit_given) recorded "
        "verbatim into transaction_disputes for the DB hash.",
        f"  KB doc: {KB_DOC_ID} (already loaded in your system prompt).",
        "  Set eligible_for_provisional_credit = TRUE only when ALL of:",
        "    1. credit-card account opened ≥ 60 days ago;",
        "    2. dispute_reason ∈ {unauthorized_fraudulent_charge,",
        "       duplicate_charge, goods_services_not_received "
        "(purchase > 30 days ago)};",
        "    3. transaction amount $25.00 ≤ x ≤ card-tier provisional max",
        "       (Entry 2500 / Mid 5000 / Premium 10000 / Elite 15000 / "
        "Invitation 25000);",
        "    4. customer has filed ≤ 2 disputes in the past 12 months;",
        "    5. for any NON-FRAUD reason (any dispute_reason ≠",
        "       unauthorized_fraudulent_charge, INCLUDING duplicate_charge"
        " and goods_services_not_received): contacted_merchant MUST be true.",
        "  Otherwise set eligible_for_provisional_credit = FALSE. The doc's",
        "  NOT-Eligible scenarios include: dispute_reason ∈ "
        "{incorrect_amount,",
        "  goods_services_not_as_described, canceled_subscription_still_"
        "charging, refund_never_processed}; transaction amount under $25.00"
        " or above card-tier max; any non-fraud reason with",
        "  contacted_merchant=false; goods_services_not_received with"
        " purchase < 30 days ago.",
        "  Criteria 2 (reason category) and 5 (non-fraud merchant contact)",
        "  are decidable from this dispute call's OWN arguments alone --",
        "  no extra read tool is needed to apply them.",
    ]
    return "\n".join(lines)
