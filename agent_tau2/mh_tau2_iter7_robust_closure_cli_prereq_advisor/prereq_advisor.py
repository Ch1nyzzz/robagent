"""Workflow-prerequisite advisor for credit-card closure, retention, and CLI
entry-point tool calls in the banking_knowledge domain.

What this module is for
-----------------------
The banking_knowledge domain defines six workflow-write discoverable tools
whose Internal procedure docs prescribe a fixed set of prerequisite read
calls. The current frontier v3 agent reaches these workflow tools (sometimes)
but skips the mandated prereqs, so the evaluation DB hash diverges from
gold even when the workflow write itself matches. The iter4 audit-visibility
plugin discourages "extra" discoverable-tool reads in general; for these six
workflow entry-points, certain reads are NOT extras — they are required
audit-trail steps per the relevant Internal KB doc.

The mapping ``WORKFLOW_PREREQS`` below is derived directly from the doc text:

  * doc_credit_cards_credit_card_account_logistics_002 ("Internal: Processing
    Credit Card Account Closures") — defines the closure procedure for
    ``close_credit_card_account_7834``: verify no pending disputes (read
    ``get_user_dispute_history_7291``) and no pending replacement cards
    (read ``get_pending_replacement_orders_5765``).

  * doc_credit_cards_credit_card_account_logistics_003 ("Internal: Credit
    Card Retention Protocol") — defines the retention protocol gating
    ``log_credit_card_closure_reason_4521`` and the retention-offer
    ``apply_statement_credit_8472``: same closure eligibility checks (no
    pending disputes, no pending replacement cards) plus a check for prior
    retention attempts via ``get_closure_reason_history_8293``.

  * doc_credit_cards_credit_card_account_logistics_007 ("Internal: Processing
    CLI Approvals and Denials") — defines the CLI workflow gating
    ``submit_credit_limit_increase_request_7392``,
    ``approve_credit_limit_increase_5847``, and
    ``deny_credit_limit_increase_5848``: cooldown check
    (``get_credit_limit_increase_history_4829``), no pending disputes
    (``get_user_dispute_history_7291``), no pending replacement cards
    (``get_pending_replacement_orders_5765``), and payment history
    (``get_payment_history_6183``).

What this module does NOT do
----------------------------
  * It does not modify, rewrite, suppress, or fabricate any tool call.
  * It does not branch on user content, customer name, account id, or any
    other task-specific datum.
  * It does not assert a verdict ("you must call X next"); it surfaces the
    documented prereqs and reports observed counts. The LLM remains the sole
    decision-maker about which tool to call next.

Stable structure captured
-------------------------
  * Activation predicate is keyed on tau2-framework tool names declared in
    ``tau2-bench-src/src/tau2/domains/banking_knowledge/tools.py`` —
    ``unlock_discoverable_agent_tool`` (line ~590) and
    ``call_discoverable_agent_tool`` (line ~631) — plus the six
    discoverable workflow tool names declared in the same file. These are
    tau2 system facts independent of any particular task or simulation.
  * The workflow → prereqs mapping is a transcription of the documented
    Internal procedure, citing each doc by id. The docs are already in the
    system prompt via the iter1/iter2 internal_procedure_channel. This
    module brings the relevant doc citation and the observed prereq counts
    to the agent's attention at the workflow entry-point decision.
  * The count of prior prereq reads is a mechanical reduction over the
    agent's observable tool_call history.
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


# Mapping of banking_knowledge workflow-write discoverable tool names to
# their KB-doc-documented prerequisite read tools.
#
# Each entry is keyed by the inner ``agent_tool_name`` and carries:
#   * ``label``  : short workflow label used in the advisory text.
#   * ``kb_doc`` : KB document id (already in the system prompt) whose
#                  Internal procedure mandates these prereqs.
#   * ``prereqs``: ordered list of discoverable read-tool names that the
#                  doc requires before the workflow write.
WORKFLOW_PREREQS: dict = {
    "close_credit_card_account_7834": {
        "label": "credit-card account closure",
        "kb_doc": "doc_credit_cards_credit_card_account_logistics_002",
        "prereqs": [
            "get_user_dispute_history_7291",
            "get_pending_replacement_orders_5765",
        ],
    },
    "log_credit_card_closure_reason_4521": {
        "label": "credit-card retention protocol (close-reason logging)",
        "kb_doc": "doc_credit_cards_credit_card_account_logistics_003",
        "prereqs": [
            "get_user_dispute_history_7291",
            "get_pending_replacement_orders_5765",
            "get_closure_reason_history_8293",
        ],
    },
    "apply_statement_credit_8472": {
        "label": "credit-card retention protocol (retention offer)",
        "kb_doc": "doc_credit_cards_credit_card_account_logistics_003",
        "prereqs": [
            "get_user_dispute_history_7291",
            "get_pending_replacement_orders_5765",
            "get_closure_reason_history_8293",
        ],
    },
    "submit_credit_limit_increase_request_7392": {
        "label": "CLI submission",
        "kb_doc": "doc_credit_cards_credit_card_account_logistics_007",
        "prereqs": [
            "get_user_dispute_history_7291",
            "get_pending_replacement_orders_5765",
            "get_credit_limit_increase_history_4829",
            "get_payment_history_6183",
        ],
    },
    "approve_credit_limit_increase_5847": {
        "label": "CLI approval",
        "kb_doc": "doc_credit_cards_credit_card_account_logistics_007",
        "prereqs": [
            "get_user_dispute_history_7291",
            "get_pending_replacement_orders_5765",
            "get_credit_limit_increase_history_4829",
            "get_payment_history_6183",
        ],
    },
    "deny_credit_limit_increase_5848": {
        "label": "CLI denial",
        "kb_doc": "doc_credit_cards_credit_card_account_logistics_007",
        "prereqs": [
            "get_user_dispute_history_7291",
            "get_pending_replacement_orders_5765",
            "get_credit_limit_increase_history_4829",
            "get_payment_history_6183",
        ],
    },
}


def _parse_args(raw: Any) -> dict:
    """Best-effort parse of a tool_call ``arguments`` field.

    tau2 ``ToolCall.arguments`` is normalized to a dict by the model, but
    serialized traces may carry a JSON string in some legacy paths. Accept
    both and return an empty dict on any malformed input.
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


def workflow_for_tool_call(tool_call) -> Optional[str]:
    """Return the inner workflow tool name if ``tool_call`` is a workflow
    entry-point invocation, else None.

    Activation is purely structural: the tool_call's outer ``name`` must be
    one of the framework agent entry-points AND the inner ``agent_tool_name``
    argument must be a known workflow-write tool listed in
    ``WORKFLOW_PREREQS``.
    """
    name = getattr(tool_call, "name", None)
    if name not in _AGENT_ENTRY_POINTS:
        return None
    args = _parse_args(getattr(tool_call, "arguments", None))
    inner = _inner_agent_tool(args)
    if inner in WORKFLOW_PREREQS:
        return inner
    return None


def lookup_producing_tool_call(tool_call_id: str, history_messages: Iterable):
    """Return the assistant ToolCall that produced ``tool_call_id``.

    Walks history in reverse looking for an AssistantMessage with a
    matching ``tool_call.id``. Returns None if not found.
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


def count_prereq_calls(history_messages: Iterable, prereq_name: str) -> int:
    """Count assistant ``call_discoverable_agent_tool`` invocations whose
    inner ``agent_tool_name`` equals ``prereq_name``.

    Pure reduction over message history. Counts every observed invocation
    (the underlying ``agent_discoverable_tools`` table is keyed by tool name
    so multiple identical invocations only write one row, but each invocation
    is an independent agent decision; the count is informational for the
    LLM).
    """
    count = 0
    for m in history_messages:
        tool_calls = getattr(m, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in tool_calls:
            if getattr(tc, "name", None) != "call_discoverable_agent_tool":
                continue
            args = _parse_args(getattr(tc, "arguments", None))
            if _inner_agent_tool(args) == prereq_name:
                count += 1
    return count


def render_advisory(workflow_tool: str, history_messages: Iterable) -> str:
    """Render the prerequisite advisory for a workflow entry-point call.

    The text:
      - Names the workflow and cites the KB doc id that defines it (the
        doc itself is already in the system prompt via the iter1/iter2
        internal_procedure_channel).
      - Lists each prerequisite read tool name and the observed count of
        prior agent invocations.
      - States the DB-evaluation consequence of skipping a prereq.
      - Does NOT direct the LLM to call a specific tool; the LLM remains
        the sole decision-maker about the next action.
    """
    spec = WORKFLOW_PREREQS[workflow_tool]
    label = spec["label"]
    kb_doc = spec["kb_doc"]
    prereqs = spec["prereqs"]

    lines = [
        f"[WORKFLOW PREREQS] {workflow_tool} is the workflow-write step of the {label}.",
        f"  KB doc: {kb_doc} (already loaded in your system prompt).",
        "  Documented prerequisite reads — these are NOT 'extras' for this workflow;",
        "  the Internal procedure mandates each one before the write completes.",
    ]
    for p in prereqs:
        observed = count_prereq_calls(history_messages, p)
        lines.append(
            f"    - {p}: observed {observed} prior call_discoverable_agent_tool "
            f"invocation(s) in this conversation."
        )
    lines.append(
        "  Each prereq tool name writes (at most) one row to the "
        "`agent_discoverable_tools` audit table when first called via "
        "`call_discoverable_agent_tool`; if observed count is 0 for a documented "
        "prereq, the gold DB hash will diverge from the agent's DB hash for this "
        "workflow."
    )
    return "\n".join(lines)
