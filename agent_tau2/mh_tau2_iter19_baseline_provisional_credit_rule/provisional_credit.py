"""Deterministic provisional-credit eligibility rule for credit-card
transaction disputes (banking_knowledge domain).

Source of truth: the knowledge-base article "Provisional Credit Eligibility
Guidelines (Internal)". A credit-card transaction dispute is ELIGIBLE for
provisional credit only when ALL of the following hold:

  1. the credit-card account has been open for >= 60 days;
  2. the dispute reason is one of unauthorized_fraudulent_charge,
     duplicate_charge, or goods_services_not_received (the last only when the
     purchase was made more than 30 days ago);
  3. the transaction amount is between $25.00 and the card tier's maximum;
  4. the customer has filed no more than 2 disputes in the past 12 months;
  5. for any NON-FRAUD dispute (reason other than
     unauthorized_fraudulent_charge) the customer must have contacted the
     merchant first.

This module enforces only the subset of those criteria that is *fully
decidable from a file_credit_card_transaction_dispute call's own arguments*:

  * criterion 2 — the dispute reason must be a provisional-eligible category;
  * criterion 5 — a non-fraud dispute must have contacted_merchant == true.

Both of those can only ever prove a dispute INELIGIBLE; neither can prove one
eligible. The other criteria (account age, transaction amount, prior-dispute
count, and the goods_services_not_received 30-day window, which needs the
current date) are deliberately left to the agent, since they cannot be decided
from the call alone.

The corrector is therefore strictly one-directional: it flips
eligible_for_provisional_credit from true to false when the policy clearly
forbids provisional credit, and never the other way. A genuinely eligible
dispute (a fraud charge, or a non-fraud charge with contacted_merchant true)
matches none of the flip conditions, so a correct call is never altered.
"""
from __future__ import annotations

import json
from typing import Any

# The discoverable tool that files a credit-card transaction dispute. Matched
# by substring so a changed numeric suffix still resolves. It is specific
# enough not to collide with get_user_dispute_history or submit_cash_back_dispute.
DISPUTE_TOOL_SUBSTRING = "file_credit_card_transaction_dispute"

FRAUD_REASON = "unauthorized_fraudulent_charge"

# Criterion 2 — the only dispute reasons that can qualify for provisional
# credit at all. Every other reason is ineligible regardless of other facts.
PROVISIONAL_ELIGIBLE_REASONS = {
    "unauthorized_fraudulent_charge",
    "duplicate_charge",
    "goods_services_not_received",
}


def _is_false(value: Any) -> bool:
    """True only when `value` clearly represents boolean false."""
    if value is False:
        return True
    if isinstance(value, str) and value.strip().lower() in ("false", "no"):
        return True
    return False


def _is_true(value: Any) -> bool:
    """True only when `value` clearly represents boolean true."""
    if value is True:
        return True
    if isinstance(value, str) and value.strip().lower() in ("true", "yes"):
        return True
    return False


def clearly_ineligible(args: dict) -> bool:
    """Return True iff the dispute call's own arguments PROVE the dispute is
    ineligible for provisional credit. Conservative: returns False whenever a
    criterion cannot be decided from the arguments alone."""
    reason = args.get("dispute_reason")
    if not isinstance(reason, str):
        return False
    reason = reason.strip()

    # Criterion 2 — the reason must be a provisional-eligible category.
    if reason not in PROVISIONAL_ELIGIBLE_REASONS:
        return True

    # Criterion 5 — a non-fraud dispute requires contacting the merchant first.
    if reason != FRAUD_REASON and _is_false(args.get("contacted_merchant")):
        return True

    return False


def correct_dispute_arguments(args: dict) -> bool:
    """Mutate a file_credit_card_transaction_dispute argument dict in place,
    flipping eligible_for_provisional_credit true -> false when policy forbids
    it. Returns True when a change was made."""
    if not isinstance(args, dict):
        return False
    if not _is_true(args.get("eligible_for_provisional_credit")):
        return False
    if clearly_ineligible(args):
        args["eligible_for_provisional_credit"] = False
        return True
    return False


def correct_tool_call(tool_call) -> bool:
    """Inspect one tau2 ToolCall; if it files a credit-card transaction dispute
    with an over-stated provisional-credit flag, correct it in place. Returns
    True when a change was made. Never raises."""
    try:
        name = getattr(tool_call, "name", "") or ""
        arguments = getattr(tool_call, "arguments", None)
        if not isinstance(arguments, dict):
            return False

        # Case 1 — the dispute tool is called directly by name.
        if DISPUTE_TOOL_SUBSTRING in name:
            return correct_dispute_arguments(arguments)

        # Case 2 — the call is routed through the discoverable-tool dispatcher.
        inner_name = arguments.get("agent_tool_name") or arguments.get(
            "discoverable_tool_name"
        )
        if not isinstance(inner_name, str) or DISPUTE_TOOL_SUBSTRING not in inner_name:
            return False

        inner = arguments.get("arguments")
        if isinstance(inner, dict):
            return correct_dispute_arguments(inner)
        if isinstance(inner, str):
            try:
                parsed = json.loads(inner)
            except (ValueError, TypeError):
                return False
            if not isinstance(parsed, dict):
                return False
            if correct_dispute_arguments(parsed):
                arguments["arguments"] = json.dumps(parsed)
                return True
        return False
    except Exception:
        return False
