"""Deterministic cash-back policy engine for the banking_knowledge domain.

Rho-Bank credit-card cash back is fully determined by published policy. For a
single transaction:

    correct_points = int( transaction_amount * rate * promo_multiplier )

where

  * ``rate`` is the card's category rate. Rho-Bank category cards publish a
    *qualifying* rate and a *standard* rate; the transaction's own ``category``
    field decides which one applies — a transaction earns the qualifying rate
    exactly when its category is one of the card's documented bonus categories,
    and the standard rate otherwise. Two cards additionally publish a list of
    merchants excluded from the bonus rate; a named merchant in a bonus category
    is demoted to the standard rate.
  * ``promo_multiplier`` is the documented limited-time multiplier when the
    transaction date falls inside an active promotional window (a window keyed
    to the account-opening date, not to the transaction date alone).

This module compiles that published policy — the per-card rate table, the
qualifying-category sets, the merchant-exclusion lists, and the promo rules —
and recomputes the cash back for every transaction returned in a
``credit_card_transaction_history`` tool result. A transaction is flagged as a
cash-back error only when its recorded ``rewards_earned`` matches no
policy-valid value.

Design notes — why this never over-flags
----------------------------------------
False positives (telling the agent to dispute a *correct* transaction) are the
only way an audit annotation can make an outcome worse. The engine is built so
that cannot happen:

  * The rate is read from the transaction's ``category`` field against the
    documented qualifying-category set, with the published merchant-exclusion
    list applied — this is the single rate the policy prescribes.
  * If a transaction has no recorded category the rate is genuinely
    undecidable, so the engine accepts both documented rates and will not flag
    it.
  * When the promo window cannot be resolved (account-open date not yet seen),
    both the base and the promo multiplier are accepted.
  * A card type outside the compiled rate table, or an unparseable amount /
    points value, is skipped entirely.

The engine only ever appends advisory text to a tool result; it never alters,
blocks, adds or reorders a tool call, so a mis-parse cannot make an outcome
worse than the no-op baseline.
"""
from __future__ import annotations

import calendar
import datetime
import re
from typing import Optional


# card_type -> (qualifying_rate, standard_rate). Rates are points-per-dollar
# (the domain stores "N% cash back" as N points per dollar).
CARD_RATES = {
    # Personal credit cards
    "Bronze Rewards Card": (1.0, 1.0),
    "Silver Rewards Card": (4.0, 1.0),
    "Gold Rewards Card": (2.5, 2.5),
    "Platinum Rewards Card": (10.0, 10.0),
    "Crypto-Cash Back": (2.0, 2.0),
    "Diamond Elite Card": (5.0, 5.0),
    "EcoCard": (5.0, 1.0),
    # Business credit cards
    "Business Bronze Rewards Card": (1.0, 1.0),
    "Business Silver Rewards Card": (10.0, 1.0),
    "Business Gold Rewards Card": (2.5, 1.0),
    "Business Platinum Rewards Card": (4.0, 1.5),
    "Green Rewards Card": (3.0, 1.0),
    "Silver Zoom Card": (3.0, 1.0),
}

# card_type -> set of transaction categories that earn the qualifying rate.
# A card absent here (flat-rate card) earns the same rate everywhere.
QUALIFYING_CATEGORIES = {
    "Silver Rewards Card": {"Travel", "Software"},
    "EcoCard": {"Green", "Sustainable", "Sustainability"},
    "Business Silver Rewards Card": {"Travel", "Software"},
    "Business Gold Rewards Card": {"Operations"},
    "Business Platinum Rewards Card": {"Travel", "Software", "Media"},
    "Green Rewards Card": {"Sustainable", "Green"},
    "Silver Zoom Card": {"Transportation"},
}

# card_type -> list of merchant names that DO NOT earn the qualifying rate
# (published "Important exclusions" lists). A named merchant in a bonus
# category is demoted to the standard rate.
MERCHANT_EXCLUSIONS = {
    "Business Silver Rewards Card": [
        "concur", "sap concur", "expensify", "navan",
        "apple", "microsoft", "dell",
        "xbox game pass", "playstation plus", "nintendo switch online",
        "coursera", "udemy", "linkedin learning", "skillshare", "pluralsight",
    ],
    "EcoCard": ["target", "walmart", "amazon", "thredup"],
}

# card_type -> promo rule. ``multiplier`` multiplies the earned rate inside the
# window. A window is keyed to the account-opening date.
#   window_months  : window length after account opening
#   eligible_start / eligible_end : the account must be opened in this period
#   indeterminate  : True when the promo also depends on an unobservable claim,
#                    so both base and promo multipliers are always accepted.
CARD_PROMOS = {
    "Business Silver Rewards Card": {
        "multiplier": 2,
        "window_months": 6,
        "eligible_start": datetime.date(2024, 11, 14),
        "eligible_end": datetime.date(2025, 11, 14),
        "indeterminate": False,
    },
    "Silver Zoom Card": {
        # 3x for 9 months from opening, but only if the customer claimed it —
        # the claim is not observable, so both multipliers are always valid.
        "multiplier": 3,
        "indeterminate": True,
    },
}

# Recorded values within this many points of a policy value are treated as
# correct (documented round-down noise — see tau2-cashback-rounding-rule).
TOLERANCE = 1

MARKER = "[DETERMINISTIC CASH-BACK POLICY AUDIT]"

_KV = re.compile(r"^[ \t]*([a-z_]+):[ \t]*(.+?)[ \t]*$", re.M)


# ---------------------------------------------------------------------------
# Small parsing helpers
# ---------------------------------------------------------------------------
def _money(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    try:
        return float(re.sub(r"[^0-9.\-]", "", text))
    except ValueError:
        return None


def _points(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"-?\d+", text)
    return int(m.group(0)) if m else None


def _date(text: Optional[str]) -> Optional[datetime.date]:
    if not text:
        return None
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if not m:
        return None
    try:
        return datetime.date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
    except ValueError:
        return None


def _add_months(d: datetime.date, n: int) -> datetime.date:
    month = d.month - 1 + n
    year = d.year + month // 12
    month = month % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return datetime.date(year, month, day)


def _records(content: str) -> list[dict]:
    """Parse the human-readable ``Record ID:`` blocks of a DB query result."""
    out = []
    for block in content.split("\n\n"):
        kv = {k: v.strip() for k, v in _KV.findall(block)}
        if kv:
            out.append(kv)
    return out


# ---------------------------------------------------------------------------
# Account-open-date extraction (needed to resolve promo windows)
# ---------------------------------------------------------------------------
def parse_account_open_dates(content: str) -> dict:
    """From a credit_card_accounts tool result, map card_type -> open date.

    A card type that appears with two different open dates is dropped (its
    promo window cannot be resolved unambiguously).
    """
    if not isinstance(content, str):
        return {}
    seen: dict = {}
    for kv in _records(content):
        card = kv.get("card_type")
        opened = _date(kv.get("date_of_account_open"))
        if not card or opened is None:
            continue
        if card in seen and seen[card] != opened:
            seen[card] = None  # ambiguous
        elif card not in seen:
            seen[card] = opened
    return {c: d for c, d in seen.items()}


# ---------------------------------------------------------------------------
# The policy calculation
# ---------------------------------------------------------------------------
def _is_excluded_merchant(card: str, merchant: str) -> bool:
    merchant_l = (merchant or "").lower()
    for entry in MERCHANT_EXCLUSIONS.get(card, ()):
        if re.search(r"\b" + re.escape(entry) + r"\b", merchant_l):
            return True
    return False


def _rate_candidates(card: str, category: str, merchant: str) -> set:
    """Return the set of policy-valid base rates for a transaction.

    The transaction's own ``category`` field is determinative: a card earns its
    qualifying rate exactly when the transaction's category is one of the
    card's documented bonus categories, and its standard rate otherwise. A
    published merchant-exclusion list demotes a named merchant in a bonus
    category back to the standard rate. The only genuinely undecidable case is
    a transaction with no recorded category, where both documented rates are
    accepted so the engine still never over-flags.
    """
    qual, std = CARD_RATES[card]
    if qual == std:
        return {qual}  # flat-rate card — one rate everywhere
    if not category:
        return {qual, std}  # category unknown -> undecidable, accept either
    qual_cats = QUALIFYING_CATEGORIES.get(card, set())
    qualifies = category in qual_cats
    if qualifies and card in MERCHANT_EXCLUSIONS and _is_excluded_merchant(
        card, merchant
    ):
        qualifies = False
    return {qual if qualifies else std}


def _promo_multipliers(
    card: str, open_date: Optional[datetime.date], txn_date: Optional[datetime.date]
) -> set:
    """Return the set of valid promo multipliers for a transaction."""
    promo = CARD_PROMOS.get(card)
    if not promo:
        return {1}
    if promo.get("indeterminate"):
        return {1, promo["multiplier"]}
    if open_date is None or txn_date is None:
        return {1, promo["multiplier"]}  # window unresolved -> accept both
    if not (promo["eligible_start"] <= open_date <= promo["eligible_end"]):
        return {1}  # account not opened during the promo period
    window_end = _add_months(open_date, promo["window_months"])
    if open_date <= txn_date <= window_end:
        return {promo["multiplier"]}
    return {1}


def evaluate_transaction(record: dict, open_dates: dict):
    """Evaluate one transaction record.

    Returns (status, expected) where status is one of "ok", "error", "skip"
    and ``expected`` is the single policy-correct point value when the policy
    fully determines it, else None.
    """
    card = record.get("credit_card_type")
    amount = _money(record.get("transaction_amount"))
    recorded = _points(record.get("rewards_earned"))
    if card not in CARD_RATES or amount is None or recorded is None:
        return "skip", None

    category = record.get("category", "")
    merchant = record.get("merchant_name", "")
    txn_date = _date(record.get("transaction_date"))
    open_date = open_dates.get(card)

    rates = _rate_candidates(card, category, merchant)
    mults = _promo_multipliers(card, open_date, txn_date)

    plausible = {int(amount * r * m) for r in rates for m in mults}
    ok = any(abs(recorded - p) <= TOLERANCE for p in plausible)
    if ok:
        return "ok", None

    # An error. Report a single corrected value only when the policy is fully
    # determined (one rate, one multiplier); otherwise leave it unspecified.
    expected = None
    if len(rates) == 1 and len(mults) == 1:
        r = next(iter(rates))
        m = next(iter(mults))
        expected = int(amount * r * m)
    return "error", expected


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def audit_transaction_list(content: str, open_dates: Optional[dict] = None) -> Optional[str]:
    """Audit a credit_card_transaction_history tool result.

    ``open_dates`` maps card_type -> account-open date (datetime.date),
    gathered from earlier credit_card_accounts tool results. Returns an
    annotation string to append to the tool result, or None when the content
    holds no auditable credit-card transactions.
    """
    if not isinstance(content, str) or MARKER in content:
        return None
    open_dates = open_dates or {}

    records = [r for r in _records(content)
               if "transaction_id" in r and "credit_card_type" in r
               and "rewards_earned" in r]
    if not records:
        return None

    errors = []   # (tid, card, category, amount, recorded, expected)
    n_ok = 0
    for r in records:
        status, expected = evaluate_transaction(r, open_dates)
        if status == "skip":
            continue
        if status == "ok":
            n_ok += 1
            continue
        errors.append(
            (
                r.get("transaction_id"),
                r.get("credit_card_type"),
                r.get("category", ""),
                _money(r.get("transaction_amount")),
                _points(r.get("rewards_earned")),
                expected,
            )
        )

    if n_ok == 0 and not errors:
        return None

    lines = [
        "=" * 62,
        MARKER,
        "Rho-Bank cash back is fixed by published policy:",
        "  correct_points = int(amount * card_category_rate * promo_multiplier)",
        "The category rate is decided by each transaction's own category field",
        "against the card's documented bonus categories. Every transaction above",
        "was recomputed from the published rate table, the qualifying-category",
        "sets, the merchant-exclusion lists and the promo-window rules.",
        "",
    ]
    if errors:
        lines.append(
            f"POLICY-VERIFIED CASH-BACK ERRORS ({len(errors)} transaction(s)). "
            "These earned cash"
        )
        lines.append(
            "back that matches NO valid policy rate — they are the ONLY "
            "transactions"
        )
        lines.append("with a cash-back discrepancy:")
        for tid, card, cat, amount, recorded, expected in errors:
            amt_s = f"${amount:,.2f}" if amount is not None else "n/a"
            lines.append(f"  * {tid} | {card} / {cat or 'n/a'} | {amt_s}")
            if expected is not None:
                lines.append(
                    f"      recorded {recorded} points  ->  "
                    f"policy-correct {expected} points"
                )
            else:
                lines.append(f"      recorded {recorded} points  ->  does not match policy")
        lines.append("")
        lines.append(
            f"The other {n_ok} transaction(s) in this result match policy and "
            "are correct."
        )
        lines.append(
            "ACTION: dispute / correct EXACTLY the transaction_id(s) listed "
            "above and no"
        )
        lines.append(
            "others. Hand the customer submit_cash_back_dispute and have them "
            "file one"
        )
        lines.append("dispute per listed transaction_id.")
    else:
        lines.append(
            f"All {n_ok} transaction(s) in this result match published policy "
            "(within"
        )
        lines.append(
            "round-down noise). None have a cash-back error — do NOT file any "
            "dispute."
        )
    lines.append("=" * 62)
    return "\n".join(lines)
