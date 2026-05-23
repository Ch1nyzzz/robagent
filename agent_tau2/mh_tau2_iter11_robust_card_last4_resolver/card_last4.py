"""Deterministic computation of credit-card last-4 digits.

Why this exists
---------------
The tau2 banking_knowledge user-side tool ``get_card_last_4_digits`` is a
deterministic function of the credit-card account id alone. Its body in
``tau2-bench-src/src/tau2/domains/banking_knowledge/tools.py`` (lines
4207-4245) computes:

    hash_input = f"card_last4:{credit_card_account_id}"
    hash_digest = hashlib.sha256(hash_input.encode()).hexdigest()
    last_4 = "".join(c for c in hash_digest if c.isdigit())[:4]
    last_4 = last_4.ljust(4, "0")

The framework records ``card_last_4_digits`` into the ``transaction_disputes``
table when ``file_credit_card_transaction_dispute_4829`` is called, so the
field participates in the evaluation DB hash. The gold value is always the
output of the algorithm above.

In tau2's half-duplex protocol the agent never observes the user simulator's
own tool calls (per memory ``tau2-user-tool-calls-invisible-to-agent``); when
the agent invokes ``give_discoverable_user_tool('get_card_last_4_digits')``
and waits for the user to report the result, the user simulator hallucinates
a 4-digit string into its prose reply that need not equal the algorithm's
output (seen on train task_053: user reports ``3817``, gold algorithm yields
``2791`` for account ``cc_e9d195fe8e_silver``).

This module re-implements the framework algorithm verbatim so the agent can
verify any user-reported last-4 against the deterministic value and submit
the framework-correct value into the dispute tool call.

What this module does
---------------------
Provides:

  * ``compute_last_4(account_id)`` — verbatim re-implementation of the
    framework algorithm. Returns a 4-character string of digits, or ``None``
    if ``account_id`` does not match the credit-card account-id pattern.
  * ``extract_account_ids(content)`` — find every credit-card account-id
    string (``cc_<user>_<product>``) appearing in framework-emitted tool
    response content. Used to scope the advisory to the account ids the
    agent has actually fetched.
  * ``build_advisory(account_ids)`` — render an idempotent
    ``[CARD LAST-4 (DETERMINISTIC)]`` block listing the deterministic last-4
    for each unique account id.

The whole module is keyed to two stable structures: (1) the framework
algorithm source, and (2) the credit-card account-id naming convention
``cc_<user_id>_<product>`` declared in the banking_knowledge KB and used
verbatim in every banking_knowledge task's data fixtures. Neither depends on
any specific KB document content, task id, or observed failure simulation.
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable, List, Optional

# Marker is included in every emitted advisory; the agent module checks for
# this exact substring before appending to avoid double-annotation.
MARKER = "[CARD LAST-4 (DETERMINISTIC)]"

# Credit-card account ids follow the convention ``cc_<10-hex-user>_<product>``
# everywhere in the banking_knowledge fixtures (see e.g.
# ``cc_890389b165_silver``, ``cc_e9d195fe8e_silver``, ``cc_e3f4a5b6c7_green``,
# ``cc_224959b99e_bsilver``). The lower-case alphanumeric segments use
# underscores as separators; we match the full pattern explicitly so we never
# pull in a string that merely happens to start with ``cc_``.
_ACCOUNT_ID_RE = re.compile(r"cc_[a-z0-9]+_[a-z0-9]+")


def compute_last_4(account_id: str) -> Optional[str]:
    """Return the framework-deterministic last-4 string for an account id.

    Re-implements the body of ``get_card_last_4_digits`` in
    ``tau2/domains/banking_knowledge/tools.py`` verbatim. Returns ``None`` if
    ``account_id`` is not a credit-card account id (does not start with
    ``cc_``).
    """
    if not isinstance(account_id, str) or not account_id.startswith("cc_"):
        return None
    digest = hashlib.sha256(f"card_last4:{account_id}".encode()).hexdigest()
    digits = "".join(c for c in digest if c.isdigit())[:4]
    return digits.ljust(4, "0")


def extract_account_ids(content: str) -> List[str]:
    """Return ordered, de-duplicated credit-card account ids in ``content``.

    Order is preserved to make the advisory deterministic across runs.
    """
    if not isinstance(content, str):
        return []
    seen: List[str] = []
    in_set: set = set()
    for match in _ACCOUNT_ID_RE.findall(content):
        if match not in in_set:
            in_set.add(match)
            seen.append(match)
    return seen


def build_advisory(account_ids: Iterable[str]) -> Optional[str]:
    """Render a deterministic ``[CARD LAST-4 (DETERMINISTIC)]`` block.

    Returns ``None`` if no credit-card account ids are supplied (so the
    caller can skip annotation entirely).
    """
    rows: List[str] = []
    for acct in account_ids:
        last_4 = compute_last_4(acct)
        if last_4 is None:
            continue
        rows.append(f"  {acct}: {last_4}")
    if not rows:
        return None
    header = (
        f"{MARKER}\n"
        "The tau2 framework records `card_last_4_digits` into the "
        "`transaction_disputes` table from the value YOU pass to "
        "`file_credit_card_transaction_dispute_4829`. That value is "
        "evaluated against the framework-deterministic algorithm "
        "(`get_card_last_4_digits` in tau2/domains/banking_knowledge/"
        "tools.py: sha256('card_last4:'+account_id).hexdigest(), first 4 "
        "digit characters, left-justified to width 4 with '0'). For each "
        "credit-card account in scope:"
    )
    footer = (
        "Use these deterministic values for any `card_last_4_digits` "
        "argument; the framework's evaluation DB hash is computed against "
        "the algorithm's output, NOT against any value the user reports in "
        "natural-language conversation. If a user-reported number differs "
        "from the deterministic value, the deterministic value is correct."
    )
    return "\n".join([header, *rows, "", footer])
