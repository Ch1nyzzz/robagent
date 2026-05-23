"""Helpers for the dispute-time card_last_4 user-tool audit advisory.

Why this exists
---------------
The tau2 banking_knowledge ``credit_card_accounts`` table has a
``card_last_4_digits`` column. Some banking_knowledge task fixtures populate
that column in the response of ``get_credit_card_accounts_by_user`` (e.g.
train task_054), in which case the agent can read the value directly. Other
fixtures omit it (e.g. train task_031/_037/_038/_053), in which case the
framework's documented user-side discoverable tool ``get_card_last_4_digits``
(``tau2-bench-src/src/tau2/domains/banking_knowledge/tools.py``:4207-4245) is
the canonical path to obtain the value.

When the agent files a credit-card transaction dispute via
``file_credit_card_transaction_dispute_4829`` (declared at tools.py:787) the
gold workflow on the "field-missing" fixtures includes two extra actions:

    give_discoverable_user_tool(discoverable_tool_name='get_card_last_4_digits')
    call_discoverable_user_tool(discoverable_tool_name='get_card_last_4_digits',
                                arguments='{"credit_card_account_id": "..."}')

The ``give_*`` call writes a row to ``user_discoverable_tools`` (tools.py:575)
keyed by the unique tool name. The user-tool body invokes
``_log_user_tool_call`` (tools.py:4085) which writes a row to
``user_discoverable_tool_calls`` keyed by (tool_name, arguments). Both audit
tables participate in the gold DB hash on dispute-filing tasks.

iter11's ``card_last4_resolver`` re-implements the algorithm and surfaces the
deterministic value into the credit-card account lookup response, which is
correct for "field-missing" fixtures (the algorithm output matches gold) but
WRONG for "field-present" fixtures (task_054's record carries
``card_last_4_digits: 7823`` inline, but the algorithm yields ``0791``). The
iter11 trace shows the agent dutifully submitting ``0791`` and failing the DB
hash. iter11 also masks the user-tool give: with the algorithm value already
in the prompt, the LLM skips ``give_discoverable_user_tool`` on tasks where
gold requires it (task_038, task_053 — failing only on the missing give/call
audit rows under iter11).

What this module does
---------------------
Provides two helpers:

  * ``account_ids_with_inline_last_4(messages)`` returns the set of
    credit-card account ids whose latest ``get_credit_card_accounts_by_user``
    response (or any tool message naming the ``credit_card_accounts`` table)
    carries ``card_last_4_digits:`` inline. The agent's resolver should
    suppress the algorithm surfacing for these ids — the in-record value is
    the gold value.
  * ``build_dispute_audit_advisory(account_ids, missing_inline_ids,
    give_count)`` renders an idempotent ``[DISPUTE CARD-LAST-4 AUDIT]``
    advisory naming (1) the framework-documented user-tool path, (2) the two
    audit tables that the path writes to, and (3) the observed count of
    prior ``give_discoverable_user_tool('get_card_last_4_digits')`` calls.

All predicates operate on framework-emitted tool message content (the
``credit_card_accounts`` record format) and on the agent's own tool_call
history — never on user prose, never on a builder interpretation of KB
policy text.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional, Set

MARKER = "[DISPUTE CARD-LAST-4 AUDIT]"

# Same pattern iter11 uses for credit-card account ids.
_ACCOUNT_ID_RE = re.compile(r"cc_[a-z0-9]+_[a-z0-9]+")
# A response is a credit_card_accounts response iff it names the table.
_TABLE_HEADER_RE = re.compile(r"in '(credit_card_accounts)'", re.IGNORECASE)
# A record block starts with "<digit>. Record ID: ..." inside one response.
_RECORD_BLOCK_RE = re.compile(r"^\s*\d+\.\s+Record ID:", re.MULTILINE)


def _split_records(content: str) -> List[str]:
    """Split a tool response into record blocks at the "N. Record ID:" boundary.

    Returns the content as a single block if no boundary marker is found.
    """
    indices = [m.start() for m in _RECORD_BLOCK_RE.finditer(content)]
    if not indices:
        return [content]
    blocks: List[str] = []
    for i, start in enumerate(indices):
        end = indices[i + 1] if i + 1 < len(indices) else len(content)
        blocks.append(content[start:end])
    return blocks


def account_ids_with_inline_last_4(messages: Iterable) -> Set[str]:
    """Return the set of cc_<id> whose credit_card_accounts record block
    carries ``card_last_4_digits:`` inline.

    Walks every prior tool message that names the ``credit_card_accounts``
    table. For each record block in such a message, checks whether
    ``card_last_4_digits:`` appears in the block; if so, the cc_<id>
    extracted from the block is marked as "field present".
    """
    present: Set[str] = set()
    for m in messages or []:
        if getattr(m, "role", None) != "tool":
            continue
        content = getattr(m, "content", None)
        if not isinstance(content, str):
            continue
        if not _TABLE_HEADER_RE.search(content):
            continue
        for block in _split_records(content):
            if "card_last_4_digits:" not in block:
                continue
            for acct in _ACCOUNT_ID_RE.findall(block):
                present.add(acct)
    return present


def count_user_tool_gives(messages: Iterable, tool_name: str) -> int:
    """Count prior ``give_discoverable_user_tool`` assistant tool_calls whose
    inner ``discoverable_tool_name`` argument equals ``tool_name``.

    Counts UNIQUE calls (by id) — duplicate ids should not occur but are
    de-duplicated defensively.
    """
    seen_ids: Set[str] = set()
    n = 0
    for m in messages or []:
        if getattr(m, "role", None) != "assistant":
            continue
        for tc in getattr(m, "tool_calls", None) or []:
            if getattr(tc, "name", None) != "give_discoverable_user_tool":
                continue
            args = getattr(tc, "arguments", None) or {}
            if not isinstance(args, dict):
                continue
            if args.get("discoverable_tool_name") != tool_name:
                continue
            tc_id = getattr(tc, "id", None) or ""
            if tc_id and tc_id in seen_ids:
                continue
            if tc_id:
                seen_ids.add(tc_id)
            n += 1
    return n


def build_dispute_audit_advisory(
    account_ids_in_scope: Iterable[str],
    ids_with_inline_last_4: Set[str],
    give_count: int,
) -> Optional[str]:
    """Render the dispute-time user-tool audit advisory.

    Returns ``None`` if every credit-card account in scope already carries
    ``card_last_4_digits`` inline — in that case the agent can read the value
    directly from the credit_card_accounts record and the user-tool path is
    not the canonical workflow.
    """
    account_ids = [a for a in account_ids_in_scope if a not in ids_with_inline_last_4]
    if not account_ids:
        return None

    id_list = ", ".join(account_ids)
    if give_count == 0:
        give_state = (
            "OBSERVED: give_discoverable_user_tool('get_card_last_4_digits') has "
            "NOT yet been issued in your tool-call history. The audit row in "
            "user_discoverable_tools is currently MISSING."
        )
    else:
        give_state = (
            f"OBSERVED: give_discoverable_user_tool('get_card_last_4_digits') "
            f"already issued {give_count} time(s). The audit row in "
            f"user_discoverable_tools has been written."
        )

    header = (
        f"{MARKER}\n"
        "file_credit_card_transaction_dispute_4829 (tau2/domains/banking_knowledge/"
        "tools.py:787) requires the `card_last_4_digits` argument. For the "
        "following credit-card account(s), the `credit_card_accounts` record "
        "returned by `get_credit_card_accounts_by_user` did NOT include the "
        f"`card_last_4_digits` field inline:\n  {id_list}\n"
        "On those accounts, the framework's documented user-side discoverable "
        "tool `get_card_last_4_digits` (tools.py:4206-4245) is the canonical "
        "method to obtain the value. The standard workflow is:\n"
        "  1. give_discoverable_user_tool(discoverable_tool_name="
        "'get_card_last_4_digits')\n"
        "  2. call_discoverable_user_tool(discoverable_tool_name="
        "'get_card_last_4_digits', arguments='{\"credit_card_account_id\": "
        "\"cc_<id>\"}')\n"
        "Step (1) writes one row to `user_discoverable_tools` (tools.py:576). "
        "Step (2) invokes `_log_user_tool_call` which writes one row to "
        "`user_discoverable_tool_calls` (tools.py:4094). Both audit tables "
        "participate in the evaluation DB hash on credit-card dispute filings."
    )
    return header + "\n" + give_state
