"""Unit tests for v1 deterministic reducers."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from agent.v1.reducers.extract import extract_final_answer  # noqa: E402
from agent.v1.reducers.normalize import normalize_answer  # noqa: E402
from agent.v1.reducers.route import route_by_extras  # noqa: E402


# ---------------------------------------------------------------------------
# normalize_answer
# ---------------------------------------------------------------------------
def test_normalize_strips_prefix_and_quotes():
    assert normalize_answer('Final Answer: "yellow".') == "yellow"
    assert normalize_answer("Answer: 42") == "42"


def test_normalize_passes_clean_strings():
    assert normalize_answer("yellow") == "yellow"
    assert normalize_answer("1.456") == "1.456"
    assert normalize_answer(None) == ""


def test_normalize_keeps_numeric_decimal_period():
    assert normalize_answer("1.456") == "1.456"


def test_normalize_collapses_whitespace():
    assert normalize_answer("  Michele   Fitzgerald  ") == "Michele Fitzgerald"


# ---------------------------------------------------------------------------
# extract_final_answer
# ---------------------------------------------------------------------------
def test_extract_terse_passthrough():
    assert extract_final_answer("yellow") == "yellow"
    assert extract_final_answer("42") == "42"


def test_extract_picks_final_answer_line():
    txt = "Reasoning paragraph.\nMore stuff.\nFinal answer: 116"
    assert extract_final_answer(txt) == "116"


def test_extract_boxed():
    assert extract_final_answer("...\n\\boxed{Paris}") == "Paris"


def test_extract_empty():
    assert extract_final_answer("") == ""
    assert extract_final_answer(None) == ""


# ---------------------------------------------------------------------------
# route_by_extras
# ---------------------------------------------------------------------------
def test_route_file_task():
    assert route_by_extras("foo", {"file_name": "x.xlsx"}) == "NEEDS_FILE"


def test_route_direct_task():
    assert route_by_extras("What is 2+2?", {}) == "DIRECT"


def test_route_url_triggers_retrieval():
    q = "Check https://example.com/ and tell me the title."
    assert route_by_extras(q, {}) == "NEEDS_RETRIEVAL"


def test_route_citation_phrase_triggers_retrieval():
    q = "According to the article in question, how many?"
    assert route_by_extras(q, {}) == "NEEDS_RETRIEVAL"


def test_route_naked_domain_triggers_retrieval():
    q = "Look up the entry on example.org and report the value."
    assert route_by_extras(q, {}) == "NEEDS_RETRIEVAL"
