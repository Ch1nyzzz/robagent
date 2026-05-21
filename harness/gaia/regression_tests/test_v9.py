"""Unit tests for v9 — deterministic query extraction and arithmetic verifier."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from agent.v9.llm_nodes.plan_retrieval_v9 import deterministic_query_from_question  # noqa: E402
from agent.v8.tools.arithmetic import safe_arith  # noqa: E402


def test_query_extracts_multi_word_proper_noun():
    q = "What is the population of New York City as of 2020?"
    out = deterministic_query_from_question(q)
    assert "New York" in out or "York City" in out


def test_query_extracts_longest_phrase():
    q = "Look up the entry on Box Office Mojo and report the top movie."
    assert "Box Office Mojo" in deterministic_query_from_question(q)


def test_query_falls_back_to_single_capital_word():
    q = "How many calories are in an apple?"
    out = deterministic_query_from_question(q)
    # "apple" is lowercase, so we'd return the fallback (truncated question)
    assert out  # non-empty


def test_query_handles_empty():
    assert deterministic_query_from_question("") == ""


def test_safe_arith_basic():
    ok, v, _ = safe_arith("1 + 2 * 3")
    assert ok and v == 7


def test_safe_arith_sqrt():
    ok, v, _ = safe_arith("sqrt(16)")
    assert ok and v == 4


def test_safe_arith_rejects_names():
    ok, _, err = safe_arith("__import__('os').system('ls')")
    assert not ok


def test_safe_arith_pi():
    ok, v, _ = safe_arith("pi * 2")
    assert ok and abs(v - 6.283185) < 1e-3
