"""v12 unit tests — deterministic query rewriting + multi-source tool stubs.

Network calls are NOT executed here; we only verify the deterministic surface.
"""
from __future__ import annotations

from agent.v12.reducers import rewrite_query_variants


def test_rewrite_drops_modifier_clauses() -> None:
    q = "How many studio albums were published by Mercedes Sosa between 2000 and 2009?"
    out = rewrite_query_variants(q, "Mercedes Sosa albums between 2000 and 2009")
    assert "Mercedes Sosa albums between 2000 and 2009" in out  # base preserved
    # Some variant must drop the modifier clause
    assert any("between" not in v.lower() for v in out)


def test_rewrite_extracts_proper_noun() -> None:
    q = "What was the volume in m^3 of the fish bag from the University of Leicester paper?"
    out = rewrite_query_variants(q, "")
    assert any("Leicester" in v for v in out)


def test_rewrite_dedupes() -> None:
    q = "Foo Bar"
    out = rewrite_query_variants(q, "Foo Bar")
    assert len(out) == len(set(v.lower() for v in out))


def test_rewrite_extracts_quoted_title() -> None:
    q = 'In the paper "Hidden Markov Models for Speech" what was the F1?'
    out = rewrite_query_variants(q, "")
    assert any("Hidden Markov" in v for v in out)


def test_rewrite_returns_at_most_k() -> None:
    q = "How many edits to the Wikipedia page on Cats from 2010 to 2020 about the breed Persian Cats in California?"
    out = rewrite_query_variants(q, "wiki edits cats", k=4)
    assert len(out) <= 4


def test_rewrite_empty_inputs() -> None:
    assert rewrite_query_variants("", "") == []


def test_v12_router_inherits_v11() -> None:
    # v12 reducers must still expose the v11 router behavior
    from agent.v12.reducers import route_by_extras
    q = "How many High Energy Physics - Lattice articles listed in January 2020 on Arxiv had ps versions available?"
    assert route_by_extras(q, {}) == "NEEDS_RETRIEVAL"


def test_v12_normalize_inherits_v11() -> None:
    from agent.v12.reducers import normalize_answer
    assert normalize_answer("56, 000") == "56000"
    assert normalize_answer("1st") == "1"
