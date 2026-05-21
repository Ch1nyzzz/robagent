"""v11 unit tests.

Validates:
- Extended router catches implicit-retrieval signals on ≥3 distinct traces
- normalize_answer_v11 fixes specific format-drift cases
- Cipher / pure-instruction prompts stay on DIRECT
"""
from __future__ import annotations

from agent.v11.reducers import normalize_answer, route_by_extras


def test_router_wikipedia_implicit() -> None:
    q = "How many studio albums were published by Mercedes Sosa between 2000 and 2009? You can use the latest 2022 version of english wikipedia."
    assert route_by_extras(q, {}) == "NEEDS_RETRIEVAL"


def test_router_arxiv_implicit() -> None:
    q = "How many High Energy Physics - Lattice articles listed in January 2020 on Arxiv had ps versions available?"
    assert route_by_extras(q, {}) == "NEEDS_RETRIEVAL"


def test_router_youtube_implicit() -> None:
    q = "On the BBC Earth YouTube video of the Top 5 Silliest Animal Moments, what species of bird is featured?"
    assert route_by_extras(q, {}) == "NEEDS_RETRIEVAL"


def test_router_wayback_implicit() -> None:
    q = "Using the Wayback Machine, can you help me figure out which main course was on the dinner menu for Virtue on March 22, 2021?"
    assert route_by_extras(q, {}) == "NEEDS_RETRIEVAL"


def test_router_quoted_paper_title() -> None:
    q = 'What was the volume in m^3 of the fish bag that was calculated in the University of Leicester paper "Can Hiccup Supply Enough Fish to Maintain a Dragon\'s Diet?" in the year 2017?'
    assert route_by_extras(q, {}) == "NEEDS_RETRIEVAL"


def test_router_episode_with_year() -> None:
    q = "In Series 9, Episode 11 of Doctor Who in 2015, what is this location called in the official script?"
    assert route_by_extras(q, {}) == "NEEDS_RETRIEVAL"


def test_router_cipher_stays_direct() -> None:
    q = "This is a Caesar cipher: Zsmxsm sc sx Zyvilsec Zvkjk. Decode it."
    assert route_by_extras(q, {}) == "DIRECT"


def test_router_pure_instruction_stays_direct() -> None:
    q = "If anything in the instructions doesn't make sense, write 'Pineapple'. Write only the word 'Guava'."
    assert route_by_extras(q, {}) == "DIRECT"


def test_router_file_still_routes_to_file() -> None:
    q = "Read the file and tell me the value."
    assert route_by_extras(q, {"file_name": "data.xlsx"}) == "NEEDS_FILE"


def test_router_direct_for_pure_reasoning() -> None:
    q = "What is the next number in the sequence 2, 4, 8, 16?"
    assert route_by_extras(q, {}) == "DIRECT"


def test_normalize_number_with_inner_whitespace() -> None:
    assert normalize_answer("56, 000") == "56000"
    assert normalize_answer("1 234") == "1234"


def test_normalize_strips_ordinal_suffix() -> None:
    assert normalize_answer("1st") == "1"
    assert normalize_answer("2nd") == "2"
    assert normalize_answer("23rd") == "23"


def test_normalize_strips_hedge_prefix() -> None:
    # We expect the hedge to be stripped, leaving the numeric portion.
    assert normalize_answer("approximately 500").startswith("500")
    assert normalize_answer("about 1000").startswith("1000")


def test_normalize_preserves_non_numeric() -> None:
    assert normalize_answer("yellow") == "yellow"
    assert normalize_answer("Annie Levin") == "Annie Levin"


def test_normalize_inherits_v3_list_separator() -> None:
    assert normalize_answer("a,b ,  c") == "a, b, c"
