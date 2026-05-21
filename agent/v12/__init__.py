"""v12 — multi-source deterministic retrieval.

v10 had 24 tasks blocked by `claim_unverified` because the wiki-only path
returned 0 hits. v12 adds two extra free retrieval sources that fan out in
parallel on a wiki miss:

- DuckDuckGo HTML (`duckduckgo_html_search`) — broad web text search, no key
- arXiv API (`arxiv_search`) — preprint metadata + abstracts, no key

A deterministic query-rewriter (`rewrite_query_variants`) generates a
small set of structural template variants when the first query gets 0 hits.
All sources flow through the existing claim/source protocol — no new LLM nodes.
"""
