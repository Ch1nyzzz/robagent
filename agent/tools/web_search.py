"""web_search tool — query a search engine and return top results.

Provider selection by env var priority:
  TAVILY_API_KEY     → Tavily Search API
  SERPER_API_KEY     → Serper (google.serper.dev)
  (else)             → DuckDuckGo via duckduckgo_search package (no key needed)

Returns a formatted list of (title, url, snippet) entries.
"""
from __future__ import annotations

import os
from typing import Any


_NUM_RESULTS_DEFAULT = 5
_MAX_RESULTS = 10


SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for a query and return the top results as text "
            "(title, url, snippet). Use for general knowledge lookup, finding "
            "documents, identifying entities. Follow up with url_fetch on a "
            "result URL when you need full content."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
                "num_results": {
                    "type": "integer",
                    "description": f"Number of results to return (1..{_MAX_RESULTS}). Default {_NUM_RESULTS_DEFAULT}.",
                },
            },
            "required": ["query"],
        },
    },
}


def _fmt(results: list[dict]) -> str:
    if not results:
        return "[no results]"
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title") or "(no title)"
        url = r.get("url") or ""
        snippet = r.get("snippet") or ""
        lines.append(f"{i}. {title}\n   {url}\n   {snippet}")
    return "\n".join(lines)


def _search_tavily(query: str, n: int) -> list[dict]:
    import requests
    key = os.environ["TAVILY_API_KEY"]
    resp = requests.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"query": query, "max_results": n, "search_depth": "advanced"},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
        for r in data.get("results", [])
    ]


def _search_serper(query: str, n: int) -> list[dict]:
    import requests
    key = os.environ["SERPER_API_KEY"]
    resp = requests.post(
        "https://google.serper.dev/search",
        json={"q": query, "num": n},
        headers={"X-API-KEY": key, "Content-Type": "application/json"},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        {"title": r.get("title", ""), "url": r.get("link", ""), "snippet": r.get("snippet", "")}
        for r in data.get("organic", [])[:n]
    ]


def _search_ddg(query: str, n: int) -> list[dict]:
    from duckduckgo_search import DDGS
    with DDGS() as ddgs:
        raw = list(ddgs.text(query, max_results=n))
    return [
        {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
        for r in raw
    ]


def run(args: dict) -> str:
    query = (args.get("query") or "").strip()
    if not query:
        return "ERROR: web_search requires a non-empty query argument."
    n = max(1, min(int(args.get("num_results") or _NUM_RESULTS_DEFAULT), _MAX_RESULTS))
    providers: list[tuple[str, callable]] = []
    if os.environ.get("TAVILY_API_KEY"):
        providers.append(("tavily", _search_tavily))
    if os.environ.get("SERPER_API_KEY"):
        providers.append(("serper", _search_serper))
    providers.append(("duckduckgo", _search_ddg))
    last_err = None
    for name, fn in providers:
        try:
            results = fn(query, n)
            return f"[provider={name}]\n{_fmt(results)}"
        except Exception as e:
            last_err = f"{name}: {type(e).__name__}: {e}"
            continue
    return f"ERROR: all web_search providers failed. Last: {last_err}"
