"""DuckDuckGo HTML search — no API key required.

DuckDuckGo serves an HTML endpoint at https://html.duckduckgo.com/html/ that
returns a list of search results parseable with simple regex. We treat each
result's snippet as a candidate source body (truncated). On success, emit
`source.opened{kind=duckduckgo}` so the (claim, source) protocol can use it.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any
from urllib.parse import quote_plus

import httpx

from agent.events import EventLog


_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; robagent-harness/1.0)",
    "Accept": "text/html,application/xhtml+xml",
}
_TIMEOUT = httpx.Timeout(20.0, connect=10.0)
_MAX_BODY_CHARS = 4000

# Two anchor patterns observed across the DuckDuckGo HTML response.
_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)
_SNIPPET_RE = re.compile(
    r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)


def _strip_tags(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def _quote_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def duckduckgo_html_search(
    query: str,
    *,
    log: EventLog,
    parent: str,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Return a list of {source_id, url, title, content, ok=True} on success."""
    call_id = log.emit(
        "tool.called",
        parent=parent,
        tool="duckduckgo_html_search",
        args={"query": query, "limit": limit},
    )
    url = "https://html.duckduckgo.com/html/?q=" + quote_plus(query)
    try:
        with httpx.Client(headers=_HEADERS, timeout=_TIMEOUT, follow_redirects=True) as cli:
            r = cli.get(url)
            r.raise_for_status()
        html = r.text or ""
        anchors = _RESULT_RE.findall(html)
        snippets = _SNIPPET_RE.findall(html)
        out: list[dict[str, Any]] = []
        for i, (href, title_html) in enumerate(anchors[:limit]):
            title = _strip_tags(title_html)
            snippet = _strip_tags(snippets[i]) if i < len(snippets) else ""
            body = (title + ". " + snippet)[:_MAX_BODY_CHARS]
            if not body:
                continue
            h = _quote_hash(body + href)
            source_id = f"S_ddg_{h}"
            log.emit(
                "source.opened",
                parent=call_id,
                source_id=source_id,
                kind="duckduckgo",
                url=href,
                title=title,
                content_chars=len(body),
                content_hash=h,
            )
            out.append({
                "ok": True,
                "url": href,
                "title": title,
                "content": body,
                "source_id": source_id,
                "content_hash": h,
            })
        log.emit(
            "tool.returned",
            parent=call_id,
            tool="duckduckgo_html_search",
            ok=True,
            n_hits=len(out),
        )
        return out
    except Exception as e:
        log.emit(
            "tool.returned",
            parent=call_id,
            tool="duckduckgo_html_search",
            ok=False,
            error=repr(e)[:200],
        )
        return []
