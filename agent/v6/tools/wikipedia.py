"""Wikipedia REST API tool — public, no key needed.

Emits structured tool.called / tool.returned / source.opened events.

Output is intentionally bounded (truncate page bodies) so downstream LLM nodes
operate on a constrained surface.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

import httpx

from agent.events import EventLog


_HEADERS = {"User-Agent": "robagent-harness/1.0 (research)"}
_TIMEOUT = httpx.Timeout(20.0, connect=10.0)
_MAX_BODY_CHARS = 12000


def _quote_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _strip_html(html: str) -> str:
    # very small HTML-to-text, sufficient for Wikipedia extract bodies
    text = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def wikipedia_search(
    query: str,
    *,
    log: EventLog,
    parent: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Search Wikipedia titles by free-text query. Returns top-N candidate titles."""
    call_id = log.emit(
        "tool.called",
        parent=parent,
        tool="wikipedia_search",
        args={"query": query, "limit": limit},
    )
    try:
        with httpx.Client(headers=_HEADERS, timeout=_TIMEOUT) as cli:
            r = cli.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": limit,
                    "format": "json",
                },
            )
            r.raise_for_status()
            data = r.json()
        hits = []
        for h in (data.get("query") or {}).get("search", [])[:limit]:
            hits.append({
                "title": h.get("title"),
                "snippet": _strip_html(h.get("snippet") or ""),
                "pageid": h.get("pageid"),
            })
        log.emit(
            "tool.returned",
            parent=call_id,
            tool="wikipedia_search",
            n_hits=len(hits),
            ok=True,
        )
        return hits
    except Exception as e:
        log.emit(
            "tool.returned",
            parent=call_id,
            tool="wikipedia_search",
            ok=False,
            error=repr(e)[:200],
        )
        return []


def wikipedia_fetch(
    title: str,
    *,
    log: EventLog,
    parent: str,
    section: str | None = None,
) -> dict[str, Any]:
    """Fetch the plain-text extract for a Wikipedia page title.

    Returns {source_id, url, title, content, ok}. Emits source.opened on success.
    """
    call_id = log.emit(
        "tool.called",
        parent=parent,
        tool="wikipedia_fetch",
        args={"title": title, "section": section},
    )
    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    try:
        with httpx.Client(headers=_HEADERS, timeout=_TIMEOUT) as cli:
            r = cli.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "prop": "extracts",
                    "explaintext": 1,
                    "titles": title,
                    "format": "json",
                    "redirects": 1,
                },
            )
            r.raise_for_status()
            data = r.json()
        pages = (data.get("query") or {}).get("pages") or {}
        body = ""
        resolved_title = title
        for _, page in pages.items():
            if page.get("missing"):
                continue
            body = page.get("extract") or ""
            resolved_title = page.get("title") or title
            break
        if not body:
            log.emit(
                "tool.returned",
                parent=call_id,
                tool="wikipedia_fetch",
                ok=False,
                error="page missing or empty",
            )
            return {"ok": False, "url": url, "title": title, "content": ""}
        body = body[:_MAX_BODY_CHARS]
        h = _quote_hash(body)
        source_id = f"S_wiki_{h}"
        log.emit(
            "source.opened",
            parent=call_id,
            source_id=source_id,
            kind="wikipedia",
            url=url,
            title=resolved_title,
            content_chars=len(body),
            content_hash=h,
        )
        log.emit(
            "tool.returned",
            parent=call_id,
            tool="wikipedia_fetch",
            ok=True,
            source_id=source_id,
            n_chars=len(body),
        )
        return {
            "ok": True,
            "url": url,
            "title": resolved_title,
            "content": body,
            "source_id": source_id,
            "content_hash": h,
        }
    except Exception as e:
        log.emit(
            "tool.returned",
            parent=call_id,
            tool="wikipedia_fetch",
            ok=False,
            error=repr(e)[:200],
        )
        return {"ok": False, "url": url, "title": title, "content": ""}
