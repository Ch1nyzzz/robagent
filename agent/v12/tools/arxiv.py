"""arXiv search — no API key required.

Hits the public arXiv API endpoint and parses the Atom-XML response with simple
regex (no XML parser dependency). Emits `source.opened{kind=arxiv}` so the
(claim, source) protocol consumes the abstract as a verifiable source body.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any
from urllib.parse import quote_plus

import httpx

from agent.events import EventLog


_HEADERS = {"User-Agent": "robagent-harness/1.0 (research)"}
_TIMEOUT = httpx.Timeout(20.0, connect=10.0)
_MAX_BODY_CHARS = 4000

_ENTRY_RE = re.compile(r"<entry>(.*?)</entry>", re.DOTALL)
_TITLE_RE = re.compile(r"<title>(.*?)</title>", re.DOTALL)
_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL)
_URL_RE = re.compile(r'<id>(.*?)</id>', re.DOTALL)


def _quote_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _strip(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def arxiv_search(
    query: str,
    *,
    log: EventLog,
    parent: str,
    limit: int = 3,
) -> list[dict[str, Any]]:
    call_id = log.emit(
        "tool.called",
        parent=parent,
        tool="arxiv_search",
        args={"query": query, "limit": limit},
    )
    url = (
        "http://export.arxiv.org/api/query?search_query="
        + quote_plus(query)
        + f"&start=0&max_results={limit}"
    )
    try:
        with httpx.Client(headers=_HEADERS, timeout=_TIMEOUT) as cli:
            r = cli.get(url)
            r.raise_for_status()
        text = r.text or ""
        entries = _ENTRY_RE.findall(text)
        out: list[dict[str, Any]] = []
        for ent in entries[:limit]:
            title_m = _TITLE_RE.search(ent)
            summary_m = _SUMMARY_RE.search(ent)
            url_m = _URL_RE.search(ent)
            if not (title_m and summary_m and url_m):
                continue
            title = _strip(title_m.group(1))
            summary = _strip(summary_m.group(1))
            href = _strip(url_m.group(1))
            body = (title + ". " + summary)[:_MAX_BODY_CHARS]
            if not body.strip():
                continue
            h = _quote_hash(body + href)
            source_id = f"S_arxiv_{h}"
            log.emit(
                "source.opened",
                parent=call_id,
                source_id=source_id,
                kind="arxiv",
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
            tool="arxiv_search",
            ok=True,
            n_hits=len(out),
        )
        return out
    except Exception as e:
        log.emit(
            "tool.returned",
            parent=call_id,
            tool="arxiv_search",
            ok=False,
            error=repr(e)[:200],
        )
        return []
