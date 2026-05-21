"""Generic web fetch tool — httpx + minimal HTML stripping."""
from __future__ import annotations

import hashlib
import re
from typing import Any

import httpx

from agent.events import EventLog


_HEADERS = {"User-Agent": "robagent-harness/1.0 (research)"}
_TIMEOUT = httpx.Timeout(20.0, connect=10.0)
_MAX_BODY_CHARS = 12000


def _strip_html(html: str) -> str:
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


def web_fetch(url: str, *, log: EventLog, parent: str) -> dict[str, Any]:
    """Fetch a URL, strip to text, emit source.opened on success."""
    call_id = log.emit(
        "tool.called",
        parent=parent,
        tool="web_fetch",
        args={"url": url},
    )
    try:
        with httpx.Client(
            headers=_HEADERS, timeout=_TIMEOUT, follow_redirects=True
        ) as cli:
            r = cli.get(url)
            r.raise_for_status()
            body = _strip_html(r.text or "")[:_MAX_BODY_CHARS]
        if not body:
            log.emit(
                "tool.returned",
                parent=call_id,
                tool="web_fetch",
                ok=False,
                error="empty body",
            )
            return {"ok": False, "url": url, "content": ""}
        h = hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]
        source_id = f"S_web_{h}"
        log.emit(
            "source.opened",
            parent=call_id,
            source_id=source_id,
            kind="web",
            url=url,
            content_chars=len(body),
            content_hash=h,
        )
        log.emit(
            "tool.returned",
            parent=call_id,
            tool="web_fetch",
            ok=True,
            source_id=source_id,
            n_chars=len(body),
        )
        return {
            "ok": True,
            "url": url,
            "content": body,
            "source_id": source_id,
            "content_hash": h,
        }
    except Exception as e:
        log.emit(
            "tool.returned",
            parent=call_id,
            tool="web_fetch",
            ok=False,
            error=repr(e)[:200],
        )
        return {"ok": False, "url": url, "content": ""}
