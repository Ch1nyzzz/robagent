"""url_fetch tool — HTTP GET a URL and return text content.

HTML pages are stripped to readable text via BeautifulSoup. Binary content
types are reported but not decoded. Content is truncated to keep tool output
within a sensible context budget.
"""
from __future__ import annotations

from typing import Any

import requests
from bs4 import BeautifulSoup


_TIMEOUT = 20
_MAX_CHARS = 20000
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "url_fetch",
        "description": (
            "Fetch a URL via HTTP GET and return its readable text content. "
            "HTML pages are converted to plain text. Output is truncated."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch. Must be http(s)://.",
                },
            },
            "required": ["url"],
        },
    },
}


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def run(args: dict) -> str:
    url = (args.get("url") or "").strip()
    if not url:
        return "ERROR: url_fetch requires a non-empty url argument."
    if not (url.startswith("http://") or url.startswith("https://")):
        return f"ERROR: url must start with http:// or https://; got {url!r}."
    try:
        resp = requests.get(
            url,
            timeout=_TIMEOUT,
            headers={"User-Agent": _USER_AGENT},
            allow_redirects=True,
        )
    except Exception as e:
        return f"ERROR fetching {url!r}: {type(e).__name__}: {e}"
    if resp.status_code >= 400:
        return f"ERROR HTTP {resp.status_code} fetching {url!r}: {resp.reason}"
    ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if ctype.startswith("text/html") or ctype in ("application/xhtml+xml",):
        text = _html_to_text(resp.text)
    elif ctype.startswith("text/") or ctype in ("application/json", "application/xml"):
        text = resp.text
    else:
        return (
            f"[non-text content {ctype!r} at {url!r}; "
            f"content_length={resp.headers.get('Content-Length', '?')}; "
            f"first 200 bytes: {resp.content[:200]!r}]"
        )
    if len(text) > _MAX_CHARS:
        text = text[:_MAX_CHARS] + f"\n\n[... truncated; total {len(text)} chars]"
    return text
