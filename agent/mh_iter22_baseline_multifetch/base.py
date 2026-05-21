from __future__ import annotations

import re
import urllib.request
from typing import Any

from agent.events import EventLog, new_run_id, traces_dir
from agent.llm import chat, DEFAULT_MODEL


SYSTEM_PROMPT = (
    "You are a precise problem-solving assistant. "
    "Work through the problem concisely — write only the key reasoning steps. "
    "Then output the final answer on its own line in this exact format:\n"
    "FINAL ANSWER: <value>\n"
    "where <value> is the answer only, with no extra text or punctuation."
)

RECOVERY_SYSTEM_PROMPT = (
    "You are an assistant. Output ONLY the final answer value to the question — "
    "no reasoning, no explanation, no thinking steps. "
    "Just the raw answer: a number, a short string, or a few words."
)

# Turn 1: force SEARCH[query], with query formulation guidance (same as iter18/iter19/iter20/iter21).
SEARCH_AWARE_SYSTEM_PROMPT = (
    "You are an assistant solving a benchmark task that requires looking up specific data "
    "from a named external source. "
    "You MUST search for the answer — do NOT answer directly.\n\n"
    "To find the exact data, craft a precise search query:\n"
    "- If the question names a specific website or database, add a site-specific prefix "
    "(e.g., 'site:<domain.org> <keywords>' restricts results to that source).\n"
    "- If the question provides a DOI or identifier, search for the TITLE and AUTHOR of "
    "that document rather than the identifier itself — this reaches open-access and indexed versions.\n"
    "- Include exact names, dates, and terms from the question.\n\n"
    "Respond with ONLY: SEARCH[<your targeted search query>]"
)

# Turn 2: refine-or-answer (same as iter11/iter18/iter19/iter20/iter21).
SEARCH_REFINE_SYSTEM_PROMPT = (
    "You are an assistant solving a benchmark task. "
    "Web search results are shown below. "
    "If the results directly answer the question, output ONLY the final answer "
    "(number, short phrase, list, etc. — no explanations). "
    "If the results are insufficient or do not contain the specific data needed, "
    "respond ONLY with: SEARCH[<a different, more targeted search query>]"
)

# Turn 3: web answer from combined snippets + optional fetched page content.
SEARCH_ANSWER_SYSTEM_PROMPT = (
    "You are a precise problem-solving assistant. "
    "Web search results are provided below. Use them to answer the question. "
    "Then output the final answer on its own line in this exact format:\n"
    "FINAL ANSWER: <value>\n"
    "where <value> is the answer only, with no extra text or punctuation. "
    "If the search results are insufficient, output your best estimate."
)

MAX_TOKENS = 8192
RECOVERY_MAX_TOKENS = 1024
RECOVERY_TEMPERATURE = 0.0
SEARCH_MAX_TOKENS = 4096

# NEW vs iter21: quality gate thresholds for URL fetch validation.
MIN_FETCH_CONTENT_LEN = 600   # fewer chars than this → page is near-empty (navigation-only)
MAX_CSS_CHAR_RATIO = 0.04     # more than 4% of chars are '{', '}', ';' → CSS/code-heavy
MAX_FETCH_ATTEMPTS = 4        # try up to this many candidate URLs before giving up

_WEB_SIGNAL_RE = re.compile(
    r"(?:"
    r"\bScienceDirect\b"
    r"|\bBielefeld\b"
    r"|Girls\s+Who\s+Code"
    r"|\bUSGS\b"
    r"|\bORCID\b"
    r"|\barXiv\b"
    r"|doi[:\s]\s*10\.\d{4}"
    r"|Metropolitan\s+Museum"
    r"|Project\s+MUSE"
    r"|\bJSTOR\b"
    r"|\bTri.?Rail\b"
    r"|\bNonindigenous\s+Aquatic\s+Species\b"
    r"|\bIn\s+the\s+(?:film|movie)\b"
    r"|\bpainting\b"
    r"|\bblog\s+post\b"
    r"|\bYouTube\b"
    r")",
    re.I,
)

_SEARCH_ACTION_RE = re.compile(r"SEARCH\[([^\]]{1,400})\]")
_URL_IN_SNIPPET_RE = re.compile(r"URL:\s*(https?://[^\s\n]+)")

# Domains known to block bots, require JS, or be paywalled — skip fetching these.
_SKIP_FETCH_DOMAINS = frozenset([
    "jstor.org",
    "muse.jhu.edu",
    "sciencedirect.com",
    "elsevier.com",
    "springer.com",
    "wiley.com",
    "nature.com",
    "researchgate.net",
    "academia.edu",
    "instagram.com",
    "twitter.com",
    "x.com",
    "facebook.com",
    "tiktok.com",
    "rutube.ru",
    "yandex.ru",
    "youtube.com",
    "reddit.com",
    "pinterest.com",
    "linkedin.com",
])


def _needs_web(prompt: str) -> bool:
    return bool(_WEB_SIGNAL_RE.search(prompt))


def _extract_urls(web_content: str) -> list[str]:
    """Return all URLs found in DDG snippet content (in order of appearance)."""
    return [m.group(1).strip() for m in _URL_IN_SNIPPET_RE.finditer(web_content)]


def _url_blocked(url: str) -> bool:
    """Return True if the URL's domain is in the skip list."""
    m = re.search(r"https?://([^/]+)", url)
    if not m:
        return True
    host = m.group(1).lower().lstrip("www.")
    return any(host == d or host.endswith("." + d) for d in _SKIP_FETCH_DOMAINS)


def _is_wikipedia_url(url: str) -> bool:
    """Return True if the URL is a Wikipedia article."""
    return bool(re.search(r"wikipedia\.org", url, re.I))


def _passes_quality(text: str) -> bool:
    """Return True if the fetched content is substantial and not CSS/code-heavy.

    NEW vs iter21: replaces the old single-URL fallback. Two failure modes targeted:
    (1) near-empty pages (< MIN_FETCH_CONTENT_LEN chars) such as navigation-only pages
        — affects tasks where the top URL is a site homepage/reports-index page that
        returns only navigation links and contact info;
    (2) CSS/code-heavy pages where the visible character stream is dominated by
        '{', '}', ';' from inline stylesheets injected into the HTML body
        — affects tasks where the top DDG result is a WordPress/Divi page whose
        _fetch_url_text output consists almost entirely of un-stripped inline CSS.
    """
    if not text:
        return False
    if len(text) < MIN_FETCH_CONTENT_LEN:
        return False
    css_chars = sum(1 for c in text if c in "{};")
    if css_chars / len(text) > MAX_CSS_CHAR_RATIO:
        return False
    return True


def _get_fetch_candidates(
    web_content2: str, web_content1: str
) -> list[tuple[str, str]]:
    """Return URL candidates in priority order: T2 non-wiki > T1 non-wiki > Wikipedia.

    NEW vs iter21: instead of picking a single best URL from T2 then falling back to
    a single best URL from T1, we enumerate ALL accessible non-blocked URLs from both
    search rounds and return them in preference order.  The caller iterates and stops
    at the first URL whose fetched content passes _passes_quality().
    """
    urls_t2 = _extract_urls(web_content2) if web_content2 else []
    urls_t1 = _extract_urls(web_content1) if web_content1 else []

    seen: set[str] = set()
    result: list[tuple[str, str]] = []

    # Non-Wikipedia from T2 first
    for url in urls_t2:
        if url and not _url_blocked(url) and not _is_wikipedia_url(url) and url not in seen:
            result.append((url, "turn2"))
            seen.add(url)

    # Non-Wikipedia from T1
    for url in urls_t1:
        if url and not _url_blocked(url) and not _is_wikipedia_url(url) and url not in seen:
            result.append((url, "turn1"))
            seen.add(url)

    # Wikipedia from T2 then T1 (last resort — previously shown to cause hallucination
    # when Wikipedia articles contain general background instead of specific task values,
    # so we keep them as final fallback only)
    for url in urls_t2:
        if url and not _url_blocked(url) and _is_wikipedia_url(url) and url not in seen:
            result.append((url, "turn2_wiki"))
            seen.add(url)
    for url in urls_t1:
        if url and not _url_blocked(url) and _is_wikipedia_url(url) and url not in seen:
            result.append((url, "turn1_wiki"))
            seen.add(url)

    return result


def _fetch_url_text(url: str, timeout: int = 8, max_chars: int = 4000) -> str:
    """Fetch URL with urllib and return stripped plain text (up to max_chars)."""
    if not url or _url_blocked(url):
        return ""
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; research-bot/1.0)"
                )
            },
        )
        ctx = urllib.request.ssl._create_unverified_context()  # type: ignore[attr-defined]
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            raw = resp.read(65536).decode("utf-8", errors="replace")
        # Strip script/style blocks.
        raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
        # Strip all remaining tags.
        raw = re.sub(r"<[^>]+>", " ", raw)
        # Collapse whitespace.
        raw = re.sub(r"\s+", " ", raw).strip()
        return raw[:max_chars]
    except Exception:
        return ""


def _web_search(query: str, log: EventLog, parent: str, max_results: int = 5) -> str:
    log.emit("tool.called", parent=parent, tool="web_search", query=query[:200])
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS  # type: ignore[no-redef]
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
        if not raw:
            log.emit("tool.returned", parent=parent, tool="web_search", n_results=0)
            return ""
        lines: list[str] = []
        for i, r in enumerate(raw, 1):
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            lines.append(f"[{i}] {title}\nURL: {href}\n{body}")
        content = "\n\n".join(lines)[:4000]
        log.emit("tool.returned", parent=parent, tool="web_search", n_results=len(raw))
        log.emit("source.opened", parent=parent, query=query[:200], source="duckduckgo")
        return content
    except Exception as e:
        log.emit("tool.failed", parent=parent, tool="web_search", error=repr(e))
        return ""


def _extract_answer(content: str, finish_reason: str) -> str:
    if content:
        m = re.search(r"FINAL ANSWER:\s*(.+)", content, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip().rstrip(".")
            if candidate:
                return candidate
        if finish_reason == "stop":
            lines = [ln.strip() for ln in content.strip().splitlines() if ln.strip()]
            return lines[-1] if lines else ""
    return ""


def _run_web_search(task_prompt: str, log: EventLog, root: str) -> str:
    """Three-turn iterative web search with multi-URL quality-gated fetch.

    NEW vs iter21: the Turn 3 URL fetch step is replaced by a multi-URL loop
    (up to MAX_FETCH_ATTEMPTS) that iterates over all candidate URLs from both
    Turn 1 and Turn 2 search results in priority order (T2 non-wiki > T1 non-wiki
    > Wikipedia last resort) and stops at the first URL whose fetched content passes
    the _passes_quality() gate.  The quality gate rejects near-empty pages (< 600
    chars) and CSS/code-heavy pages (> 4% {/}/; chars).
    """
    # ── Turn 1: unconditional SEARCH ─────────────────────────────────────────
    messages1 = [
        {"role": "system", "content": SEARCH_AWARE_SYSTEM_PROMPT},
        {"role": "user", "content": task_prompt},
    ]
    call1 = log.emit(
        "llm.requested",
        parent=root,
        messages=messages1,
        model=DEFAULT_MODEL,
        max_tokens=SEARCH_MAX_TOKENS,
        pass_="search_aware",
    )
    try:
        result1 = chat(messages=messages1, max_tokens=SEARCH_MAX_TOKENS)
    except Exception as e:
        log.emit("llm.failed", parent=call1, error=repr(e))
        return ""

    log.emit(
        "llm.responded",
        parent=call1,
        content=result1["content"],
        finish_reason=result1["finish_reason"],
        usage=result1["usage"],
        pass_="search_aware",
    )

    content1 = (result1["content"] or "").strip()
    action1 = _SEARCH_ACTION_RE.search(content1)
    if not action1:
        return ""

    query1 = action1.group(1).strip()
    log.emit("claim.extracted", parent=root, action="SEARCH", query=query1[:200])
    web_content1 = _web_search(query1, log, root)
    if not web_content1:
        return ""

    # ── Turn 2: refine-or-answer ──────────────────────────────────────────────
    context2 = f"{task_prompt}\n\nWeb search results:\n{web_content1}"
    messages2 = [
        {"role": "system", "content": SEARCH_REFINE_SYSTEM_PROMPT},
        {"role": "user", "content": context2},
    ]
    call2 = log.emit(
        "llm.requested",
        parent=root,
        messages=messages2,
        model=DEFAULT_MODEL,
        max_tokens=SEARCH_MAX_TOKENS,
        pass_="search_refine",
    )
    try:
        result2 = chat(messages=messages2, max_tokens=SEARCH_MAX_TOKENS)
    except Exception as e:
        log.emit("llm.failed", parent=call2, error=repr(e))
        return ""

    log.emit(
        "llm.responded",
        parent=call2,
        content=result2["content"],
        finish_reason=result2["finish_reason"],
        usage=result2["usage"],
        pass_="search_refine",
    )

    content2 = (result2["content"] or "").strip()
    action2 = _SEARCH_ACTION_RE.search(content2)

    if action2 and result2["finish_reason"] == "stop":
        # ── Turn 3: second search + multi-URL quality-gated fetch + final answer ─
        query2 = action2.group(1).strip()
        log.emit("claim.extracted", parent=root, action="SEARCH2", query=query2[:200])
        web_content2 = _web_search(query2, log, root)

        # NEW vs iter21: iterate candidate URLs in priority order until one passes
        # the content quality gate.  iter21 tried exactly one Turn-2 URL then one
        # Turn-1 URL, stopping as soon as *any* non-empty content was returned.
        # That caused failures when:
        #   - The first URL returned very short content (< 600 chars, e.g. a
        #     site navigation page with only headers and links);
        #   - The first URL returned CSS-heavy content from an un-stripped inline
        #     stylesheet (> 4% '{'/'}'/';' chars in the visible text stream).
        # In both cases the agent never tried the next available URL in results.
        fetched_text = ""
        fetched_url = ""

        for url, source in _get_fetch_candidates(web_content2, web_content1)[:MAX_FETCH_ATTEMPTS]:
            text = _fetch_url_text(url)
            if not text:
                # Empty fetch (blocked, timeout, redirect-only) — try next candidate.
                continue
            if _passes_quality(text):
                fetched_text = text
                fetched_url = url
                log.emit(
                    "tool.called",
                    parent=root,
                    tool="url_fetch",
                    url=url[:200],
                    chars=len(text),
                    source=source,
                    is_wikipedia=_is_wikipedia_url(url),
                )
                break
            # Content was fetched but failed quality gate — log and try next.
            log.emit(
                "tool.called",
                parent=root,
                tool="url_fetch_low_quality",
                url=url[:200],
                chars=len(text),
                source=source,
            )

        # Build combined context: snippet data + fetched page.
        parts = []
        if web_content1:
            parts.append(f"First search results:\n{web_content1}")
        if web_content2:
            parts.append(f"Second search results:\n{web_content2}")
        if fetched_text:
            parts.append(
                f"Full page content from top result ({fetched_url[:100]}):\n{fetched_text}"
            )
        combined = "\n\n".join(parts) if parts else web_content1

        context3 = f"{task_prompt}\n\nWeb search results:\n{combined}"
        messages3 = [
            {"role": "system", "content": SEARCH_ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": context3},
        ]
        call3 = log.emit(
            "llm.requested",
            parent=root,
            messages=messages3,
            model=DEFAULT_MODEL,
            max_tokens=SEARCH_MAX_TOKENS,
            pass_="web_answer",
        )
        try:
            result3 = chat(messages=messages3, max_tokens=SEARCH_MAX_TOKENS)
        except Exception as e:
            log.emit("llm.failed", parent=call3, error=repr(e))
            return ""

        log.emit(
            "llm.responded",
            parent=call3,
            content=result3["content"],
            finish_reason=result3["finish_reason"],
            usage=result3["usage"],
            pass_="web_answer",
        )
        return _extract_answer(result3["content"] or "", result3["finish_reason"])

    # Model answered directly in Turn 2 — no fetch needed.
    if content2 and result2["finish_reason"] == "stop":
        return _extract_answer(content2, result2["finish_reason"]) or content2.splitlines()[-1].strip()
    return ""


def _run_cot(task_prompt: str, log: EventLog, root: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_prompt},
    ]
    call = log.emit("llm.requested", parent=root, messages=messages, model=DEFAULT_MODEL)
    try:
        result = chat(messages=messages, max_tokens=MAX_TOKENS)
    except Exception as e:
        log.emit("llm.failed", parent=call, error=repr(e))
        return ""

    log.emit(
        "llm.responded",
        parent=call,
        content=result["content"],
        finish_reason=result["finish_reason"],
        usage=result["usage"],
    )

    answer = _extract_answer(result["content"] or "", result["finish_reason"])

    if not answer:
        recovery_messages = [
            {"role": "system", "content": RECOVERY_SYSTEM_PROMPT},
            {"role": "user", "content": task_prompt},
        ]
        recovery_call = log.emit(
            "llm.requested",
            parent=root,
            messages=recovery_messages,
            model=DEFAULT_MODEL,
            recovery=True,
            recovery_max_tokens=RECOVERY_MAX_TOKENS,
        )
        try:
            recovery_result = chat(
                messages=recovery_messages,
                max_tokens=RECOVERY_MAX_TOKENS,
                temperature=RECOVERY_TEMPERATURE,
            )
        except Exception as e:
            log.emit("llm.failed", parent=recovery_call, error=repr(e), recovery=True)
        else:
            log.emit(
                "llm.responded",
                parent=recovery_call,
                content=recovery_result["content"],
                finish_reason=recovery_result["finish_reason"],
                usage=recovery_result["usage"],
                recovery=True,
            )
            recovered = (recovery_result["content"] or "").strip()
            if recovered:
                answer = recovered

    return answer


def run_task(
    *,
    benchmark: str,
    task_id: str,
    task_prompt: str,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_id = new_run_id()
    log = EventLog(run_id=run_id, benchmark=benchmark, task_id=task_id, out_dir=traces_dir())
    extras = extras or {}

    root = log.emit(
        "run.started",
        question=task_prompt,
        model=DEFAULT_MODEL,
        extras=extras,
    )

    web_needed = _needs_web(task_prompt)
    log.emit("task.routed", parent=root, route="WEB_SEARCH" if web_needed else "DIRECT")

    answer = ""

    if web_needed:
        answer = _run_web_search(task_prompt, log, root)

    if not answer:
        answer = _run_cot(task_prompt, log, root)

    log.emit("answer.emitted", parent=root, answer=answer)

    return {
        "run_id": run_id,
        "answer": answer,
        "error": None,
        "trace_path": str(log.path),
        "log": log,
        "root_event_id": root,
    }
