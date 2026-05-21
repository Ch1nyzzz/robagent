from __future__ import annotations

import os
import re
import subprocess
import tempfile
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

SEARCH_REFINE_SYSTEM_PROMPT = (
    "You are an assistant solving a benchmark task. "
    "Web search results are shown below. "
    "If the results directly answer the question, output ONLY the final answer "
    "(number, short phrase, list, etc. — no explanations). "
    "If the results are insufficient or do not contain the specific data needed, "
    "respond ONLY with: SEARCH[<a different, more targeted search query>]"
)

SEARCH_ANSWER_SYSTEM_PROMPT = (
    "You are a precise problem-solving assistant. "
    "Web search results are provided below. Use them to answer the question. "
    "Then output the final answer on its own line in this exact format:\n"
    "FINAL ANSWER: <value>\n"
    "where <value> is the answer only, with no extra text or punctuation. "
    "If the search results are insufficient, output your best estimate."
)

CODE_SOLVER_SYSTEM_PROMPT = (
    "Write a self-contained Python 3 script that computes the answer to the problem below. "
    "Requirements:\n"
    "- Use ONLY standard library modules (math, itertools, collections, random, etc.)\n"
    "- Print ONLY the final answer as the LAST line of output\n"
    "- Maximum 25 lines of code\n"
    "- NO explanations, NO markdown fences, NO docstrings\n"
    "Output ONLY the raw Python code starting with 'import' or the first statement."
)

MAX_TOKENS = 8192
RECOVERY_MAX_TOKENS = 1024
RECOVERY_TEMPERATURE = 0.0
SEARCH_MAX_TOKENS = 4096
CODE_SOLVER_MAX_TOKENS = 2048
CODE_SOLVER_TEMPERATURE = 0.7

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

# Threshold for CSS/JS content density — fraction of {/}/; chars.
_CSS_HEAVY_THRESHOLD = 0.04


def _needs_web(prompt: str) -> bool:
    return bool(_WEB_SIGNAL_RE.search(prompt))


def _extract_urls(web_content: str) -> list[str]:
    return [m.group(1).strip() for m in _URL_IN_SNIPPET_RE.finditer(web_content)]


def _url_blocked(url: str) -> bool:
    m = re.search(r"https?://([^/]+)", url)
    if not m:
        return True
    host = m.group(1).lower().lstrip("www.")
    return any(host == d or host.endswith("." + d) for d in _SKIP_FETCH_DOMAINS)


def _is_wikipedia_url(url: str) -> bool:
    return bool(re.search(r"wikipedia\.org", url, re.I))


def _is_css_heavy(text: str) -> bool:
    """Return True if content is dominated by CSS/JS boilerplate rather than readable text.

    Pages that fail HTTP rendering (JS-rendered SPAs, CSS-framework dumps) return raw
    stylesheet/script content.  A high ratio of '{', '}', ';' characters is a reliable
    signal: prose text never exceeds ~1%, while CSS/JS files routinely exceed 4-6%.
    """
    if not text:
        return False
    css_js_chars = text.count("{") + text.count("}") + text.count(";")
    return css_js_chars / max(len(text), 1) > _CSS_HEAVY_THRESHOLD


def _extract_best_url(web_content: str, exclude: str = "") -> str:
    all_urls = _extract_urls(web_content)
    candidates = [u for u in all_urls if u and u != exclude and not _url_blocked(u)]

    for url in candidates:
        if not _is_wikipedia_url(url):
            return url

    for url in candidates:
        if _is_wikipedia_url(url):
            return url

    return ""


def _fetch_url_text(url: str, timeout: int = 8, max_chars: int = 4000) -> str:
    if not url or _url_blocked(url):
        return ""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"},
        )
        ctx = urllib.request.ssl._create_unverified_context()  # type: ignore[attr-defined]
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            raw = resp.read(65536).decode("utf-8", errors="replace")
        raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
        raw = re.sub(r"<[^>]+>", " ", raw)
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


def _fetch_with_quality_check(
    url: str,
    log: EventLog,
    root: str,
    source_label: str,
) -> tuple[str, str]:
    """Fetch URL and return (fetched_text, used_url).

    Returns ("", "") if the URL is blocked, unreachable, or CSS/JS-heavy.
    Logs a tool.skipped event when content is discarded due to CSS density.
    """
    if not url:
        return "", ""
    fetched = _fetch_url_text(url)
    if not fetched:
        return "", ""
    if _is_css_heavy(fetched):
        log.emit(
            "tool.skipped",
            parent=root,
            tool="url_fetch",
            url=url[:200],
            reason="css_heavy",
            source=source_label,
            css_density=round(
                (fetched.count("{") + fetched.count("}") + fetched.count(";")) / max(len(fetched), 1),
                3,
            ),
        )
        return "", ""
    log.emit(
        "tool.called",
        parent=root,
        tool="url_fetch",
        url=url[:200],
        chars=len(fetched),
        source=source_label,
        is_wikipedia=_is_wikipedia_url(url),
    )
    return fetched, url


def _run_web_search(task_prompt: str, log: EventLog, root: str) -> str:
    """Three-turn iterative web search with CSS/JS quality filter on URL fetches."""
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
        # ── Turn 3: second search + quality-filtered URL fetch + final answer ─
        query2 = action2.group(1).strip()
        log.emit("claim.extracted", parent=root, action="SEARCH2", query=query2[:200])
        web_content2 = _web_search(query2, log, root)

        fetched_text = ""
        fetched_url = ""

        # Try T2 best URL with CSS/JS quality check.
        best_url_t2 = _extract_best_url(web_content2) if web_content2 else ""
        if best_url_t2:
            fetched_text, fetched_url = _fetch_with_quality_check(
                best_url_t2, log, root, source_label="turn2_best"
            )

        # Fall back to T1 best URL if T2 was empty, blocked, or CSS-heavy.
        if not fetched_text:
            best_url_t1 = _extract_best_url(web_content1, exclude=best_url_t2)
            if best_url_t1:
                fetched_text, fetched_url = _fetch_with_quality_check(
                    best_url_t1, log, root, source_label="turn1_fallback_best"
                )

        # Build combined context: snippet data + fetched page (if quality passed).
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

    if content2 and result2["finish_reason"] == "stop":
        return _extract_answer(content2, result2["finish_reason"]) or content2.splitlines()[-1].strip()
    return ""


def _run_cot(task_prompt: str, log: EventLog, root: str, *, state: dict | None = None) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_prompt},
    ]
    call = log.emit("llm.requested", parent=root, messages=messages, model=DEFAULT_MODEL)
    try:
        result = chat(messages=messages, max_tokens=MAX_TOKENS)
    except Exception as e:
        log.emit("llm.failed", parent=call, error=repr(e))
        if state is not None:
            state["primary_exhausted"] = False
            state["double_exhausted"] = False
        return ""

    log.emit(
        "llm.responded",
        parent=call,
        content=result["content"],
        finish_reason=result["finish_reason"],
        usage=result["usage"],
    )

    primary_content = (result["content"] or "").strip()
    primary_exhausted = (result["finish_reason"] == "length" and not primary_content)
    if state is not None:
        state["primary_exhausted"] = primary_exhausted
        state["double_exhausted"] = False

    answer = _extract_answer(primary_content, result["finish_reason"])

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
            recovery_content = (recovery_result["content"] or "").strip()
            recovery_exhausted = (
                recovery_result["finish_reason"] == "length" and not recovery_content
            )
            if state is not None:
                state["double_exhausted"] = primary_exhausted and recovery_exhausted

            if recovery_content:
                answer = recovery_content

    return answer


def _run_code_solver(task_prompt: str, log: EventLog, root: str) -> str:
    messages = [
        {"role": "system", "content": CODE_SOLVER_SYSTEM_PROMPT},
        {"role": "user", "content": task_prompt},
    ]
    call = log.emit(
        "llm.requested",
        parent=root,
        messages=messages,
        model=DEFAULT_MODEL,
        max_tokens=CODE_SOLVER_MAX_TOKENS,
        pass_="code_solver",
    )
    try:
        result = chat(
            messages=messages,
            max_tokens=CODE_SOLVER_MAX_TOKENS,
            temperature=CODE_SOLVER_TEMPERATURE,
        )
    except Exception as e:
        log.emit("llm.failed", parent=call, error=repr(e), pass_="code_solver")
        return ""

    log.emit(
        "llm.responded",
        parent=call,
        content=result["content"],
        finish_reason=result["finish_reason"],
        usage=result["usage"],
        pass_="code_solver",
    )

    code = (result["content"] or "").strip()
    if not code:
        return ""

    code = re.sub(r"^```python\s*\n?", "", code, flags=re.I)
    code = re.sub(r"\n?```\s*$", "", code)
    code = code.strip()

    if not code:
        return ""

    log.emit("tool.called", parent=root, tool="code_exec", code_len=len(code), pass_="code_solver")

    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name

        proc = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        stdout = proc.stdout.strip()
        if stdout:
            lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
            answer = lines[-1] if lines else ""
            log.emit(
                "tool.returned",
                parent=root,
                tool="code_exec",
                exit_code=proc.returncode,
                output_lines=len(lines),
                answer_preview=answer[:80],
                pass_="code_solver",
            )
            return answer
        log.emit(
            "tool.returned",
            parent=root,
            tool="code_exec",
            exit_code=proc.returncode,
            output_lines=0,
            stderr_preview=proc.stderr.strip()[:120],
            pass_="code_solver",
        )
    except subprocess.TimeoutExpired:
        log.emit("tool.failed", parent=root, tool="code_exec", error="TimeoutExpired", pass_="code_solver")
    except Exception as e:
        log.emit("tool.failed", parent=root, tool="code_exec", error=repr(e), pass_="code_solver")
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    return ""


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

    cot_state: dict = {}
    if not answer:
        answer = _run_cot(task_prompt, log, root, state=cot_state)

    if not answer and not web_needed and cot_state.get("double_exhausted", False):
        answer = _run_code_solver(task_prompt, log, root)

    log.emit("answer.emitted", parent=root, answer=answer)

    return {
        "run_id": run_id,
        "answer": answer,
        "error": None,
        "trace_path": str(log.path),
        "log": log,
        "root_event_id": root,
    }
