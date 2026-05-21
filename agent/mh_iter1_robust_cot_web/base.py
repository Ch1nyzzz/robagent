from __future__ import annotations

import re
from typing import Any

from agent.events import EventLog, new_run_id, traces_dir
from agent.llm import chat, DEFAULT_MODEL


# ── Prompts ───────────────────────────────────────────────────────────────────

# Primary COT prompt: externalises reasoning so DeepSeek doesn't exhaust all
# tokens on hidden chain-of-thought before emitting any visible output.
SYSTEM_PROMPT = (
    "You are a precise problem-solving assistant. "
    "Work through the problem concisely — write only the key reasoning steps. "
    "Then output the final answer on its own line in this exact format:\n"
    "FINAL ANSWER: <value>\n"
    "where <value> is the answer only, with no extra text or punctuation."
)

# Recovery: no reasoning budget wasted; model must commit immediately.
RECOVERY_SYSTEM_PROMPT = (
    "You are an assistant. Output ONLY the final answer value to the question — "
    "no reasoning, no explanation, no thinking steps. "
    "Just the raw answer: a number, a short string, or a few words."
)

# Web Turn 1: model decides whether to issue a search or answer directly.
SEARCH_AWARE_SYSTEM_PROMPT = (
    "You are an assistant solving a benchmark task that requires looking up "
    "specific data from a named external source. "
    "You MUST search for the answer — do NOT answer directly.\n\n"
    "To find the exact data, craft a precise search query:\n"
    "- If the question names a specific website or database, add a site-specific "
    "prefix (e.g., 'site:<domain.org> <keywords>' restricts results to that source).\n"
    "- If the question provides a DOI or identifier, search for the TITLE and AUTHOR "
    "of that document rather than the identifier itself — this reaches open-access "
    "and indexed versions.\n"
    "- Include exact names, dates, and terms from the question.\n\n"
    "Respond with ONLY: SEARCH[<your targeted search query>]"
)

# Web Turn 2: refine query or answer if results are sufficient.
SEARCH_REFINE_SYSTEM_PROMPT = (
    "You are an assistant solving a benchmark task. "
    "Web search results are shown below. "
    "If the results directly answer the question, output ONLY the final answer "
    "(number, short phrase, list, etc. — no explanations). "
    "If the results are insufficient or do not contain the specific data needed, "
    "respond ONLY with: SEARCH[<a different, more targeted search query>]"
)

# Web Turn 3: final answer from combined search results.
SEARCH_ANSWER_SYSTEM_PROMPT = (
    "You are a precise problem-solving assistant. "
    "Web search results are provided below. Use them to answer the question. "
    "Then output the final answer on its own line in this exact format:\n"
    "FINAL ANSWER: <value>\n"
    "where <value> is the answer only, with no extra text or punctuation. "
    "If the search results are insufficient, output your best estimate."
)

# ── Token budgets ─────────────────────────────────────────────────────────────

# 8192 gives DeepSeek-V4-Pro room for internal chain-of-thought plus visible
# answer; v0's 2048 caused ~46% of tasks to truncate with empty content.
MAX_TOKENS = 8192
# Recovery pass: no extended thinking expected, so 4096 is sufficient.
RECOVERY_MAX_TOKENS = 4096
# Per web-turn budget.
SEARCH_MAX_TOKENS = 4096

# ── Deterministic web-signal gate ─────────────────────────────────────────────
# General content signals that indicate live external data is required.
# None of these are task-specific entity literals — they are category-level
# indicators: academic identifiers, temporal specificity markers, content types.
_WEB_SIGNAL_RE = re.compile(
    r"(?:"
    r"doi[:\s/]\s*10\.\d{4}"              # Academic DOI
    r"|https?://"                          # Explicit URL in prompt
    r"|\baccording\s+to\b"               # Citing an external source
    r"|\bas\s+of\s+(?:20\d\d|\d{4})\b"  # Time-specific data ("as of 2020")
    r"|\bfrom\s+the\s+year\s+\d{4}\b"   # Year-range queries
    r"|\bthrough\s+20\d\d\b"             # Year-range end ("through 2020")
    r"|\bget\s+the\s+data\b"             # Explicit data-retrieval instruction
    r"|\barxiv\b"                         # Academic preprint server
    r"|\byoutube\b"                       # Video platform
    r"|\bpainting\b"                      # Artworks often need image/museum lookup
    r"|\bexhibition\b"                    # Gallery/museum exhibitions
    r"|\bcitation\b"                      # Citation fact-checking
    r")",
    re.I,
)

_SEARCH_ACTION_RE = re.compile(r"SEARCH\[([^\]]{1,400})\]")


def _needs_web(prompt: str) -> bool:
    return bool(_WEB_SIGNAL_RE.search(prompt))


# ── Tools ─────────────────────────────────────────────────────────────────────

def _web_search(query: str, log: EventLog, parent: str, max_results: int = 5) -> str:
    """Run DuckDuckGo text search; return formatted snippets or empty on failure."""
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


# ── Deterministic answer extraction ──────────────────────────────────────────

def _extract_answer(content: str, finish_reason: str) -> str:
    """Extract answer via FINAL ANSWER: marker; fall back to last non-empty line."""
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


# ── Pipeline stages ───────────────────────────────────────────────────────────

def _run_web_search(task_prompt: str, log: EventLog, root: str) -> str:
    """Three-turn iterative web search: ask → maybe-refine → answer."""
    # Turn 1: model emits SEARCH[query] or answers directly.
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

    # Model answered directly without searching.
    if result1["finish_reason"] == "stop" and content1 and not _SEARCH_ACTION_RE.search(content1):
        answer = _extract_answer(content1, result1["finish_reason"])
        return answer or content1.splitlines()[-1].strip()

    action1 = _SEARCH_ACTION_RE.search(content1)
    if not action1:
        return ""

    query1 = action1.group(1).strip()
    log.emit("claim.extracted", parent=root, action="SEARCH", query=query1[:200])
    web1 = _web_search(query1, log, root)
    if not web1:
        return ""

    # Turn 2: refine or answer with first-round snippets.
    context2 = f"{task_prompt}\n\nWeb search results:\n{web1}"
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
        # Turn 3: second search + final answer.
        query2 = action2.group(1).strip()
        log.emit("claim.extracted", parent=root, action="SEARCH2", query=query2[:200])
        web2 = _web_search(query2, log, root)

        combined = (
            f"First search results:\n{web1}\n\nSecond search results:\n{web2}"
            if web2
            else web1
        )
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

    # Model answered directly in Turn 2.
    if content2 and result2["finish_reason"] == "stop":
        return _extract_answer(content2, result2["finish_reason"]) or content2.splitlines()[-1].strip()
    return ""


def _run_cot(task_prompt: str, log: EventLog, root: str) -> str:
    """COT primary call (8192 tokens) with 4096-token recovery on empty output."""
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
        # Recovery: suppress extended reasoning to force an immediate answer.
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


# ── Entry point ───────────────────────────────────────────────────────────────

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
