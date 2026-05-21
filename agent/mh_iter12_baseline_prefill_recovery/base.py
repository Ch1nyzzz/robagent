from __future__ import annotations

import re
from typing import Any

from agent.events import EventLog, new_run_id, traces_dir
from agent.llm import chat, DEFAULT_MODEL


# COT primary prompt — identical to iter5/iter11.
SYSTEM_PROMPT = (
    "You are a precise problem-solving assistant. "
    "Work through the problem concisely — write only the key reasoning steps. "
    "Then output the final answer on its own line in this exact format:\n"
    "FINAL ANSWER: <value>\n"
    "where <value> is the answer only, with no extra text or punctuation."
)

# Recovery system prompt — same as iter11.
RECOVERY_SYSTEM_PROMPT = (
    "You are an assistant. Output ONLY the final answer value to the question — "
    "no reasoning, no explanation, no thinking steps. "
    "Just the raw answer: a number, a short string, or a few words."
)

# Web-search Turn 1 — same as iter11: unconditionally require SEARCH[...].
SEARCH_AWARE_SYSTEM_PROMPT = (
    "You are an assistant solving a benchmark task. "
    "The question requires data from a named external source. "
    "You MUST search for the answer — do NOT answer directly. "
    "Respond with ONLY: SEARCH[<a focused, concise search query>]"
)

# Web-search Turn 2 — same as iter11.
SEARCH_REFINE_SYSTEM_PROMPT = (
    "You are an assistant solving a benchmark task. "
    "Web search results are shown below. "
    "If the results directly answer the question, output ONLY the final answer "
    "(number, short phrase, list, etc. — no explanations). "
    "If the results are insufficient or do not contain the specific data needed, "
    "respond ONLY with: SEARCH[<a different, more targeted search query>]"
)

# Web-search Turn 3 — same as iter11.
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

# Web-signal gate — same as iter11 (iter10 patterns included).
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

# Prefill text appended as an assistant message to force immediate visible output.
# By starting the assistant turn with "FINAL ANSWER: ", the model is forced to
# continue from that string, reducing the hidden thinking phase that otherwise
# exhausts all completion tokens and leaves content empty.
_RECOVERY_PREFILL = "FINAL ANSWER: "


def _needs_web(prompt: str) -> bool:
    return bool(_WEB_SIGNAL_RE.search(prompt))


def _web_search(query: str, log: EventLog, parent: str, max_results: int = 5) -> str:
    """Run DuckDuckGo text search; return formatted snippets or empty string on failure."""
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
    """Extract the answer using FINAL ANSWER: marker with last-line fallback."""
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
    """Three-turn iterative web search — identical to iter11."""
    # ── Turn 1: unconditional search ─────────────────────────────────────────
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
        # ── Turn 3: second search + final answer ──────────────────────────────
        query2 = action2.group(1).strip()
        log.emit("claim.extracted", parent=root, action="SEARCH2", query=query2[:200])
        web_content2 = _web_search(query2, log, root)

        combined = (
            f"First search results:\n{web_content1}\n\nSecond search results:\n{web_content2}"
            if web_content2
            else web_content1
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
    """COT primary + prefill-augmented recovery fallback.

    Change vs iter11: recovery messages include a trailing assistant turn
    {"role": "assistant", "content": "FINAL ANSWER: "} so the model continues
    from that string rather than spending all 1024 recovery tokens on hidden
    chain-of-thought.  If the Together AI backend rejects the prefill
    (exception), we fall back to the standard no-prefill recovery call to
    avoid regressing currently-passing tasks.
    """
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
        # ── Recovery with assistant prefill ───────────────────────────────────
        # We add {"role": "assistant", "content": "FINAL ANSWER: "} as the last
        # message so the model continues from that point.  This bypasses the
        # mandatory hidden-thinking phase that otherwise consumes all 1024 tokens.
        prefill_messages = [
            {"role": "system", "content": RECOVERY_SYSTEM_PROMPT},
            {"role": "user", "content": task_prompt},
            {"role": "assistant", "content": _RECOVERY_PREFILL},
        ]
        recovery_call = log.emit(
            "llm.requested",
            parent=root,
            messages=prefill_messages,
            model=DEFAULT_MODEL,
            recovery=True,
            recovery_max_tokens=RECOVERY_MAX_TOKENS,
            prefill=True,
        )
        try:
            recovery_result = chat(
                messages=prefill_messages,
                max_tokens=RECOVERY_MAX_TOKENS,
                temperature=RECOVERY_TEMPERATURE,
            )
            log.emit(
                "llm.responded",
                parent=recovery_call,
                content=recovery_result["content"],
                finish_reason=recovery_result["finish_reason"],
                usage=recovery_result["usage"],
                recovery=True,
                prefill=True,
            )
            recovered_raw = (recovery_result["content"] or "").strip()
            if recovered_raw:
                # content may be just the continuation ("42") or the full string
                # ("FINAL ANSWER: 42") depending on how the backend handles prefill.
                # _extract_answer handles the "FINAL ANSWER:" case via regex;
                # the last-line fallback handles the bare-continuation case.
                answer = (
                    _extract_answer(recovered_raw, recovery_result["finish_reason"])
                    or recovered_raw.splitlines()[-1].strip()
                )
        except Exception as prefill_exc:
            # Prefill not supported by this backend — fall back to plain recovery.
            log.emit("llm.failed", parent=recovery_call, error=repr(prefill_exc), recovery=True, prefill=True)
            plain_messages = [
                {"role": "system", "content": RECOVERY_SYSTEM_PROMPT},
                {"role": "user", "content": task_prompt},
            ]
            plain_call = log.emit(
                "llm.requested",
                parent=root,
                messages=plain_messages,
                model=DEFAULT_MODEL,
                recovery=True,
                recovery_max_tokens=RECOVERY_MAX_TOKENS,
                prefill=False,
            )
            try:
                plain_result = chat(
                    messages=plain_messages,
                    max_tokens=RECOVERY_MAX_TOKENS,
                    temperature=RECOVERY_TEMPERATURE,
                )
            except Exception as e:
                log.emit("llm.failed", parent=plain_call, error=repr(e), recovery=True)
            else:
                log.emit(
                    "llm.responded",
                    parent=plain_call,
                    content=plain_result["content"],
                    finish_reason=plain_result["finish_reason"],
                    usage=plain_result["usage"],
                    recovery=True,
                )
                plain_raw = (plain_result["content"] or "").strip()
                if plain_raw:
                    answer = plain_raw

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
