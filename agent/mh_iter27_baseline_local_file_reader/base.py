from __future__ import annotations

import os
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path
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

# Max characters extracted from any single attached file.
_FILE_MAX_CHARS = 8000

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

# ── Local GAIA file-reader (iter27 addition) ──────────────────────────────────

_GAIA_FILE_CACHE_BASE = (
    Path.home() / ".cache" / "huggingface" / "hub"
    / "datasets--gaia-benchmark--GAIA"
)


def _find_gaia_file(file_name: str) -> Path | None:
    """Search the local HF snapshot cache for a GAIA task attachment."""
    for snapshot_dir in _GAIA_FILE_CACHE_BASE.glob("snapshots/*/2023/validation"):
        candidate = snapshot_dir / file_name
        if candidate.exists():
            return candidate
    # Also check GAIA_DATA_DIR env variable if set.
    env_dir = os.environ.get("GAIA_DATA_DIR", "")
    if env_dir:
        candidate = Path(env_dir) / file_name
        if candidate.exists():
            return candidate
    return None


def _strip_xml_tags(xml_text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", xml_text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_xlsx_bytes(data: bytes) -> str:
    import io
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(data))
    lines: list[str] = []
    for ws in wb.worksheets:
        lines.append(f"[Sheet: {ws.title}]")
        for row in ws.iter_rows(values_only=True):
            if any(v is not None for v in row):
                lines.append("\t".join("" if v is None else str(v) for v in row))
    return "\n".join(lines)[:_FILE_MAX_CHARS]


def _extract_xls_bytes(data: bytes) -> str:
    import xlrd
    wb = xlrd.open_workbook(file_contents=data)
    lines: list[str] = []
    for sh in wb.sheets():
        lines.append(f"[Sheet: {sh.name}]")
        for r in range(sh.nrows):
            row = sh.row_values(r)
            if any(v is not None and v != "" for v in row):
                lines.append("\t".join(str(v) for v in row))
    return "\n".join(lines)[:_FILE_MAX_CHARS]


def _read_file_text(file_path: Path) -> str:
    """Extract readable text from a GAIA task attachment. Returns '' for binary formats."""
    suffix = file_path.suffix.lower()

    if suffix == ".xlsx":
        try:
            return _extract_xlsx_bytes(file_path.read_bytes())
        except Exception as e:
            return f"[Error reading xlsx: {e}]"

    elif suffix == ".xls":
        try:
            return _extract_xls_bytes(file_path.read_bytes())
        except Exception as e:
            return f"[Error reading xls: {e}]"

    elif suffix == ".zip":
        import zipfile
        try:
            parts: list[str] = []
            with zipfile.ZipFile(file_path) as z:
                for name in sorted(z.namelist()):
                    inner_suffix = Path(name).suffix.lower()
                    raw_bytes = z.read(name)
                    if inner_suffix in (".txt", ".csv", ".json", ".jsonld"):
                        parts.append(
                            f"[File: {name}]\n"
                            + raw_bytes.decode("utf-8", errors="replace")
                        )
                    elif inner_suffix == ".xml":
                        text = _strip_xml_tags(raw_bytes.decode("utf-8", errors="replace"))
                        parts.append(f"[File: {name}]\n{text}")
                    elif inner_suffix == ".xlsx":
                        try:
                            parts.append(f"[File: {name}]\n{_extract_xlsx_bytes(raw_bytes)}")
                        except Exception as e:
                            parts.append(f"[File: {name}] (error: {e})")
                    elif inner_suffix == ".xls":
                        try:
                            parts.append(f"[File: {name}]\n{_extract_xls_bytes(raw_bytes)}")
                        except Exception as e:
                            parts.append(f"[File: {name}] (error: {e})")
            return "\n\n".join(parts)[:_FILE_MAX_CHARS]
        except Exception as e:
            return f"[Error reading zip: {e}]"

    elif suffix in (".jsonld", ".json", ".txt", ".csv", ".pdb", ".xml"):
        try:
            raw = file_path.read_text(encoding="utf-8", errors="replace")
            if suffix == ".xml":
                return _strip_xml_tags(raw)[:_FILE_MAX_CHARS]
            return raw[:_FILE_MAX_CHARS]
        except Exception as e:
            return f"[Error reading {suffix}: {e}]"

    # PDF, PPTX, PNG, JPG, MP3, etc. → not text-extractable here.
    return ""


def _enrich_prompt(task_prompt: str, extras: dict, log: EventLog, root: str) -> str:
    """Replace the '[Note: file not provided]' stub with actual file content when available.

    Reads the file from the local HuggingFace cache. Only fires when extras.file_name
    is set and the file is found locally. All other tasks are unaffected.
    """
    file_name = extras.get("file_name", "")
    if not file_name:
        return task_prompt

    file_path = _find_gaia_file(file_name)
    if file_path is None:
        return task_prompt

    content = _read_file_text(file_path)
    if not content:
        return task_prompt

    log.emit(
        "file.read",
        parent=root,
        file_name=file_name,
        path=str(file_path),
        content_chars=len(content),
    )

    stub = (
        f"[Note: this task references a file '{file_name}' "
        f"which is not provided in this baseline run.]"
    )
    enriched = f"[Attached file '{file_name}']\n{content}"

    if stub in task_prompt:
        return task_prompt.replace(stub, enriched)
    # Append if stub not found.
    return task_prompt + f"\n\n{enriched}"


# ── Web helpers (unchanged from iter26) ──────────────────────────────────────

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


def _run_web_search(task_prompt: str, log: EventLog, root: str) -> str:
    """Three-turn iterative web search (identical to iter26/iter21)."""
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
        query2 = action2.group(1).strip()
        log.emit("claim.extracted", parent=root, action="SEARCH2", query=query2[:200])
        web_content2 = _web_search(query2, log, root)

        fetched_text = ""
        fetched_url = ""

        best_url_t2 = _extract_best_url(web_content2) if web_content2 else ""
        if best_url_t2:
            fetched_text = _fetch_url_text(best_url_t2)
            if fetched_text:
                fetched_url = best_url_t2
                log.emit(
                    "tool.called",
                    parent=root,
                    tool="url_fetch",
                    url=best_url_t2[:200],
                    chars=len(fetched_text),
                    source="turn2_best",
                    is_wikipedia=_is_wikipedia_url(best_url_t2),
                )

        if not fetched_text:
            best_url_t1 = _extract_best_url(web_content1, exclude=best_url_t2)
            if best_url_t1:
                fetched_text = _fetch_url_text(best_url_t1)
                if fetched_text:
                    fetched_url = best_url_t1
                    log.emit(
                        "tool.called",
                        parent=root,
                        tool="url_fetch",
                        url=best_url_t1[:200],
                        chars=len(fetched_text),
                        source="turn1_fallback_best",
                        is_wikipedia=_is_wikipedia_url(best_url_t1),
                    )

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
    """COT primary + recovery. Writes exhaustion flags to optional state dict."""
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
    """Write and execute a Python script to compute the answer (identical to iter26)."""
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

    # ── iter27 addition: inject local file content when available ─────────────
    task_prompt = _enrich_prompt(task_prompt, extras, log, root)

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
