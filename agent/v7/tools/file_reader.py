"""File reader tool for GAIA file_name extras.

Resolves a task to a local cached path (downloading on demand from HF), then
parses the file to plain text by extension. Emits tool.called / tool.returned
/ source.opened events.

Supported text-extractable extensions:
  - txt, json, jsonld, csv, py, xml, html, md
  - pdf (via pypdf)
  - docx (via python-docx if available, otherwise via zipfile XML)
  - xlsx (via openpyxl)
  - pptx (via python-pptx if available, otherwise via zipfile XML)

For unsupported (mp3, png, jpg, pdb, zip), returns ok=False and the workflow
falls back to honest BLOCKED.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any

from agent.events import EventLog

_MAX_BODY_CHARS = 16000
_HF_REPO = "gaia-benchmark/GAIA"


def gaia_file_path(task_id: str, file_name: str) -> str | None:
    """Resolve a GAIA file_name to a local filesystem path.

    Downloads from HF on demand. Returns None if unable to fetch.
    """
    if not file_name:
        return None
    try:
        from huggingface_hub import hf_hub_download

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        # GAIA layout: 2023/{split}/{file_name}; we know validation here.
        for split in ("validation", "test"):
            try:
                p = hf_hub_download(
                    repo_id=_HF_REPO,
                    repo_type="dataset",
                    filename=f"2023/{split}/{file_name}",
                    token=token,
                )
                if os.path.exists(p):
                    return p
            except Exception:
                continue
    except Exception:
        pass
    return None


def _truncate(text: str) -> str:
    return text[:_MAX_BODY_CHARS]


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _read_json(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    return json.dumps(data, indent=2, ensure_ascii=False, default=str)


def _read_csv(path: str) -> str:
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        rdr = csv.reader(f)
        for i, row in enumerate(rdr):
            rows.append(",".join(row))
            if i > 500:
                rows.append("...(truncated)")
                break
    return "\n".join(rows)


def _read_pdf(path: str) -> str:
    try:
        import pypdf

        reader = pypdf.PdfReader(path)
        parts = []
        for page in reader.pages[:20]:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n\n".join(parts)
    except Exception:
        try:
            import pdfplumber

            parts = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages[:20]:
                    parts.append(page.extract_text() or "")
            return "\n\n".join(parts)
        except Exception as e:
            return f"(pdf parse failed: {e})"


def _read_xlsx(path: str) -> str:
    try:
        from openpyxl import load_workbook

        wb = load_workbook(path, read_only=True, data_only=True)
        parts = []
        for ws in wb.worksheets:
            parts.append(f"--- Sheet: {ws.title} ---")
            for i, row in enumerate(ws.iter_rows(values_only=True)):
                cells = [str(c) if c is not None else "" for c in row]
                parts.append("\t".join(cells))
                if i > 200:
                    parts.append("...(truncated)")
                    break
        return "\n".join(parts)
    except Exception as e:
        return f"(xlsx parse failed: {e})"


def _read_docx(path: str) -> str:
    try:
        with zipfile.ZipFile(path) as z:
            with z.open("word/document.xml") as f:
                xml = f.read().decode("utf-8", errors="replace")
        # strip namespaces and keep text inside <w:t>
        text_parts = re.findall(r"<w:t[^>]*>([^<]*)</w:t>", xml)
        return "\n".join(text_parts)
    except Exception as e:
        return f"(docx parse failed: {e})"


def _read_pptx(path: str) -> str:
    try:
        out = []
        with zipfile.ZipFile(path) as z:
            slide_names = sorted(
                n for n in z.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml")
            )
            for sn in slide_names[:40]:
                with z.open(sn) as f:
                    xml = f.read().decode("utf-8", errors="replace")
                texts = re.findall(r"<a:t[^>]*>([^<]*)</a:t>", xml)
                if texts:
                    out.append(f"--- {sn} ---")
                    out.extend(texts)
        return "\n".join(out)
    except Exception as e:
        return f"(pptx parse failed: {e})"


def read_gaia_file(
    task_id: str,
    file_name: str,
    *,
    log: EventLog,
    parent: str,
) -> dict[str, Any]:
    """Resolve, parse, and return a structured source dict.

    Returns {ok, kind, path, content, source_id} or {ok=False, reason}.
    """
    call_id = log.emit(
        "tool.called",
        parent=parent,
        tool="read_gaia_file",
        args={"task_id": task_id, "file_name": file_name},
    )
    path = gaia_file_path(task_id, file_name)
    if not path:
        log.emit(
            "tool.returned",
            parent=call_id,
            tool="read_gaia_file",
            ok=False,
            error="file path resolution failed",
        )
        return {"ok": False, "reason": "path_not_found"}

    ext = file_name.lower().rsplit(".", 1)[-1] if "." in file_name else ""
    try:
        if ext in {"txt", "py", "xml", "html", "md", "log"}:
            text = _read_text(path)
        elif ext in {"json", "jsonld"}:
            text = _read_json(path)
        elif ext == "csv":
            text = _read_csv(path)
        elif ext == "pdf":
            text = _read_pdf(path)
        elif ext == "xlsx":
            text = _read_xlsx(path)
        elif ext == "docx":
            text = _read_docx(path)
        elif ext == "pptx":
            text = _read_pptx(path)
        else:
            log.emit(
                "tool.returned",
                parent=call_id,
                tool="read_gaia_file",
                ok=False,
                error=f"unsupported extension: {ext}",
            )
            return {"ok": False, "reason": f"unsupported_extension:{ext}"}

        text = _truncate(text or "")
        if not text.strip():
            log.emit(
                "tool.returned",
                parent=call_id,
                tool="read_gaia_file",
                ok=False,
                error="empty content",
            )
            return {"ok": False, "reason": "empty_content"}

        h = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]
        source_id = f"S_file_{h}"
        log.emit(
            "source.opened",
            parent=call_id,
            source_id=source_id,
            kind=f"file:{ext}",
            url=f"file://{path}",
            content_chars=len(text),
            content_hash=h,
        )
        log.emit(
            "tool.returned",
            parent=call_id,
            tool="read_gaia_file",
            ok=True,
            source_id=source_id,
            n_chars=len(text),
        )
        return {
            "ok": True,
            "kind": f"file:{ext}",
            "title": file_name,
            "url": f"file://{path}",
            "content": text,
            "source_id": source_id,
            "content_hash": h,
        }
    except Exception as e:
        log.emit(
            "tool.returned",
            parent=call_id,
            tool="read_gaia_file",
            ok=False,
            error=repr(e)[:200],
        )
        return {"ok": False, "reason": f"parse_error:{e}"}
