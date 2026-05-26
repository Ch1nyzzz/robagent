"""file_read tool — read a GAIA attached file by file_name.

GAIA files live in the HuggingFace dataset cache, downloaded when
`bench/gaia/loader.iter_tasks` calls `load_dataset`. The tool resolves a
file_name to its on-disk path and returns text content (with format-specific
parsing for spreadsheets and zip archives).

Tool spec is OpenAI function-calling shape; consumed by `agent.tools.TOOL_SPECS`.
"""
from __future__ import annotations

import glob
import io
import os
import re
import zipfile
from typing import Any


_GAIA_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--gaia-benchmark--GAIA"
    "/snapshots/*/2023/{split}/{name}"
)

_MAX_CHARS = 20000
_MEMBER_MAX_CHARS = 8000


SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "file_read",
        "description": (
            "Read the text content of an attached file (referenced by file_name "
            "in the task). Supports .txt, .csv, .py, .json, .jsonld, .md, .xml, "
            ".xlsx, .xls, .zip (lists members + parses each)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": "The attached file's name as named in the task prompt.",
                },
            },
            "required": ["file_name"],
        },
    },
}


def _find_path(file_name: str) -> str | None:
    for split in ("validation", "test"):
        matches = glob.glob(_GAIA_GLOB.format(split=split, name=file_name))
        if matches:
            return matches[0]
    return None


def _strip_xml_tags(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return text


def _read_xlsx(path_or_bytes) -> str:
    import openpyxl
    if isinstance(path_or_bytes, (bytes, bytearray)):
        wb = openpyxl.load_workbook(io.BytesIO(path_or_bytes))
    else:
        wb = openpyxl.load_workbook(path_or_bytes)
    lines: list[str] = []
    for ws in wb.worksheets:
        if ws.title:
            lines.append(f"Sheet: {ws.title}")
        for row in ws.iter_rows(values_only=True):
            lines.append("\t".join("" if v is None else str(v) for v in row))
    return "\n".join(lines)


def _read_xls(path_or_bytes) -> str:
    import xlrd
    if isinstance(path_or_bytes, (bytes, bytearray)):
        wb = xlrd.open_workbook(file_contents=path_or_bytes)
    else:
        wb = xlrd.open_workbook(path_or_bytes)
    lines: list[str] = []
    for i in range(wb.nsheets):
        ws = wb.sheet_by_index(i)
        for r in range(ws.nrows):
            lines.append("\t".join(str(ws.cell_value(r, c)) for c in range(ws.ncols)))
    return "\n".join(lines)


def _zip_member_text(z: zipfile.ZipFile, name: str) -> str:
    ext = os.path.splitext(name.lower())[1]
    try:
        data = z.read(name)
        if ext in (".txt", ".csv", ".py", ".json", ".jsonld", ".md"):
            return data.decode("utf-8", errors="replace")[:_MEMBER_MAX_CHARS]
        if ext == ".xml":
            return _strip_xml_tags(data.decode("utf-8", errors="replace"))[:_MEMBER_MAX_CHARS]
        if ext == ".xlsx":
            return _read_xlsx(data)[:_MEMBER_MAX_CHARS]
        if ext == ".xls":
            return _read_xls(data)[:_MEMBER_MAX_CHARS]
    except Exception as e:
        return f"[error parsing zip member {name!r}: {type(e).__name__}: {e}]"
    return ""


def _file_to_text(path: str, file_name: str) -> str:
    ext = os.path.splitext(file_name.lower())[1]
    if ext in (".txt", ".csv", ".py", ".json", ".jsonld", ".md"):
        with open(path, "r", errors="replace") as f:
            return f.read()[:_MAX_CHARS]
    if ext == ".xml":
        with open(path, "r", errors="replace") as f:
            return _strip_xml_tags(f.read())[:_MAX_CHARS]
    if ext == ".xlsx":
        return _read_xlsx(path)[:_MAX_CHARS]
    if ext == ".xls":
        return _read_xls(path)[:_MAX_CHARS]
    if ext == ".zip":
        parts: list[str] = []
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                member_text = _zip_member_text(z, name)
                if member_text:
                    parts.append(f"[File: {name}]\n{member_text}")
        return ("\n\n".join(parts))[:_MAX_CHARS]
    return f"[unsupported file extension {ext!r}; raw bytes path: {path}]"


def run(args: dict) -> str:
    file_name = (args.get("file_name") or "").strip()
    if not file_name:
        return "ERROR: file_read requires a non-empty file_name argument."
    path = _find_path(file_name)
    if not path:
        return f"ERROR: file {file_name!r} not found in GAIA cache."
    try:
        return _file_to_text(path, file_name)
    except Exception as e:
        return f"ERROR reading {file_name!r}: {type(e).__name__}: {e}"
