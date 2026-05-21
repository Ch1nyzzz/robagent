"""Extra deterministic readers for v13.

- `pdb_read`: structural-record extraction from PDB chemistry files.
- `zip_read`: recurse a zip archive, dispatch each entry to the existing
  v7 `read_gaia_file` machinery, return concatenated text.
- `audio_read`: transcribe via Together's Whisper endpoint
  (`openai/whisper-large-v3`, free tier compatible with the existing key).
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import zipfile
from pathlib import Path
from typing import Any

from agent.events import EventLog


_MAX_BODY_CHARS = 16000


def _quote_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]


def _truncate(text: str) -> str:
    return text[:_MAX_BODY_CHARS]


# ------------------------------------------------------------- pdb_read --
_PDB_INFO_RECORDS = ("HEADER", "TITLE", "COMPND", "SOURCE", "KEYWDS", "EXPDTA",
                     "AUTHOR", "REVDAT", "JRNL", "REMARK   2", "REMARK   3")


def pdb_read(path: str) -> str:
    """Extract title / header / metadata records from a PDB file as plain text.

    PDB files are line-oriented with 6-char record types in columns 1-6.
    """
    out: list[str] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i > 5000:
                    break
                rec = line[:10].strip()
                if any(line.startswith(p) for p in _PDB_INFO_RECORDS):
                    out.append(line.rstrip())
    except Exception as e:
        return f"(pdb parse failed: {e})"
    return "\n".join(out)


# ----------------------------------------------------------- zip_read ---
def zip_read(path: str) -> str:
    """Recursively dump readable text content from a zip archive.

    Internally reads each entry's bytes and emits a concatenated string with
    section headers. Binary entries (images, audio) are listed by name only.
    """
    parts: list[str] = []
    try:
        with zipfile.ZipFile(path) as z:
            for info in z.infolist()[:50]:
                if info.is_dir():
                    continue
                name = info.filename
                ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
                parts.append(f"--- entry: {name} ({info.file_size} bytes) ---")
                if ext in {"txt", "py", "md", "json", "jsonld", "csv", "xml", "html", "log"}:
                    try:
                        with z.open(info) as fh:
                            raw = fh.read(_MAX_BODY_CHARS)
                        parts.append(raw.decode("utf-8", errors="replace"))
                    except Exception as e:
                        parts.append(f"(read failed: {e})")
                elif ext == "pdb":
                    try:
                        with z.open(info) as fh:
                            raw = fh.read(_MAX_BODY_CHARS).decode("utf-8", errors="replace")
                        for line in raw.splitlines()[:200]:
                            if any(line.startswith(p) for p in _PDB_INFO_RECORDS):
                                parts.append(line)
                    except Exception as e:
                        parts.append(f"(pdb in zip failed: {e})")
                # leave unknown / binary entries as metadata only
    except Exception as e:
        return f"(zip parse failed: {e})"
    return "\n".join(parts)


# --------------------------------------------------------- audio_read ---
def audio_read(path: str) -> str:
    """Transcribe an audio file via Together's free Whisper endpoint."""
    try:
        import requests
    except Exception:
        return "(transcription unavailable: requests not installed)"
    api_key = os.environ.get("TOGETHER_AI_API") or ""
    if not api_key:
        return "(transcription unavailable: TOGETHER_AI_API missing)"
    try:
        with open(path, "rb") as fh:
            audio_bytes = fh.read()
    except Exception as e:
        return f"(audio open failed: {e})"
    try:
        resp = requests.post(
            "https://api.together.xyz/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (Path(path).name, audio_bytes, "audio/mpeg")},
            data={"model": "openai/whisper-large-v3", "response_format": "text"},
            timeout=120,
        )
        if resp.status_code != 200:
            return f"(transcription HTTP {resp.status_code}: {resp.text[:120]})"
        text = resp.text or ""
        if not text:
            return "(empty transcription)"
        return text
    except Exception as e:
        return f"(transcription request failed: {e})"


# ------------------------------------------------------- entrypoint -----
def read_extra_file(
    task_id: str,
    file_name: str,
    path: str,
    ext: str,
    *,
    log: EventLog,
    parent: str,
) -> dict[str, Any]:
    """Top-level dispatcher for v13-supported extensions.

    Returns {ok, kind, content, source_id} on success or {ok=False, reason}.
    """
    call_id = log.emit(
        "tool.called",
        parent=parent,
        tool="read_extra_file",
        args={"task_id": task_id, "file_name": file_name, "ext": ext},
    )
    text = ""
    kind = f"file:{ext}"
    try:
        if ext == "pdb":
            text = pdb_read(path)
        elif ext == "zip":
            text = zip_read(path)
        elif ext in {"mp3", "wav", "m4a", "flac", "ogg"}:
            text = audio_read(path)
            kind = f"audio:{ext}"
        else:
            log.emit(
                "tool.returned",
                parent=call_id,
                tool="read_extra_file",
                ok=False,
                error=f"unsupported extension: {ext}",
            )
            return {"ok": False, "reason": f"unsupported_extension:{ext}"}
    except Exception as e:
        log.emit(
            "tool.returned",
            parent=call_id,
            tool="read_extra_file",
            ok=False,
            error=repr(e)[:200],
        )
        return {"ok": False, "reason": f"parse_error:{e}"}

    text = _truncate(text or "")
    if not text.strip():
        log.emit(
            "tool.returned",
            parent=call_id,
            tool="read_extra_file",
            ok=False,
            error="empty content",
        )
        return {"ok": False, "reason": "empty_content"}

    h = _quote_hash(text)
    source_id = f"S_file_{h}"
    log.emit(
        "source.opened",
        parent=call_id,
        source_id=source_id,
        kind=kind,
        url=f"file://{path}",
        title=file_name,
        content_chars=len(text),
        content_hash=h,
    )
    log.emit(
        "tool.returned",
        parent=call_id,
        tool="read_extra_file",
        ok=True,
        source_id=source_id,
        n_chars=len(text),
    )
    return {
        "ok": True,
        "kind": kind,
        "title": file_name,
        "url": f"file://{path}",
        "content": text,
        "source_id": source_id,
        "content_hash": h,
    }
