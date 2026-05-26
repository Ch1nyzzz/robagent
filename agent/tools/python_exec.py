"""python_exec tool — run Python code in a subprocess and return stdout+stderr.

Runs as `python -c <code>` with a wall-time limit. Output combines stdout
and stderr; the LLM sees whichever the script produced. Designed for
calculations, parsing snippets, quick data wrangling — not long-running jobs.
"""
from __future__ import annotations

import subprocess
import sys
from typing import Any


_TIMEOUT = 30
_MAX_CHARS = 12000


SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "python_exec",
        "description": (
            "Execute Python code (single subprocess, wall-time capped at "
            f"{_TIMEOUT}s). Returns combined stdout+stderr. Use for math, "
            "string manipulation, JSON/CSV parsing, quick computation. "
            "Make sure to print() any value you want to see."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source code to execute. Must print() any output you want.",
                },
            },
            "required": ["code"],
        },
    },
}


def run(args: dict) -> str:
    code = args.get("code") or ""
    if not code.strip():
        return "ERROR: python_exec requires a non-empty code argument."
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: python_exec exceeded {_TIMEOUT}s wall-time limit."
    except Exception as e:
        return f"ERROR launching python_exec: {type(e).__name__}: {e}"
    out = (proc.stdout or "")
    err = (proc.stderr or "")
    combined = out
    if err:
        combined = (combined + "\n[stderr]\n" + err).strip()
    if proc.returncode != 0:
        combined = f"[exit_code={proc.returncode}]\n{combined}"
    if len(combined) > _MAX_CHARS:
        combined = combined[:_MAX_CHARS] + f"\n\n[... truncated; total {len(combined)} chars]"
    return combined or "[no output]"
