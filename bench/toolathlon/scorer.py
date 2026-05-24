"""Toolathlon scoring.

Toolathlon scoring is binary 0/1, computed by per-task Python verifiers in
`Toolathlon-src/tasks/finalpool/<name>/evaluation/main.py`. The verifier
runs INSIDE the task container (Step 6 of run_single_decoupled.sh) and the
container writes `eval_res.json` into the host-bound dump directory:

    {"pass": true|false, "details": "<verifier stdout summary>"}

`status.json` next to it tracks lifecycle (preprocess/running/evaluation).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def score_from_dump(dump_dir: str | Path) -> dict[str, Any]:
    """Read score from a per-task dump directory.

    Returns:
      {
        "score":   0.0 | 1.0,
        "passed":  bool,
        "details": str,
        "status":  {preprocess, running, evaluation},
        "error":   None | str,
      }

    A missing eval_res.json => score 0 with error description (so the outer
    pipeline doesn't crash when a task dies before eval).
    """
    p = Path(dump_dir)
    eval_res = p / "eval_res.json"
    status = p / "status.json"

    status_data = None
    if status.exists():
        try:
            status_data = json.loads(status.read_text(encoding="utf-8"))
        except Exception:
            status_data = None

    if not eval_res.exists():
        return {
            "score": 0.0,
            "passed": False,
            "details": "",
            "status": status_data,
            "error": f"eval_res.json missing under {p}",
        }

    try:
        data = json.loads(eval_res.read_text(encoding="utf-8"))
    except Exception as e:
        return {
            "score": 0.0,
            "passed": False,
            "details": "",
            "status": status_data,
            "error": f"failed to parse eval_res.json: {e!r}",
        }

    passed = bool(data.get("pass"))
    return {
        "score": 1.0 if passed else 0.0,
        "passed": passed,
        "details": data.get("details", ""),
        "status": status_data,
        "error": None,
    }
