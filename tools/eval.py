"""Eval tool for the harness-grow builder agent.

Three scopes, increasing cost:
  unit       — run a small list of task_ids (cheap, fast feedback). Traces
               written to traces/unit_test_runs/ so canonical eval data is
               not polluted.
  train_30   — meta_harness/train_task_ids.txt (deterministic 30-task sample
               held out from test-135).
  full_165   — full GAIA validation set. Expensive — call sparingly.

Usage:
  python tools/eval.py --scope unit --agent-version v14 --tasks <id1> <id2> ...
  python tools/eval.py --scope train_30 --agent-version v14
  python tools/eval.py --scope full_165 --agent-version v14

Always returns JSON to stdout:
  {"scope": ..., "agent_version": ..., "n": ..., "correct": ..., "acc": ...,
   "summary_path": ..., "per_task": [{"task_id": ..., "score": ...}, ...]}
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_IDS = ROOT / "meta_harness" / "train_task_ids.txt"
TEST_IDS = ROOT / "meta_harness" / "test_task_ids.txt"


def _run_subset(
    agent_version: str,
    task_ids_file: Path,
    parallel: int,
    summary_name: str,
    traces_subdir: str | None = None,
) -> dict:
    env = os.environ.copy()
    if traces_subdir is not None:
        env["TRACES_DIR"] = str(ROOT / "traces" / traces_subdir)
    cmd = [
        sys.executable,
        str(ROOT / "run_benchmark.py"),
        "gaia",
        "--agent-version",
        agent_version,
        "--task-ids-file",
        str(task_ids_file),
        "--parallel",
        str(parallel),
    ]
    started = time.time()
    print(f"  cmd: {' '.join(cmd)}", file=sys.stderr, flush=True)
    res = subprocess.run(cmd, cwd=ROOT, env=env)
    elapsed = time.time() - started

    summary_path = ROOT / "traces" / f"gaia_{agent_version}__summary.jsonl"
    if not summary_path.exists():
        return {"error": "no_summary", "elapsed": elapsed, "exit_code": res.returncode}

    rows = [json.loads(l) for l in summary_path.read_text().splitlines() if l.strip()]
    correct = sum(int(r.get("score", 0)) for r in rows)
    return {
        "summary_path": str(summary_path.relative_to(ROOT)),
        "n": len(rows),
        "correct": correct,
        "acc": correct / max(len(rows), 1),
        "elapsed_seconds": elapsed,
        "per_task": [
            {"task_id": r["task_id"], "score": r.get("score", 0), "answer": r.get("answer"), "error": r.get("error")}
            for r in rows
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scope", choices=["unit", "train_30", "full_165"], required=True)
    p.add_argument("--agent-version", required=True)
    p.add_argument("--tasks", nargs="*", default=None, help="task_ids (required for --scope unit)")
    p.add_argument("--parallel", type=int, default=4)
    args = p.parse_args()

    if args.scope == "unit":
        if not args.tasks:
            print("error: --scope unit requires --tasks <id1> <id2> ...", file=sys.stderr)
            sys.exit(2)
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
            f.write("\n".join(args.tasks))
            ids_file = Path(f.name)
        # Protect canonical summary from being clobbered by the unit run.
        canonical = ROOT / "traces" / f"gaia_{args.agent_version}__summary.jsonl"
        backup = None
        if canonical.exists():
            backup = canonical.with_suffix(".jsonl.bak_pre_unit")
            canonical.replace(backup)
        try:
            result = _run_subset(
                args.agent_version,
                ids_file,
                args.parallel,
                f"unit_{args.agent_version}",
                traces_subdir="unit_test_runs",
            )
            # Stash the unit summary under unit_test_runs, restore canonical.
            unit_dest = ROOT / "traces" / "unit_test_runs" / f"gaia_{args.agent_version}__unit_{int(time.time())}.jsonl"
            unit_dest.parent.mkdir(parents=True, exist_ok=True)
            if canonical.exists():
                canonical.replace(unit_dest)
                result["summary_path"] = str(unit_dest.relative_to(ROOT))
        finally:
            if backup and backup.exists():
                backup.replace(canonical)
            ids_file.unlink(missing_ok=True)
    elif args.scope == "train_30":
        if not TRAIN_IDS.exists():
            print(f"error: {TRAIN_IDS} missing", file=sys.stderr)
            sys.exit(3)
        result = _run_subset(args.agent_version, TRAIN_IDS, args.parallel, args.agent_version)
    else:  # full_165
        # Use both train + test ids as the full set
        full_ids = sorted(set(TRAIN_IDS.read_text().split()) | set(TEST_IDS.read_text().split()))
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
            f.write("\n".join(full_ids))
            ids_file = Path(f.name)
        result = _run_subset(args.agent_version, ids_file, args.parallel, args.agent_version)
        ids_file.unlink()

    result["scope"] = args.scope
    result["agent_version"] = args.agent_version
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
