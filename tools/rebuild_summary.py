"""Rebuild a GAIA summary jsonl from per-task trace files.

Usage:
  python tools/rebuild_summary.py --agent-version v14 [--out traces/gaia_v14__summary.jsonl]

Reads the latest trace per task_id from traces/runs/, extracts score / answer /
ground_truth / route from the events, writes a 165-row summary.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def latest_trace(task_id: str) -> str | None:
    files = glob.glob(str(ROOT / f"traces/runs/gaia__{task_id}__*.jsonl"))
    return max(files, key=os.path.getmtime) if files else None


def extract(trace_path: str) -> dict:
    events = [json.loads(l) for l in open(trace_path)]
    score = 0
    answer = ""
    ground = ""
    route = None
    error = None
    for e in events:
        t = e["type"]
        f = e["fields"]
        if t == "eval.scored":
            score = int(f.get("score", 0))
            ground = f.get("ground_truth", "")
        elif t == "answer.emitted":
            answer = f.get("answer", "")
        elif t == "task.routed":
            route = f.get("route")
        elif t in ("llm.failed", "run.failed"):
            error = f.get("error")
    return {"score": float(score), "answer": answer, "ground_truth": ground, "route": route, "error": error}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agent-version", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    from bench.gaia.loader import iter_tasks

    out_path = Path(args.out) if args.out else ROOT / "traces" / f"gaia_{args.agent_version}__summary.jsonl"
    rows = []
    missing = 0
    for task in iter_tasks("validation"):
        tid = task["task_id"]
        p_trace = latest_trace(tid)
        if not p_trace:
            missing += 1
            rows.append({
                "task_id": tid, "level": task["level"], "score": 0.0,
                "answer": None, "ground_truth": task["final_answer"],
                "trace_path": None, "error": "no_trace",
            })
            continue
        ex = extract(p_trace)
        rows.append({
            "task_id": tid,
            "level": task["level"],
            "score": ex["score"],
            "answer": ex["answer"],
            "ground_truth": ex["ground_truth"] or task["final_answer"],
            "trace_path": p_trace,
            "error": ex["error"],
            "route": ex["route"],
        })

    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    total = len(rows)
    correct = sum(1 for r in rows if r["score"])
    print(f"rebuilt: {out_path.relative_to(ROOT)}  n={total} correct={correct} acc={correct/total:.3f}  missing={missing}")


if __name__ == "__main__":
    main()
