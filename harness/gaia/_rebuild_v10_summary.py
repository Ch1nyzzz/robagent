"""Rebuild v10 summary from trace files (handles truncated summary case)."""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench.gaia.scorer import question_scorer
from bench.gaia.loader import iter_tasks


def main():
    gt = {}
    for t in iter_tasks(split="validation"):
        gt[t["task_id"]] = {
            "final_answer": t["final_answer"],
            "level": t["level"],
        }

    records = []
    for fp in sorted(glob.glob(str(ROOT / "traces" / "runs" / "gaia__*.jsonl")), key=os.path.getmtime):
        try:
            events = [json.loads(l) for l in open(fp)]
        except Exception:
            continue
        if not events:
            continue
        v = (events[0].get("fields") or {}).get("agent_version")
        if v != "v10":
            continue
        tid = events[0]["task_id"]
        answer = ""
        for e in events:
            if e["type"] == "answer.emitted":
                answer = (e.get("fields") or {}).get("answer", "") or ""
        error = None
        for e in events:
            if e["type"] == "run.failed":
                error = (e.get("fields") or {}).get("error", "")
        if tid not in gt:
            continue
        try:
            score = float(int(bool(question_scorer(answer, gt[tid]["final_answer"]))))
        except Exception:
            score = 0.0
        records.append({
            "task_id": tid,
            "level": str(gt[tid]["level"]),
            "score": score,
            "answer": answer,
            "ground_truth": gt[tid]["final_answer"],
            "trace_path": fp,
            "error": error,
        })

    seen = set()
    final = []
    for r in reversed(records):
        if r["task_id"] in seen:
            continue
        seen.add(r["task_id"])
        final.append(r)
    final.reverse()

    with open(ROOT / "traces" / "gaia_v10__summary.jsonl", "w") as f:
        for r in final:
            f.write(json.dumps(r, default=str) + "\n")
    correct = sum(int(r["score"]) for r in final)
    by_lv = {"1": [0, 0], "2": [0, 0], "3": [0, 0]}
    for r in final:
        lv = str(r["level"])
        if lv in by_lv:
            by_lv[lv][1] += 1
            by_lv[lv][0] += int(r["score"])
    print(f"wrote {len(final)} records")
    print(f"v10: {correct}/{len(final)} = {correct/len(final):.3f}")
    for lv in ("1", "2", "3"):
        cc, tt = by_lv[lv]
        if tt:
            print(f"  L{lv}: {cc}/{tt} = {cc/tt:.3f}")


if __name__ == "__main__":
    main()
