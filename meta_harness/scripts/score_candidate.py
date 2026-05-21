"""Score a candidate on train-30 from its summary jsonl, update frontier + evolution_summary.

Usage:
  python meta_harness/scripts/score_candidate.py \
      --agent-name v0 --summary-path traces/gaia__summary.jsonl
  python meta_harness/scripts/score_candidate.py \
      --agent-name mh_iter1_<slug> --summary-path traces/gaia_mh_iter1_<slug>__summary.jsonl \
      --hypothesis "..." --changes "..."
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MH = ROOT / "meta_harness"
TRAIN_FILE = MH / "train_task_ids.txt"
# FRONTIER and SUMMARY are set in main() after parsing --logs-dir.
FRONTIER: Path  # type: ignore[assignment]
SUMMARY: Path  # type: ignore[assignment]


def load_train_ids() -> set[str]:
    return {ln.strip() for ln in TRAIN_FILE.read_text().splitlines() if ln.strip()}


def load_summary(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            rows[r["task_id"]] = r
    return rows


def score_on_subset(rows: dict[str, dict], ids: set[str]) -> dict:
    hit = [r for tid, r in rows.items() if tid in ids]
    n = len(hit)
    correct = sum(1 for r in hit if (r.get("score") or 0) > 0)
    return {"correct": correct, "total": n, "acc": round(correct / max(n, 1), 4)}


def update_frontier(agent: str, rows: dict[str, dict], train_ids: set[str]) -> dict:
    frontier: dict = {}
    if FRONTIER.exists():
        frontier = json.loads(FRONTIER.read_text())
    per_task = frontier.setdefault("per_task", {})
    for tid in train_ids:
        r = rows.get(tid)
        if r is None:
            continue
        s = float(r.get("score") or 0)
        cur = per_task.get(tid)
        if cur is None or s > cur["score"]:
            per_task[tid] = {"agent": agent, "score": s, "answer": r.get("answer")}
    frontier_correct = sum(1 for v in per_task.values() if v["score"] > 0)
    frontier["frontier_score"] = {
        "correct": frontier_correct,
        "total": len(train_ids),
        "acc": round(frontier_correct / max(len(train_ids), 1), 4),
    }
    frontier["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    FRONTIER.write_text(json.dumps(frontier, indent=2))
    return frontier["frontier_score"]


def append_summary(entry: dict) -> None:
    with SUMMARY.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agent-name", required=True)
    p.add_argument("--summary-path", required=True, type=Path)
    p.add_argument("--iteration", type=int, default=0)
    p.add_argument("--hypothesis", default="")
    p.add_argument("--changes", default="")
    p.add_argument("--no-frontier", action="store_true",
                   help="just score, do not update frontier (for ad-hoc evaluation)")
    p.add_argument("--logs-dir", type=Path, default=MH / "logs",
                   help="frontier_val.json + evolution_summary.jsonl live here")
    args = p.parse_args()

    global FRONTIER, SUMMARY
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    FRONTIER = args.logs_dir / "frontier_val.json"
    SUMMARY = args.logs_dir / "evolution_summary.jsonl"

    train_ids = load_train_ids()
    rows = load_summary(args.summary_path)
    train_score = score_on_subset(rows, train_ids)

    print(f"[{args.agent_name}] train: {train_score['correct']}/{train_score['total']} = {train_score['acc']}")

    frontier_after = None
    if not args.no_frontier:
        frontier_after = update_frontier(args.agent_name, rows, train_ids)
        print(f"[frontier after] {frontier_after['correct']}/{frontier_after['total']} = {frontier_after['acc']}")

    entry = {
        "iteration": args.iteration,
        "agent": args.agent_name,
        "hypothesis": args.hypothesis,
        "changes": args.changes,
        "train_score": train_score,
        "frontier_after": frontier_after,
        "summary_path": str(args.summary_path),
        "scored_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    append_summary(entry)


if __name__ == "__main__":
    main()
