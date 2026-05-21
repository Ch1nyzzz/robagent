"""Re-score an agent version by reading the latest trace file per task_id.

Useful when the summary file got overwritten by a partial retry run, or
when we want score per version independently of the runner's summary path.

Usage:
    python harness/gaia/_score_from_traces.py [agent_version_marker]

agent_version_marker is the optional 'agent_version' field value to match
in run.started events. Defaults to 'v1'. Pass 'any' to include every trace.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench.gaia.scorer import question_scorer  # noqa: E402


def _latest_traces() -> dict[str, str]:
    files = sorted(
        glob.glob(str(ROOT / "traces" / "runs" / "gaia__*.jsonl")),
        key=lambda p: os.stat(p).st_mtime,
        reverse=True,
    )
    out: dict[str, str] = {}
    for fp in files:
        base = os.path.basename(fp)
        parts = base.split("__")
        if len(parts) < 3:
            continue
        out.setdefault(parts[1], fp)
    return out


def _gt_map() -> dict[str, dict]:
    out: dict[str, dict] = {}
    p = ROOT / "traces" / "gaia__summary.jsonl"
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[rec["task_id"]] = rec
    return out


def main(marker: str = "v1") -> None:
    latest = _latest_traces()
    gt = _gt_map()
    correct = 0
    total = 0
    by_level_total = Counter()
    by_level_correct = Counter()
    fail = Counter()
    honest_blocks = 0
    blocks_v0_fabricated = 0

    for tid, fp in latest.items():
        events = [json.loads(line) for line in open(fp)]
        if not events:
            continue
        version = (events[0].get("fields") or {}).get("agent_version")
        if marker != "any" and version != marker:
            continue
        ground = gt.get(tid)
        if not ground:
            continue
        # find answer.emitted
        answer = ""
        for e in events:
            if e.get("type") == "answer.emitted":
                answer = e.get("fields", {}).get("answer", "") or ""
        # find blocked, finish_reason
        blocked = None
        fr = None
        for e in events:
            if e.get("type") == "agent.blocked":
                blocked = e.get("fields", {}).get("reason")
            if e.get("type") == "llm.responded":
                fr = e.get("fields", {}).get("finish_reason")
        try:
            sc = bool(question_scorer(answer, ground["final_answer"] if "final_answer" in ground else ground["ground_truth"]))
        except Exception:
            sc = False
        lv = ground.get("level")
        total += 1
        by_level_total[lv] += 1
        if sc:
            correct += 1
            by_level_correct[lv] += 1
        else:
            if fr == "length":
                fail["truncated"] += 1
            elif blocked:
                fail[f"blocked_{blocked}"] += 1
            else:
                fail["wrong_answer"] += 1
        if blocked:
            honest_blocks += 1
            v0_ans = (gt[tid].get("answer") or "") if "answer" in gt[tid] else ""
            if v0_ans.strip():
                blocks_v0_fabricated += 1

    print(f"version={marker}  score={correct}/{total} = {correct/max(total,1):.3f}")
    for lv in ("1", "2", "3"):
        print(f"  L{lv}: {by_level_correct[lv]}/{by_level_total[lv]}")
    print("failures:", dict(fail))
    print(f"honest_blocks: {honest_blocks}  (v0 had fabricated answers for {blocks_v0_fabricated} of these)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v1")
