"""Recompute baseline pass-rate over a (possibly re-cut) split, without re-running.

Looks up each task_id from the split file in a pool of already-recorded result
JSONs (from prior `enterpriseops_smoke.py` runs). Useful when you reshuffle a
train/test split and don't want to spend API budget re-evaluating tasks whose
results are already on disk.

Run
---
    python meta_harness/scripts/recompute_eog_baseline.py \\
        --split-file meta_harness/enterpriseops_calendar_train_task_ids.txt \\
        --results-glob 'meta_harness/logs_components_enterpriseops_calendar/baseline/*/{train,test}/results/run_1/results_*.json'

The script aggregates over ALL matched runs of the latest available result per
task_id (most-recent file wins on duplicate). It re-prints the same summary
block that the smoke script does.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from statistics import mean, median


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split-file", required=True,
                   help="txt file with one task_id per line")
    p.add_argument("--results-glob", required=True,
                   help="glob pattern for upstream results_*.json files. "
                        "Quote it so the shell doesn't expand. "
                        "Brace expansion ({a,b}) is supported.")
    p.add_argument("--label", default=None,
                   help="Label printed in the summary header.")
    return p.parse_args()


def _expand_braces(pat: str) -> list[str]:
    """Tiny brace-expansion: {a,b}/{c,d} → 4 combos. One brace group max here."""
    out = [pat]
    while any("{" in p and "}" in p for p in out):
        next_out = []
        for p in out:
            i = p.find("{")
            j = p.find("}", i)
            if i < 0 or j < 0:
                next_out.append(p)
                continue
            head, body, tail = p[:i], p[i + 1:j], p[j + 1:]
            for opt in body.split(","):
                next_out.append(head + opt + tail)
        out = next_out
    return out


def load_results(pattern: str) -> dict[str, dict]:
    """Map task_id -> result dict (newest file wins)."""
    matches: list[str] = []
    for p in _expand_braces(pattern):
        matches.extend(glob.glob(p))
    if not matches:
        sys.exit(f"no JSONs matched pattern: {pattern}")
    matches.sort(key=lambda p: Path(p).stat().st_mtime)  # oldest first
    by_id: dict[str, dict] = {}
    for path in matches:
        try:
            data = json.loads(Path(path).read_text())
        except json.JSONDecodeError:
            continue
        # task_id encoded in filename: results_<mode>__<domain>__<task_id>.json
        stem = Path(path).stem.removeprefix("results_")
        parts = stem.split("__", 2)
        task_id = parts[2] if len(parts) == 3 else stem
        runs = data.get("runs") or []
        first = runs[0] if runs else {}
        vs = first.get("verification_summary") or {}
        by_id[task_id] = {  # later writes overwrite earlier — newest wins
            "task_id": task_id,
            "src_file": path,
            "passed": bool(first.get("overall_success")),
            "error": first.get("error"),
            "wall_ms": first.get("execution_time_ms"),
            "n_tools_called": len(first.get("tools_used") or []),
            "verifier_pass_rate": vs.get("pass_rate"),
            "verifier_total": vs.get("total"),
            "verifier_passed": vs.get("passed"),
            "n_steps_proxy": len(first.get("conversation_flow") or []),
        }
    return by_id


def main() -> None:
    args = parse_args()
    wanted = [ln.strip() for ln in Path(args.split_file).read_text().splitlines()
              if ln.strip()]
    pool = load_results(args.results_glob)

    summaries = []
    missing = []
    for tid in wanted:
        if tid in pool:
            summaries.append(pool[tid])
        else:
            missing.append(tid)

    n = len(summaries)
    label = args.label or Path(args.split_file).stem
    print("=" * 72)
    print(f" RECOMPUTE — {label}  ({n} tasks matched, {len(missing)} missing)")
    print("=" * 72)
    if missing:
        print(f" MISSING (no result on disk for these task_ids):")
        for tid in missing[:10]:
            print(f"   - {tid}")
        if len(missing) > 10:
            print(f"   ... +{len(missing) - 10} more")
        print()

    if n == 0:
        return

    walls = [s["wall_ms"] for s in summaries if s.get("wall_ms")]
    steps = [s["n_steps_proxy"] for s in summaries if s.get("n_steps_proxy")]
    tools = [s["n_tools_called"] for s in summaries if s.get("n_tools_called") is not None]
    vrates = [s["verifier_pass_rate"] for s in summaries
              if s.get("verifier_pass_rate") is not None]
    n_pass = sum(1 for s in summaries if s.get("passed"))
    n_err = sum(1 for s in summaries if s.get("error"))

    print(f" tasks passed (all verif) : {n_pass}/{n}  ({100 * n_pass / n:.1f}%)")
    print(f" tasks with hard error    : {n_err}")
    if vrates:
        print(f" verifier pass rate (mean): {mean(vrates):.3f}  "
              f"(median {median(vrates):.3f})")
    if walls:
        print(f" wall per task ms         : "
              f"median {median(walls):.0f}  mean {mean(walls):.0f}  "
              f"max {max(walls)}")
    if steps:
        print(f" conv steps (proxy)       : "
              f"median {median(steps):.0f}  max {max(steps)}")
    if tools:
        print(f" tools / task             : "
              f"median {median(tools):.0f}  max {max(tools)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
