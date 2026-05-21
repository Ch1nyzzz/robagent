"""Finalize iter 10: print scoreboard, claim-graph stats, and emit a
markdown-formatted block for FINAL_REPORT_v10.md.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _v10_summary() -> tuple[int, int, dict]:
    p = ROOT / "traces" / "gaia_v10__summary.jsonl"
    total = 0
    correct = 0
    by_level = {"1": [0, 0], "2": [0, 0], "3": [0, 0]}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        total += 1
        s = int(rec.get("score", 0))
        correct += s
        lv = str(rec.get("level"))
        if lv in by_level:
            by_level[lv][1] += 1
            by_level[lv][0] += s
    return correct, total, by_level


def _v0_summary() -> tuple[int, int]:
    p = ROOT / "traces" / "gaia__summary.jsonl"
    total = correct = 0
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        total += 1
        correct += int(rec.get("score", 0))
    return correct, total


def main():
    c10, n10, levels = _v10_summary()
    c0, n0 = _v0_summary()
    print(f"v0  : {c0}/{n0} = {c0/max(n0,1):.3f}")
    print(f"v10 : {c10}/{n10} = {c10/max(n10,1):.3f}")
    print(f"delta: {c10-c0} (vs v0 baseline)")
    for lv in ("1", "2", "3"):
        print(f"  L{lv}: v10 {levels[lv][0]}/{levels[lv][1]}")

    print("\n--- claim graph stats ---")
    subprocess.run(
        [sys.executable, str(ROOT / "harness/gaia/_claim_graph_stats.py"), "v10"],
        check=False,
    )

    print("\n--- v0 vs v10 comparison ---")
    subprocess.run(
        [sys.executable, str(ROOT / "harness/gaia/_compare_v0_v10.py")],
        check=False,
    )


if __name__ == "__main__":
    main()
