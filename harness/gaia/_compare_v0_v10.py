"""Compare v0 baseline summary against v10 summary by task_id.

Useful for the FINAL_REPORT comparison table:
- how many tasks did v10 win where v0 lost?
- how many did v10 lose where v0 won?
- routing distribution and blocked-honestly counts
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(p: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    fp = ROOT / "traces" / p
    if not fp.exists():
        return out
    for line in fp.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[rec["task_id"]] = rec
    return out


def main() -> None:
    v0 = _load("gaia__summary.jsonl")
    v10 = _load("gaia_v10__summary.jsonl")

    common = set(v0) & set(v10)
    only_v0 = set(v0) - common
    only_v10 = set(v10) - common
    print(f"v0 tasks: {len(v0)}, v10 tasks: {len(v10)}, common: {len(common)}")
    if only_v0:
        print(f"v0-only: {len(only_v0)}")
    if only_v10:
        print(f"v10-only: {len(only_v10)}")

    v0c = sum(int(v0[t].get("score", 0)) for t in v0)
    v10c = sum(int(v10[t].get("score", 0)) for t in v10)
    print(f"v0 score:  {v0c}/{len(v0)} = {v0c/max(len(v0),1):.3f}")
    print(f"v10 score: {v10c}/{len(v10)} = {v10c/max(len(v10),1):.3f}")

    # Win/loss matrix on common subset
    won_by_v10 = 0
    lost_by_v10 = 0
    both_right = 0
    both_wrong = 0
    for tid in common:
        s0 = int(v0[tid].get("score", 0))
        s10 = int(v10[tid].get("score", 0))
        if s0 == 1 and s10 == 1:
            both_right += 1
        elif s0 == 0 and s10 == 0:
            both_wrong += 1
        elif s10 == 1 and s0 == 0:
            won_by_v10 += 1
        else:
            lost_by_v10 += 1
    print(f"\nWin/loss on common-{len(common)}:")
    print(f"  both correct:   {both_right}")
    print(f"  v10 wins:       {won_by_v10}")
    print(f"  v10 loses:      {lost_by_v10}")
    print(f"  both wrong:     {both_wrong}")

    # Level breakdown
    for lv in ("1", "2", "3"):
        c0 = sum(1 for t, r in v0.items() if str(r.get("level")) == lv and int(r.get("score", 0)) == 1)
        c10 = sum(1 for t, r in v10.items() if str(r.get("level")) == lv and int(r.get("score", 0)) == 1)
        n0 = sum(1 for t, r in v0.items() if str(r.get("level")) == lv)
        n10 = sum(1 for t, r in v10.items() if str(r.get("level")) == lv)
        print(f"  L{lv}: v0 {c0}/{n0}  v10 {c10}/{n10}")


if __name__ == "__main__":
    main()
