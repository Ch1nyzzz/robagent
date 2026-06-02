"""Split GAIA validation (165) into 30 train + 135 test.

Stratified by level. Deterministic via seed. Writes:
  ballast/train_task_ids.txt
  ballast/test_task_ids.txt
  ballast/logs/split_meta.json
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "traces" / "gaia__summary.jsonl"
OUT_DIR = ROOT / "ballast"
SEED = 7
TRAIN_PER_LEVEL = {"1": 9, "2": 15, "3": 6}  # ~ matches L1:53 L2:86 L3:26 (≈30 total)


def main() -> None:
    by_level: dict[str, list[dict]] = defaultdict(list)
    with SUMMARY.open() as f:
        for line in f:
            r = json.loads(line)
            by_level[str(r["level"])].append(r)

    rng = random.Random(SEED)
    train, test = [], []
    for lvl, rows in sorted(by_level.items()):
        rng.shuffle(rows)
        k = TRAIN_PER_LEVEL[lvl]
        train.extend(rows[:k])
        test.extend(rows[k:])

    rng.shuffle(train)
    rng.shuffle(test)

    (OUT_DIR / "train_task_ids.txt").write_text("\n".join(r["task_id"] for r in train) + "\n")
    (OUT_DIR / "test_task_ids.txt").write_text("\n".join(r["task_id"] for r in test) + "\n")

    v0_train_correct = sum(1 for r in train if r.get("score", 0) > 0)
    v0_test_correct = sum(1 for r in test if r.get("score", 0) > 0)

    meta = {
        "seed": SEED,
        "n_train": len(train),
        "n_test": len(test),
        "train_per_level": TRAIN_PER_LEVEL,
        "v0_baseline_on_train": {
            "correct": v0_train_correct,
            "total": len(train),
            "acc": round(v0_train_correct / len(train), 4),
        },
        "v0_baseline_on_test": {
            "correct": v0_test_correct,
            "total": len(test),
            "acc": round(v0_test_correct / len(test), 4),
        },
        "level_counts": {lvl: len(rows) for lvl, rows in by_level.items()},
    }
    (OUT_DIR / "logs" / "split_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
