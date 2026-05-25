"""Build train/test split for EnterpriseOps-Gym <domain> oracle mode.

Mirrors the convention used for GAIA / tau2 / toolathlon / sopbench:
  - small ``train`` set used as the proposer's evolution frontier
  - larger ``test`` set held back as the unbiased baseline / validation pool

Determinism: ``random.Random(SEED).shuffle()`` over task_ids sorted ascending.
Re-running with the same SEED + same HF revision produces identical splits.

Run
---
    set -a && source /data/home/yuhan/robagent/.env && set +a && \\
    python meta_harness/scripts/select_enterpriseops_split.py --domain calendar

Output
------
    meta_harness/enterpriseops_<domain>_train_task_ids.txt
    meta_harness/enterpriseops_<domain>_test_task_ids.txt
    meta_harness/logs/enterpriseops_<domain>_split_meta.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "meta_harness"
META_DIR = OUT_DIR / "logs"

# HF release ships the ~60% "public split" of the original dataset. Sizes per
# domain are therefore smaller than the paper's full numbers. We default n_test
# to "all remaining tasks after n_train" so the script stays robust if the HF
# split is re-cut upstream.
DOMAINS = ["calendar", "csm", "drive", "email", "hr", "itsm", "teams", "hybrid"]
DEFAULT_N_TRAIN = 30


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True, choices=DOMAINS)
    p.add_argument("--mode", default="oracle",
                   help="HF dataset config (tool-mode); split is independent of mode")
    p.add_argument("--hf-dataset", default="ServiceNow-AI/EnterpriseOps-Gym")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN,
                   help=f"Train set size (default {DEFAULT_N_TRAIN}).")
    p.add_argument("--n-test", type=int, default=None,
                   help="Test set size; defaults to (total - n_train).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import load_dataset

    ds = load_dataset(args.hf_dataset, args.mode, split=args.domain)
    task_ids = sorted(str(row["task_id"]) for row in ds)
    total = len(task_ids)

    n_train = args.n_train
    n_test = args.n_test if args.n_test is not None else (total - n_train)

    if n_train + n_test > total:
        raise SystemExit(
            f"requested train+test={n_train + n_test} exceeds available {total} "
            f"task(s) in domain '{args.domain}'"
        )

    rng = random.Random(args.seed)
    shuffled = list(task_ids)
    rng.shuffle(shuffled)
    train = shuffled[:n_train]
    test = shuffled[n_train:n_train + n_test]
    holdout = shuffled[n_train + n_test:]

    train_path = OUT_DIR / f"enterpriseops_{args.domain}_train_task_ids.txt"
    test_path = OUT_DIR / f"enterpriseops_{args.domain}_test_task_ids.txt"
    train_path.write_text("\n".join(train) + "\n")
    test_path.write_text("\n".join(test) + "\n")

    META_DIR.mkdir(exist_ok=True)
    meta = {
        "domain": args.domain,
        "mode": args.mode,
        "hf_dataset": args.hf_dataset,
        "seed": args.seed,
        "total_tasks": total,
        "n_train": len(train),
        "n_test": len(test),
        "n_holdout": len(holdout),
        "train_file": str(train_path.relative_to(ROOT)),
        "test_file": str(test_path.relative_to(ROOT)),
        "first_train": train[:3],
        "first_test": test[:3],
        "first_holdout": holdout[:3],
    }
    (META_DIR / f"enterpriseops_{args.domain}_split_meta.json").write_text(
        json.dumps(meta, indent=2)
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
