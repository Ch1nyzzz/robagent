"""Extract the validation-sample subset score from a versioned summary."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(version: str = "v10") -> None:
    sample_ids = set(
        (ROOT / "harness/gaia/_validation_sample_n1.txt")
        .read_text()
        .strip()
        .splitlines()
    )
    summary = ROOT / "traces" / f"gaia_{version}__summary.jsonl"
    if not summary.exists():
        print(f"no summary at {summary}")
        return
    total = correct = 0
    seen: set[str] = set()
    for line in summary.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        tid = rec["task_id"]
        if tid in sample_ids:
            seen.add(tid)
            total += 1
            correct += int(rec.get("score", 0))
    print(f"{version} sample subset: {correct}/{total} (of {len(sample_ids)} sample tasks; {len(sample_ids)-len(seen)} not yet run)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v10")
