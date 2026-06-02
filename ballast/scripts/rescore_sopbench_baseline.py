"""Rescore an existing SOP-Bench baseline results.json without re-running LLM calls.

Useful when the summarization or frontier-update logic was buggy on the first run
and you just want to recompute summary + frontier + evolution_summary entry from
the already-stored per-task results.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[2]))

from ballast.scripts.run_sopbench_baseline import (  # noqa: E402
    _is_correct, _summarize, _update_frontier, logs_dir_for,
)


def rescore(domain: str, agent_name: str, iteration: int) -> None:
    logs_dir = logs_dir_for(domain)
    results_path = logs_dir / f"{agent_name}__train__results.json"
    if not results_path.exists():
        raise SystemExit(f"no results file: {results_path}")
    data = json.loads(results_path.read_text())
    task_results = data.get("results") or []
    subset_ids = [str(t) for t in data.get("subset_ids", [])]

    summary = _summarize(task_results)
    data["summary"] = summary
    results_path.write_text(json.dumps(data, indent=2, default=str))
    print(f"[{domain}] resummarized -> n={summary['n_tasks']} "
          f"correct={summary['n_correct']} completed={summary['n_completed']} "
          f"tsr={summary['tsr']} ecr={summary['ecr']}")

    frontier_path = logs_dir / "frontier_val.json"
    frontier_after = _update_frontier(frontier_path, agent_name, task_results, subset_ids)
    print(f"[{domain}] frontier after: "
          f"{frontier_after['correct']}/{frontier_after['total']} = {frontier_after['acc']}")

    evo_path = logs_dir / "evolution_summary.jsonl"
    if evo_path.exists():
        lines = [ln for ln in evo_path.read_text().splitlines() if ln.strip()]
        if lines:
            last = json.loads(lines[-1])
            if last.get("iteration") == iteration and last.get("agent") == agent_name:
                last["train_score"] = summary
                last["frontier_after"] = frontier_after
                last["rescored_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
                lines[-1] = json.dumps(last, default=str)
                evo_path.write_text("\n".join(lines) + "\n")
                print(f"[{domain}] updated last evolution_summary row")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True)
    p.add_argument("--agent-name", default="v0")
    p.add_argument("--iteration", type=int, default=0)
    args = p.parse_args()
    rescore(args.domain, args.agent_name, args.iteration)


if __name__ == "__main__":
    main()
