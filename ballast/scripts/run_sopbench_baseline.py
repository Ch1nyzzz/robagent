"""Run a SopBenchAgent baseline on one domain's train + test subsets.

Per-domain split: first 30 rows = train, rest = test (matches the spirit of
the GAIA 30-task train pool). For each subset this writes:

  ballast/logs_components_sopbench_<domain>/
      <agent_name>__<subset>__results.json    # task-by-task scores + summary
      frontier_val.json                       # per-task best (frontier)
      evolution_summary.jsonl                 # iter 0 row for v0; appended for later iters

Usage:
  python ballast/scripts/run_sopbench_baseline.py \
      --domain dangerous_goods --agent-name v0 --iteration 0 \
      --train-size 30 --max-workers 16
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amazon_sop_bench.benchmarks import load_benchmark  # noqa: E402
from amazon_sop_bench.evaluation.evaluator import Evaluator  # noqa: E402

from agent.sopbench_agent import SopBenchAgent  # noqa: E402

LOGS_ROOT = ROOT / "ballast"


def logs_dir_for(domain: str) -> Path:
    d = LOGS_ROOT / f"logs_components_sopbench_{domain}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _task_result_to_dict(tr: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(tr):
        return dataclasses.asdict(tr)
    if hasattr(tr, "__dict__"):
        return {k: v for k, v in tr.__dict__.items() if not k.startswith("_")}
    return {"raw": str(tr)}


def _is_correct(r: dict[str, Any]) -> bool:
    """A task is correct iff the evaluator left success=True. The evaluator
    re-writes success to False when the parsed output doesn't match expected
    (see run_single_task in amazon_sop_bench), so success here means
    "agent finished AND output matched ground truth"."""
    return bool(r.get("success"))


def _crashed(r: dict[str, Any]) -> bool:
    """The agent failed to produce any output (vs. produced a wrong output)."""
    err = (r.get("error") or "")
    return r.get("success") is False and not str(err).startswith("Output mismatch")


def _summarize(task_results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(task_results)
    correct = sum(1 for r in task_results if _is_correct(r))
    completed = sum(1 for r in task_results if not _crashed(r))
    return {
        "n_tasks": n,
        "n_completed": completed,
        "n_correct": correct,
        "tsr": round(correct / max(n, 1), 4),
        "ecr": round(completed / max(n, 1), 4),
    }


def _update_frontier(
    frontier_path: Path,
    agent_name: str,
    task_results: list[dict[str, Any]],
    subset_ids: list[str],
) -> dict[str, Any]:
    frontier: dict[str, Any] = {}
    if frontier_path.exists():
        frontier = json.loads(frontier_path.read_text())
    per_task = frontier.setdefault("per_task", {})
    for tr in task_results:
        tid = str(tr.get("task_id"))
        if tid not in subset_ids:
            continue
        score = 1.0 if _is_correct(tr) else 0.0
        cur = per_task.get(tid)
        if cur is None or score > cur.get("score", 0):
            per_task[tid] = {
                "agent": agent_name,
                "score": score,
                "predicted": tr.get("predicted_output"),
            }
    frontier_correct = sum(1 for v in per_task.values() if v.get("score", 0) > 0)
    frontier["frontier_score"] = {
        "correct": frontier_correct,
        "total": len(subset_ids),
        "acc": round(frontier_correct / max(len(subset_ids), 1), 4),
    }
    frontier["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    frontier_path.write_text(json.dumps(frontier, indent=2, default=str))
    return frontier["frontier_score"]


def run_subset(
    domain: str,
    subset_name: str,
    subset_indices: list[int],
    agent: SopBenchAgent,
    agent_name: str,
    max_workers: int,
    logs_dir: Path,
) -> dict[str, Any]:
    """Run `agent` on the given row indices of `domain`'s test set."""
    print(f"[{domain}/{subset_name}] loading benchmark...")
    benchmark = load_benchmark(domain)
    total = len(benchmark.tasks)
    subset_indices = [i for i in subset_indices if 0 <= i < total]
    benchmark.tasks = [benchmark.tasks[i] for i in subset_indices]
    subset_ids = [t.task_id for t in benchmark.tasks]
    print(f"[{domain}/{subset_name}] {len(benchmark.tasks)}/{total} tasks selected; workers={max_workers}")

    evaluator = Evaluator.__new__(Evaluator)  # bypass __init__ which re-loads
    evaluator.benchmark_name = domain
    evaluator.agent = agent
    evaluator.max_tasks = None
    evaluator.resume = False
    evaluator.max_workers = max_workers
    from amazon_sop_bench.config import get_config
    from amazon_sop_bench.evaluation.metrics import MetricsCalculator
    from amazon_sop_bench.evaluation.parser import OutputParser
    from amazon_sop_bench.evaluation.reporter import ResultReporter
    evaluator.config = get_config()
    evaluator.output_dir = logs_dir / f"{agent_name}__{subset_name}__traces"
    evaluator.output_dir.mkdir(parents=True, exist_ok=True)
    evaluator.parser = OutputParser()
    evaluator.metrics_calculator = MetricsCalculator()
    evaluator.reporter = ResultReporter()
    evaluator.existing_results = {}
    evaluator.benchmark = benchmark

    t0 = time.time()
    results = evaluator.run()
    elapsed = time.time() - t0
    task_dicts = [_task_result_to_dict(tr) for tr in results.task_results]
    summary = _summarize(task_dicts)
    out = {
        "domain": domain,
        "subset": subset_name,
        "agent": agent_name,
        "subset_ids": subset_ids,
        "summary": summary,
        "results": task_dicts,
        "wall_seconds": round(elapsed, 1),
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "tsr": results.task_success_rate,
        "ecr": results.execution_completion_rate,
        "tool_accuracy": results.tool_accuracy,
    }
    out_path = logs_dir / f"{agent_name}__{subset_name}__results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(
        f"[{domain}/{subset_name}] done in {elapsed:.1f}s   "
        f"TSR={results.task_success_rate:.1%}  "
        f"ECR={results.execution_completion_rate:.1%}  "
        f"-> {out_path}"
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True)
    p.add_argument("--agent-name", default="v0")
    p.add_argument("--iteration", type=int, default=0)
    p.add_argument("--train-size", type=int, default=30)
    p.add_argument("--max-workers", type=int, default=16)
    p.add_argument("--max-iterations", type=int, default=15,
                   help="max agent turns per task")
    p.add_argument("--subsets", default="train,test",
                   help="comma-separated subsets to run: train, test, or both")
    p.add_argument("--hypothesis", default="single function-calling LLM (v0 baseline)")
    p.add_argument("--changes", default="")
    p.add_argument("--plugin-json", default="",
                   help="JSON {component, workflow_patch} from a proposer; "
                        "stored verbatim into evolution_summary for the audit trail. "
                        "Empty for the v0 baseline.")
    args = p.parse_args()

    logs_dir = logs_dir_for(args.domain)
    benchmark = load_benchmark(args.domain)
    total = len(benchmark.tasks)
    train_idx = list(range(min(args.train_size, total)))
    test_idx = list(range(args.train_size, total))
    print(f"[{args.domain}] total={total}  train={len(train_idx)}  test={len(test_idx)}")

    agent = SopBenchAgent(max_iterations=args.max_iterations)

    subsets_to_run = {s.strip() for s in args.subsets.split(",") if s.strip()}
    outputs: dict[str, dict[str, Any]] = {}
    if "train" in subsets_to_run:
        outputs["train"] = run_subset(
            args.domain, "train", train_idx, agent, args.agent_name, args.max_workers, logs_dir
        )
    if "test" in subsets_to_run:
        outputs["test"] = run_subset(
            args.domain, "test", test_idx, agent, args.agent_name, args.max_workers, logs_dir
        )

    # Update frontier + evolution_summary using the TRAIN result (the gate signal).
    if "train" in outputs:
        train_out = outputs["train"]
        frontier_path = logs_dir / "frontier_val.json"
        train_ids_str = [str(t) for t in train_idx]
        frontier_after = _update_frontier(
            frontier_path, args.agent_name, train_out["results"], train_ids_str
        )
        print(f"[{args.domain}] frontier after train: "
              f"{frontier_after['correct']}/{frontier_after['total']} = {frontier_after['acc']}")

        evo_path = logs_dir / "evolution_summary.jsonl"
        entry = {
            "iteration": args.iteration,
            "agent": args.agent_name,
            "hypothesis": args.hypothesis,
            "changes": args.changes,
            "train_score": train_out["summary"],
            "test_score": outputs.get("test", {}).get("summary"),
            "frontier_after": frontier_after,
            "wall_seconds_train": train_out["wall_seconds"],
            "wall_seconds_test": outputs.get("test", {}).get("wall_seconds"),
            "scored_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "accepted": True if args.iteration == 0 else None,
        }
        if args.plugin_json:
            try:
                entry["plugin"] = json.loads(args.plugin_json)
            except json.JSONDecodeError as e:
                print(f"[warn] --plugin-json not valid JSON, dropped: {e}")
        with evo_path.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        print(f"[{args.domain}] appended evolution_summary row -> {evo_path}")


if __name__ == "__main__":
    main()
