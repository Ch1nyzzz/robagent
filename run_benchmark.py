from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

import importlib

from bench.gaia.loader import build_prompt as gaia_prompt, iter_tasks as gaia_tasks
from bench.gaia.scorer import question_scorer as gaia_score
from bench.tau2bench.loader import build_prompt as tau2_prompt, iter_tasks as tau2_tasks
from bench.tau2bench.scorer import score_baseline as tau2_score


def _resolve_run_task(agent_version: str):
    """Resolve `run_task` from `agent.base` (v0), `agent.v{N}.base`,
    `agent.<name>.base` (meta-harness), or `agent.component_runtime.base`
    (the fixed graph runtime; active component set from workflow YAML)."""
    if agent_version in ("0", "v0", "base", None, ""):
        mod = importlib.import_module("agent.base")
    elif agent_version == "component_runtime":
        mod = importlib.import_module("agent.component_runtime.base")
    elif agent_version.startswith("mh"):
        mod = importlib.import_module(f"agent.{agent_version}.base")
    else:
        v = agent_version if agent_version.startswith("v") else f"v{agent_version}"
        mod = importlib.import_module(f"agent.{v}.base")
    return mod.run_task


RESULTS_DIR = Path(__file__).resolve().parent / "traces"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PARALLEL = 16


def _summary_path(benchmark: str) -> Path:
    # The evolution loop overrides this via BENCHMARK_SUMMARY_PATH so each iter's
    # summary file is independent (mirrors sopbench's per-iter layout). Mirrors
    # the env-override convention `events.py:traces_dir()` already uses for
    # per-task trace files via TRACES_DIR.
    override = os.environ.get("BENCHMARK_SUMMARY_PATH")
    if override:
        p = Path(override)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    return RESULTS_DIR / f"{benchmark}__summary.jsonl"


def _execute_pool(
    tasks: Iterable[dict],
    worker: Callable[[dict], dict],
    parallel: int,
    limit: int | None,
    log_prefix: str,
    summary_path: Path,
) -> None:
    materialized: list[dict] = []
    for i, t in enumerate(tasks):
        if limit is not None and i >= limit:
            break
        materialized.append(t)
    n = len(materialized)
    started = time.time()
    out_lock = threading.Lock()
    out_fh = summary_path.open("w", encoding="utf-8")
    print(f"[{log_prefix}] writing summary to {summary_path}  (tasks={n}  parallel={parallel})")
    completed = 0

    def _wrapped(task: dict) -> dict:
        try:
            return worker(task)
        except Exception as e:
            return {"task_id": task.get("task_id"), "error": repr(e), "score": 0}

    try:
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            futures = {ex.submit(_wrapped, t): t for t in materialized}
            for fut in as_completed(futures):
                rec = fut.result()
                with out_lock:
                    out_fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                    out_fh.flush()
                completed += 1
                elapsed = time.time() - started
                tag = rec.get("tag", "")
                print(
                    f"[{log_prefix}] {completed:>4}/{n}  "
                    f"score={rec.get('score'):.3f}  elapsed={elapsed:.1f}s  "
                    f"{tag}  {rec.get('task_id')}",
                    flush=True,
                )
    finally:
        out_fh.close()
    print(f"[{log_prefix}] DONE total={completed}")


def run_gaia(
    parallel: int = DEFAULT_PARALLEL,
    limit: int | None = None,
    levels: list[int] | None = None,
    agent_version: str = "v0",
    task_ids: set[str] | None = None,
) -> None:
    correct = [0]
    total = [0]
    counters_lock = threading.Lock()
    run_task = _resolve_run_task(agent_version)

    def worker(task: dict) -> dict:
        prompt = gaia_prompt(task)
        result = run_task(
            benchmark="gaia",
            task_id=task["task_id"],
            task_prompt=prompt,
            extras={"level": task["level"], "file_name": task["file_name"]},
        )
        answer = result["answer"]
        score = False
        log = result.get("log")
        if answer is not None:
            try:
                score = bool(gaia_score(answer, task["final_answer"]))
            except Exception as e:
                if log is not None:
                    log.emit("eval.error", parent=result["root_event_id"], error=repr(e))
        if log is not None:
            log.emit(
                "eval.scored",
                parent=result["root_event_id"],
                score=int(score),
                ground_truth=task["final_answer"],
            )
            log.emit("run.ended", parent=result["root_event_id"])
            log.close()
        with counters_lock:
            total[0] += 1
            correct[0] += int(score)
            acc = correct[0] / max(total[0], 1)
        return {
            "task_id": task["task_id"],
            "level": task["level"],
            "score": float(int(score)),
            "answer": answer,
            "ground_truth": task["final_answer"],
            "trace_path": result["trace_path"],
            "error": result.get("error"),
            "tag": f"L{task['level']} acc={correct[0]}/{total[0]}={acc:.3f}",
        }

    tasks_iter = gaia_tasks(split="validation", levels=levels)
    if task_ids is not None:
        tasks_iter = (t for t in tasks_iter if t["task_id"] in task_ids)

    summary_name = "gaia" if agent_version in ("v0", "0", "base") else f"gaia_{agent_version}"
    _execute_pool(
        tasks=tasks_iter,
        worker=worker,
        parallel=parallel,
        limit=limit,
        log_prefix=f"gaia[{agent_version}]",
        summary_path=_summary_path(summary_name),
    )
    print(f"[gaia:{agent_version}] final acc = {correct[0]}/{total[0]} = {correct[0] / max(total[0], 1):.3f}")


def run_tau2(
    parallel: int = DEFAULT_PARALLEL,
    limit: int | None = None,
    domain: str | None = None,
    domain_limits: dict[str, int] | None = None,
) -> None:
    pos = [0]
    total = [0]
    counters_lock = threading.Lock()

    def worker(task: dict) -> dict:
        prompt = tau2_prompt(task)
        result = run_task(
            benchmark="tau2bench",
            task_id=task["task_id"],
            task_prompt=prompt,
            extras={"domain": task["domain"]},
        )
        answer = result["answer"]
        score_d = tau2_score(task, answer or "")
        log = result.get("log")
        if log is not None:
            log.emit("eval.scored", parent=result["root_event_id"], **score_d)
            log.emit("run.ended", parent=result["root_event_id"])
            log.close()
        with counters_lock:
            total[0] += 1
            pos[0] += int(score_d["score"] > 0)
        return {
            "task_id": task["task_id"],
            "domain": task["domain"],
            "score": float(score_d["score"]),
            "answer": answer,
            "trace_path": result["trace_path"],
            "error": result.get("error"),
            "tag": f"{task['domain']} any>0={pos[0]}/{total[0]}",
            **{k: v for k, v in score_d.items() if k != "score"},
        }

    _execute_pool(
        tasks=tau2_tasks(domain=domain, domain_limits=domain_limits),
        worker=worker,
        parallel=parallel,
        limit=limit,
        log_prefix="tau2",
        summary_path=_summary_path("tau2bench"),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("benchmark", choices=["gaia", "tau2bench", "all"])
    p.add_argument("--limit", type=int, default=None, help="run only first N tasks (smoke test)")
    p.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL, help="concurrent LLM calls")
    p.add_argument("--levels", type=int, nargs="*", default=None, help="GAIA levels filter")
    p.add_argument("--agent-version", default="v0", help="agent version: v0 | v1 | v2 ...")
    p.add_argument(
        "--task-ids-file",
        default=None,
        help="optional path to a newline-delimited file of task_ids to restrict the run",
    )
    p.add_argument("--domain", default=None, help="tau2 domain filter")
    p.add_argument(
        "--tau2-domain-cap",
        action="append",
        default=[],
        help="cap a tau2 domain to N tasks (sampled), repeatable: --tau2-domain-cap telecom=70",
    )
    args = p.parse_args()

    domain_limits: dict[str, int] = {}
    for spec in args.tau2_domain_cap:
        k, _, v = spec.partition("=")
        domain_limits[k.strip()] = int(v)

    task_ids: set[str] | None = None
    if args.task_ids_file:
        with open(args.task_ids_file) as fh:
            task_ids = {ln.strip() for ln in fh if ln.strip()}

    if args.benchmark in ("gaia", "all"):
        run_gaia(
            parallel=args.parallel,
            limit=args.limit,
            levels=args.levels,
            agent_version=args.agent_version,
            task_ids=task_ids,
        )
    if args.benchmark in ("tau2bench", "all"):
        run_tau2(
            parallel=args.parallel,
            limit=args.limit,
            domain=args.domain,
            domain_limits=domain_limits or None,
        )


if __name__ == "__main__":
    main()
