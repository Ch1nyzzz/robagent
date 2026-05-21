from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from agent.base import run_task
from bench.gaia.loader import build_prompt as gaia_prompt, iter_tasks as gaia_tasks
from bench.gaia.scorer import question_scorer as gaia_score
from bench.tau2bench.loader import build_prompt as tau2_prompt, iter_tasks as tau2_tasks
from bench.tau2bench.scorer import score_baseline as tau2_score


ROOT = Path(__file__).resolve().parent
TRACES = ROOT / "traces"


def _load_summary(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _failed_ids(rows: list[dict]) -> set[str]:
    return {r["task_id"] for r in rows if r.get("error") or not r.get("answer")}


def _save_summary(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def _gaia_worker(task: dict) -> dict:
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
        except Exception:
            score = False
    if log is not None:
        log.emit("eval.scored", parent=result["root_event_id"], score=int(score), ground_truth=task["final_answer"])
        log.emit("run.ended", parent=result["root_event_id"])
        log.close()
    return {
        "task_id": task["task_id"],
        "level": task["level"],
        "score": float(int(score)),
        "answer": answer,
        "ground_truth": task["final_answer"],
        "trace_path": result["trace_path"],
        "error": result.get("error"),
    }


def _tau2_worker(task: dict) -> dict:
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
    return {
        "task_id": task["task_id"],
        "domain": task["domain"],
        "score": float(score_d["score"]),
        "answer": answer,
        "trace_path": result["trace_path"],
        "error": result.get("error"),
        **{k: v for k, v in score_d.items() if k != "score"},
    }


def _retry(rows: list[dict], task_iter, worker: Callable[[dict], dict], parallel: int, tag: str) -> list[dict]:
    failed = _failed_ids(rows)
    print(f"[{tag}] failed to retry: {len(failed)}")
    if not failed:
        return rows
    todo = [t for t in task_iter if t["task_id"] in failed]
    print(f"[{tag}] resolved {len(todo)} task defs (expected {len(failed)})")
    by_id: dict[str, dict] = {r["task_id"]: r for r in rows}
    lock = threading.Lock()
    done = [0]
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = {ex.submit(worker, t): t for t in todo}
        for fut in as_completed(futures):
            new = fut.result()
            with lock:
                by_id[new["task_id"]] = new
                done[0] += 1
                print(f"[{tag}] {done[0]}/{len(todo)} task={new['task_id']} score={new['score']} err={bool(new.get('error'))}", flush=True)
    return list(by_id.values())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("benchmark", choices=["gaia", "tau2bench", "all"])
    p.add_argument("--parallel", type=int, default=8)
    args = p.parse_args()

    if args.benchmark in ("gaia", "all"):
        path = TRACES / "gaia__summary.jsonl"
        rows = _load_summary(path)
        new_rows = _retry(rows, gaia_tasks(split="validation"), _gaia_worker, args.parallel, "gaia")
        _save_summary(path, new_rows)
        print(f"[gaia] summary updated, n={len(new_rows)}")

    if args.benchmark in ("tau2bench", "all"):
        path = TRACES / "tau2bench__summary.jsonl"
        rows = _load_summary(path)
        # Need same scope: telecom sampled to 70, others full
        new_rows = _retry(
            rows,
            tau2_tasks(domain_limits={"telecom": 70}),
            _tau2_worker,
            args.parallel,
            "tau2",
        )
        _save_summary(path, new_rows)
        print(f"[tau2] summary updated, n={len(new_rows)}")


if __name__ == "__main__":
    main()
