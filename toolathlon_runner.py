"""Library-mode Toolathlon runner.

Sibling of tau2_runner.py / run_benchmark.py. Drives Toolathlon's
"decoupled" flow per task — container holds the MCP gateway + per-task
evaluator, host runs the OpenAI Agents SDK agent loop — through a pure
Python orchestrator (`agent_toolathlon.runtime.orchestrator`) so the
vendored Toolathlon-src tree stays untouched.

A candidate is a Python module under agent_toolathlon/<name>/agent.py
exposing `build_agent(**kwargs) -> TaskAgent`. v0 returns the stock
PrettyDecoupledTaskAgent.

Usage:
    # Smoke test on one LV0-a task with the v0 baseline
    python toolathlon_runner.py --candidate v0 \
        --task-ids find-alita-paper \
        --max-concurrency 1

    # Whole LV0-a corpus with v0 baseline
    python toolathlon_runner.py --candidate v0 \
        --tier-max LV0a --max-concurrency 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
TOOLATHLON_SRC = ROOT / "Toolathlon-src"
DEFAULT_DUMPS_ROOT = ROOT / "Toolathlon-runs"
IMAGE = "lockon0927/toolathlon-task-image:1016beta"

load_dotenv(ROOT / ".env")

# Default LLM endpoint: Together AI deepseek-v4-pro — same as tau2_runner.py
# (matches the project-wide baseline that GAIA + tau2-banking already used).
# Override with TOOLATHLON_OPENAI_* / TOOLATHLON_MODEL env vars in .env.
DEFAULT_BASE_URL = (
    os.environ.get("TOOLATHLON_OPENAI_BASE_URL")
    or "https://api.together.xyz/v1"
)
DEFAULT_API_KEY = (
    os.environ.get("TOOLATHLON_OPENAI_API_KEY")
    or os.environ.get("TOGETHER_API_KEY")
    or os.environ.get("TOGETHER_AI_API")
    or ""
)
DEFAULT_MODEL = os.environ.get("TOOLATHLON_MODEL", "deepseek-ai/DeepSeek-V4-Pro")
DEFAULT_PROVIDER = os.environ.get("TOOLATHLON_PROVIDER", "unified")
DEFAULT_MAX_STEPS = int(os.environ.get("TOOLATHLON_MAX_STEPS", "100"))

# Make bench/ + agent_toolathlon/ importable when this file is run as a script.
sys.path.insert(0, str(ROOT))

from bench.toolathlon.loader import iter_tasks, list_task_ids  # noqa: E402
from bench.toolathlon.scorer import score_from_dump  # noqa: E402
from agent_toolathlon.runtime.orchestrator import (  # noqa: E402
    OrchestratorConfig,
    run_task,
)
import subprocess  # noqa: E402  (still needed for the pre-clean docker run)


def _dumps_dir_for(candidate: str, dumps_root: Path) -> Path:
    return (dumps_root / candidate).resolve()


def _task_dump_dir(candidate: str, task_id: str, dumps_root: Path) -> Path:
    return _dumps_dir_for(candidate, dumps_root) / "finalpool" / task_id


def _ensure_dumps_dir(candidate: str, dumps_root: Path) -> Path:
    d = _dumps_dir_for(candidate, dumps_root)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pre_clean_dump(candidate: str, task_id: str, dumps_root: Path) -> None:
    """Use the task image (has root) to scrub leftover root-owned files."""
    try:
        subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{_dumps_dir_for(candidate, dumps_root)}:/x",
                IMAGE, "sh", "-c",
                f"rm -rf /x/finalpool/{task_id} && mkdir -p /x/finalpool/{task_id} && "
                f"chown -R {os.getuid()}:{os.getgid()} /x",
            ],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=120,
        )
    except Exception:
        pass


def _run_one_task(
    task: dict,
    candidate: str,
    dumps_root: Path,
    model: str,
    provider: str,
    max_steps: int,
    base_url: str,
    api_key: str,
) -> dict:
    task_id = task["task_id"]
    task_dump = _task_dump_dir(candidate, task_id, dumps_root)
    _pre_clean_dump(candidate, task_id, dumps_root)

    log_path = task_dump.parent.parent / "_runner_logs" / f"{task_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # LV0-a / LV0-b / LV1 tasks don't depend on the heavy remote credential
    # files (GCP / Snowflake JSONs in configs/); use Toolathlon's quickstart
    # semantics where a missing credential just warns instead of failing.
    runmode = "quickstart" if task.get("tier", "LV0a") in ("LV0a", "LV0b", "LV1") else "normal"

    cfg = OrchestratorConfig(
        toolathlon_src=TOOLATHLON_SRC,
        robagent_root=ROOT,
        candidate=candidate,
        image=IMAGE,
        output_folder=task_dump,
        logs_folder=task_dump,
        runmode=runmode,
        model=model,
        provider=provider,
        max_steps=max_steps,
        openai_base_url=base_url,
        openai_api_key=api_key,
    )
    started = time.time()
    orch_result = run_task(
        task_dir_arg=f"finalpool/{task_id}",
        cfg=cfg,
        host_log_path=log_path,
    )
    elapsed = time.time() - started

    score = score_from_dump(task_dump)

    return {
        "task_id": task_id,
        "tier": task["tier"],
        "score": score["score"],
        "passed": score["passed"],
        "details": score["details"][:500],
        "error": score["error"] or orch_result.get("error"),
        "status": score["status"],
        "host_loop_rc": orch_result.get("host_loop_rc"),
        "eval_rc": orch_result.get("eval_rc"),
        "elapsed_sec": round(elapsed, 1),
        "dump_dir": str(task_dump),
        "log_path": str(log_path),
    }


def run(
    candidate: str,
    task_ids: list[str] | None,
    tier_max: str,
    max_concurrency: int,
    dumps_root: Path,
    model: str,
    provider: str,
    max_steps: int,
    base_url: str,
    api_key: str,
    save_to: Path | None = None,
) -> dict:
    if not TOOLATHLON_SRC.exists():
        raise SystemExit(f"Toolathlon-src missing: {TOOLATHLON_SRC}")
    _ensure_dumps_dir(candidate, dumps_root)

    tasks = list(iter_tasks(task_ids=task_ids, tier_max=tier_max))
    if not tasks:
        raise SystemExit("No tasks selected (check --task-ids and --tier-max)")

    summary_path = save_to or (
        ROOT / "traces" / f"toolathlon_{candidate}__summary.jsonl"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[toolathlon:{candidate}] {len(tasks)} task(s) "
        f"tier<={tier_max} parallel={max_concurrency} → {summary_path}",
        flush=True,
    )

    out_lock = threading.Lock()
    correct = [0]
    started = time.time()
    records: list[dict] = []

    with summary_path.open("w", encoding="utf-8") as out_fh:
        with ThreadPoolExecutor(max_workers=max_concurrency) as ex:
            futures = {
                ex.submit(
                    _run_one_task,
                    t, candidate, dumps_root, model, provider, max_steps,
                    base_url, api_key,
                ): t
                for t in tasks
            }
            for fut in as_completed(futures):
                t = futures[fut]
                try:
                    rec = fut.result()
                except Exception as e:
                    rec = {
                        "task_id": t["task_id"],
                        "tier": t["tier"],
                        "score": 0.0,
                        "passed": False,
                        "error": repr(e),
                    }
                records.append(rec)
                with out_lock:
                    out_fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                    out_fh.flush()
                correct[0] += int(bool(rec.get("passed")))
                elapsed = time.time() - started
                print(
                    f"[toolathlon:{candidate}] "
                    f"{len(records):>3}/{len(tasks)} "
                    f"acc={correct[0]}/{len(records)}={correct[0]/max(len(records),1):.3f} "
                    f"elapsed={elapsed:.0f}s  "
                    f"{rec['tier']} {rec['task_id']} "
                    f"score={rec.get('score', 0)} "
                    f"host_loop={rec.get('host_loop_rc')} eval={rec.get('eval_rc')}",
                    flush=True,
                )

    return {
        "candidate": candidate,
        "n": len(records),
        "correct": correct[0],
        "acc": correct[0] / max(len(records), 1),
        "summary_path": str(summary_path),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", default="v0")
    p.add_argument("--task-ids", nargs="*", default=None,
                   help="explicit task_id list (overrides --tier-max filter)")
    p.add_argument("--task-ids-file", default=None,
                   help="newline-delimited task_ids file (overrides --task-ids)")
    p.add_argument("--tier-max", default="LV3",
                   choices=["LV0a", "LV0b", "LV1", "LV2", "LV3"],
                   help="include tasks up to this credential tier (default LV3 = all)")
    p.add_argument("--max-concurrency", type=int, default=1)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--provider", default=DEFAULT_PROVIDER)
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--api-key", default=DEFAULT_API_KEY)
    p.add_argument("--dumps-root", type=Path, default=DEFAULT_DUMPS_ROOT)
    p.add_argument("--save-to", type=Path, default=None,
                   help="override summary jsonl output path")
    p.add_argument("--list", action="store_true",
                   help="just print tasks at the given tier_max and exit")
    args = p.parse_args()

    task_ids = args.task_ids
    if args.task_ids_file:
        task_ids = [
            ln.strip() for ln in Path(args.task_ids_file).read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]

    if args.list:
        ids = list_task_ids(args.tier_max)
        if task_ids:
            ids = [i for i in ids if i in set(task_ids)]
        for i in ids:
            print(i)
        return

    if not args.api_key:
        raise SystemExit(
            "No LLM API key. Set TOOLATHLON_OPENAI_API_KEY or DEEPSEEK_API_KEY in .env."
        )

    out = run(
        candidate=args.candidate,
        task_ids=task_ids,
        tier_max=args.tier_max,
        max_concurrency=args.max_concurrency,
        dumps_root=args.dumps_root,
        model=args.model,
        provider=args.provider,
        max_steps=args.max_steps,
        base_url=args.base_url,
        api_key=args.api_key,
        save_to=args.save_to,
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
