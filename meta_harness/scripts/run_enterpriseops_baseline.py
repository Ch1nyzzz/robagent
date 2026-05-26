"""Run the EnterpriseOps wrapper agent on one domain.

Canonical entry for the evolution pipeline. Per-task it invokes our wrapper
agent against the upstream MCP-server + verifier pipeline, then writes:

  meta_harness/logs_components_enterpriseops_<domain>/
      <agent_name>__<subset>__results.json    # task-by-task + summary
      frontier_val.json                       # per-task best score (frontier)
      evolution_summary.jsonl                 # one line per iteration

Frontier-as-directory: the agent lives at `agent/enterpriseops/<v_dir>/agent.py`,
the components live at `agent/enterpriseops/<v_dir>/components_<domain>/`.
Caller passes `--agent-dir` pointing at a v_N. There is no workflow YAML.

Important — upstream venv
-------------------------
This script imports upstream `third_party/EnterpriseOps-Gym` Python modules
(LangChain, langchain_deepseek, datasets, …) which live in the upstream's
uv-managed venv, NOT in the host Python. Invoke through that interpreter:

    UPSTREAM_PY=/data/home/yuhan/robagent/third_party/EnterpriseOps-Gym/.venv/bin/python
    bash -c "set -a && source /data/home/yuhan/robagent/.env && set +a && \\
      PYTHONPATH=/data/home/yuhan/robagent:/data/home/yuhan/robagent/third_party/EnterpriseOps-Gym \\
      $UPSTREAM_PY meta_harness/scripts/run_enterpriseops_baseline.py ..."

If `langchain_deepseek` import fails: `cd third_party/EnterpriseOps-Gym && uv sync --extra all`.

Usage
-----
    "$UPSTREAM_PY" meta_harness/scripts/run_enterpriseops_baseline.py \\
        --domain calendar --agent-name v0 --iteration 0 \\
        --agent-dir agent/enterpriseops/v0 \\
        --train-split meta_harness/enterpriseops_calendar_train_task_ids.txt \\
        --subsets train --provider deepseek --concurrency 5
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

UPSTREAM = ROOT / "third_party" / "EnterpriseOps-Gym"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from benchmark.models import LLMConfig  # noqa: E402


def _load_run_task(agent_dir: Path) -> Callable:
    """Import the `run_task` callable from `<agent_dir>/agent.py`.

    The agent dir is expected to be `agent/enterpriseops/<v_N>/` containing
    `agent.py`, a `runtime/` package, and per-domain `components_<domain>/`
    directories — a regular Python package on ROOT-relative sys.path, so a
    plain `import_module` works and relative imports inside `agent.py`
    (`from .runtime import ...`) resolve against the matching v_N tree.
    """
    agent_dir = agent_dir.resolve()
    if not (agent_dir / "agent.py").exists():
        raise SystemExit(f"agent.py not found in {agent_dir}")
    rel = agent_dir.relative_to(ROOT)
    mod_name = ".".join(rel.parts) + ".agent"
    mod = importlib.import_module(mod_name)
    if not hasattr(mod, "run_task"):
        raise SystemExit(f"{agent_dir}/agent.py does not export run_task")
    return mod.run_task


# --------------------------------------------------------------------------- #
# Provider presets — kept in lock-step with `enterpriseops_smoke.py`.
# --------------------------------------------------------------------------- #
PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    "deepseek": {
        # deepseek-chat is the non-thinking alias of deepseek-v4-flash during
        # the grace period (until 2026-07-24); avoids the reasoning_content
        # round-trip requirement that breaks langchain bind_tools loops.
        "llm_provider": "deepseek",
        "llm_model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
        "temperature": 0.0,
        "max_tokens": 8192,
    },
    "openai": {
        "llm_provider": "openai",
        "llm_model": "gpt-4.1-mini",
        "env_key": "OPENAI_API_KEY",
        "temperature": 0.0,
        "max_tokens": 8192,
    },
    "anthropic": {
        "llm_provider": "anthropic",
        "llm_model": "claude-sonnet-4-6",
        "env_key": "ANTHROPIC_API_KEY",
        "temperature": 0.0,
        "max_tokens": 8192,
    },
}

_JSON_STRING_FIELDS = {"gym_servers_config", "verifiers"}
_HF_ONLY_FIELDS = {"task_id", "domain"}


# --------------------------------------------------------------------------- #
def logs_dir_for(domain: str) -> Path:
    d = ROOT / "meta_harness" / f"logs_components_enterpriseops_{domain}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_llm_config(provider: str, override: Optional[str] = None) -> LLMConfig:
    if override:
        d = json.loads(Path(override).read_text())
        return LLMConfig(**d)
    preset = PROVIDER_PRESETS[provider]
    key = os.environ.get(preset["env_key"])
    if not key:
        raise SystemExit(f"{preset['env_key']} not in env (did you source .env?)")
    return LLMConfig(
        llm_provider=preset["llm_provider"],
        llm_model=preset["llm_model"],
        llm_api_key=key,
        temperature=preset["temperature"],
        max_tokens=preset["max_tokens"],
    )


def load_hf_tasks_by_id(domain: str, mode: str, task_ids: list[str],
                        hf_dataset: str) -> dict[str, dict[str, Any]]:
    """Return task_id -> normalised task config dict, keyed for fast lookup."""
    from datasets import load_dataset  # late import; heavy module
    ds = load_dataset(hf_dataset, mode, split=domain)
    wanted = set(task_ids)
    out: dict[str, dict[str, Any]] = {}
    for row in ds:
        tid = str(row.get("task_id", ""))
        if tid not in wanted:
            continue
        cfg: dict[str, Any] = {}
        for k, v in row.items():
            if k in _HF_ONLY_FIELDS:
                continue
            if k in _JSON_STRING_FIELDS and isinstance(v, str):
                v = json.loads(v)
            cfg[k] = v
        out[tid] = cfg
    missing = wanted - out.keys()
    if missing:
        raise SystemExit(f"{len(missing)} requested task_id(s) not in HF split "
                         f"{domain}: {sorted(missing)[:5]}{' ...' if len(missing) > 5 else ''}")
    return out


# --------------------------------------------------------------------------- #
def _was_correct(task_result: dict[str, Any]) -> bool:
    return bool(task_result.get("success"))


def _summarize(task_results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(task_results)
    correct = sum(1 for r in task_results if _was_correct(r))
    completed = sum(1 for r in task_results if not r.get("error"))
    vrates = [r["verifier_pass_rate"] for r in task_results
              if r.get("verifier_pass_rate") is not None]
    return {
        "n_tasks": n,
        "n_completed": completed,
        "n_correct": correct,
        "tsr": round(correct / max(n, 1), 4),
        "ecr": round(completed / max(n, 1), 4),
        "verifier_pass_rate_mean": round(sum(vrates) / max(len(vrates), 1), 4),
    }


def _update_frontier(
    frontier_path: Path,
    agent_name: str,
    task_results: list[dict[str, Any]],
    subset_ids: list[str],
) -> dict[str, Any]:
    """frontier = best per-task score across all agents that have ever scored it.

    Mirrors `run_sopbench_baseline._update_frontier`. Strict pass (1.0) wins;
    if multiple agents tied at 1.0, the most recent score is kept.
    """
    frontier: dict[str, Any] = {}
    if frontier_path.exists():
        try:
            frontier = json.loads(frontier_path.read_text())
        except json.JSONDecodeError:
            frontier = {}
    per_task = frontier.setdefault("per_task", {})
    by_id = {str(r.get("task_id")): r for r in task_results}
    for tid in subset_ids:
        tr = by_id.get(tid)
        if tr is None:
            continue
        score = 1.0 if _was_correct(tr) else 0.0
        cur = per_task.get(tid)
        if cur is None or score > cur.get("score", 0):
            per_task[tid] = {
                "agent": agent_name,
                "score": score,
                "verifier_pass_rate": tr.get("verifier_pass_rate"),
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


# --------------------------------------------------------------------------- #
async def _run_one(
    sem: asyncio.Semaphore,
    *,
    task_id: str,
    task_cfg: dict[str, Any],
    domain: str,
    llm_config: LLMConfig,
    run_task: Callable,
    components_dir: Path,
    run_tag: str,
    traces_dir: Path,
) -> dict[str, Any]:
    async with sem:
        t0 = time.time()
        try:
            result = await run_task(
                task_cfg,
                llm_config,
                domain=domain,
                task_id=task_id,
                components_dir=components_dir,
                run_tag=run_tag,
            )
        except BaseException as exc:  # noqa: BLE001 — surface any failure
            wall = time.time() - t0
            return {
                "task_id": task_id,
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
                "wall_seconds": round(wall, 2),
                "verifier_pass_rate": None,
                "verifier_total": None,
                "verifier_passed": None,
                "tools_used": [],
            }
        wall = time.time() - t0
        runs = result.get("runs") or [{}]
        r0 = runs[0]
        vs = r0.get("verification_summary") or {}
        tr = {
            "task_id": task_id,
            "success": bool(r0.get("overall_success")),
            "error": r0.get("error"),
            "wall_seconds": round(wall, 2),
            "verifier_pass_rate": vs.get("pass_rate"),
            "verifier_passed": vs.get("passed"),
            "verifier_total": vs.get("total"),
            "tools_used": r0.get("tools_used") or [],
            "n_steps_proxy": len(r0.get("conversation_flow") or []),
            "execution_time_ms": r0.get("execution_time_ms"),
            "component_trace": result.get("component_trace") or {},
        }
        # Persist the full per-task trace separately to keep the summary file small.
        (traces_dir / f"{task_id}.json").write_text(
            json.dumps({
                "task_id": task_id,
                "domain": domain,
                "result": result,
            }, indent=2, default=str)
        )
        return tr


async def run_subset(
    *,
    domain: str,
    subset_name: str,
    subset_ids: list[str],
    task_configs: dict[str, dict[str, Any]],
    agent_name: str,
    llm_config: LLMConfig,
    run_task: Callable,
    components_dir: Path,
    concurrency: int,
    logs_dir: Path,
) -> dict[str, Any]:
    print(f"[{domain}/{subset_name}] running {len(subset_ids)} task(s); "
          f"concurrency={concurrency}; agent={agent_name}")
    traces_dir = logs_dir / f"{agent_name}__{subset_name}__traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    run_tag = f"{agent_name}__{subset_name}"

    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()
    coros = [
        _run_one(
            sem,
            task_id=tid,
            task_cfg=task_configs[tid],
            domain=domain,
            llm_config=llm_config,
            run_task=run_task,
            components_dir=components_dir,
            run_tag=run_tag,
            traces_dir=traces_dir,
        )
        for tid in subset_ids
    ]
    task_results = await asyncio.gather(*coros)
    elapsed = time.time() - t0

    summary = _summarize(task_results)
    out = {
        "domain": domain,
        "subset": subset_name,
        "agent": agent_name,
        "subset_ids": subset_ids,
        "summary": summary,
        "results": task_results,
        "wall_seconds": round(elapsed, 1),
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "components_dir": str(components_dir),
        "llm": f"{llm_config.llm_provider}/{llm_config.llm_model}",
    }
    out_path = logs_dir / f"{agent_name}__{subset_name}__results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(
        f"[{domain}/{subset_name}] done in {elapsed:.1f}s   "
        f"TSR={summary['tsr']:.1%}  ECR={summary['ecr']:.1%}  "
        f"verifier_mean={summary['verifier_pass_rate_mean']:.3f}  "
        f"-> {out_path}"
    )
    return out


# --------------------------------------------------------------------------- #
def _read_split_file(path: str | Path) -> list[str]:
    return [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]


def _load_plugin_json(arg: str) -> Optional[dict[str, Any]]:
    if not arg:
        return None
    if arg.startswith("@"):
        return json.loads(Path(arg[1:]).read_text())
    return json.loads(arg)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True,
                   choices=["calendar", "csm", "drive", "email",
                            "hr", "itsm", "teams", "hybrid"])
    p.add_argument("--mode", default="oracle",
                   choices=["oracle", "plus_5_tools",
                            "plus_10_tools", "plus_15_tools"])
    p.add_argument("--hf-dataset", default="ServiceNow-AI/EnterpriseOps-Gym")
    p.add_argument("--agent-name", default="v0")
    p.add_argument("--iteration", type=int, default=0)
    p.add_argument("--train-split", required=True,
                   help="txt file: train task_ids (one per line)")
    p.add_argument("--test-split", default=None,
                   help="txt file: test task_ids; required for the test subset")
    p.add_argument("--subsets", default="train,test",
                   help="comma-separated: train, test, or both")
    p.add_argument("--agent-dir", required=True,
                   help="path to agent/enterpriseops/<v_N>/ — must contain "
                        "agent.py, runtime/, components_<domain>/")
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--provider", default="deepseek",
                   choices=sorted(PROVIDER_PRESETS.keys()))
    p.add_argument("--llm-config", default=None,
                   help="path to override LLMConfig JSON")
    p.add_argument("--hypothesis",
                   default="upstream ReAct + LLM only (v0 baseline)")
    p.add_argument("--changes", default="")
    p.add_argument("--plugin-json", default="",
                   help="raw JSON or @path: candidate spec from proposer; "
                        "stored verbatim into evolution_summary for audit trail.")
    args = p.parse_args()

    logs_dir = logs_dir_for(args.domain)
    llm_config = build_llm_config(args.provider, args.llm_config)

    agent_dir = Path(args.agent_dir).resolve()
    run_task = _load_run_task(agent_dir)
    components_dir = agent_dir / f"components_{args.domain}"

    subsets_to_run = {s.strip() for s in args.subsets.split(",") if s.strip()}

    # Pre-load HF rows we need (union of train+test) in one pass.
    train_ids = _read_split_file(args.train_split)
    test_ids = _read_split_file(args.test_split) if args.test_split else []
    all_ids = list(dict.fromkeys(train_ids + test_ids))  # dedupe, preserve order
    print(f"[{args.domain}] loading {len(all_ids)} task config(s) from HF "
          f"({args.hf_dataset} :: {args.mode}/{args.domain}) ...")
    task_configs = load_hf_tasks_by_id(args.domain, args.mode, all_ids, args.hf_dataset)

    outputs: dict[str, dict[str, Any]] = {}
    if "train" in subsets_to_run and train_ids:
        outputs["train"] = asyncio.run(run_subset(
            domain=args.domain,
            subset_name="train",
            subset_ids=train_ids,
            task_configs=task_configs,
            agent_name=args.agent_name,
            llm_config=llm_config,
            run_task=run_task,
            components_dir=components_dir,
            concurrency=args.concurrency,
            logs_dir=logs_dir,
        ))
    if "test" in subsets_to_run and test_ids:
        outputs["test"] = asyncio.run(run_subset(
            domain=args.domain,
            subset_name="test",
            subset_ids=test_ids,
            task_configs=task_configs,
            agent_name=args.agent_name,
            llm_config=llm_config,
            run_task=run_task,
            components_dir=components_dir,
            concurrency=args.concurrency,
            logs_dir=logs_dir,
        ))

    # Frontier + evolution_summary are driven by the TRAIN subset (gate signal).
    if "train" in outputs:
        train_out = outputs["train"]
        frontier_path = logs_dir / "frontier_val.json"
        frontier_after = _update_frontier(
            frontier_path, args.agent_name, train_out["results"], train_ids
        )
        print(f"[{args.domain}] frontier after train: "
              f"{frontier_after['correct']}/{frontier_after['total']} = {frontier_after['acc']}")

        evo_path = logs_dir / "evolution_summary.jsonl"
        entry: dict[str, Any] = {
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
            "llm": f"{llm_config.llm_provider}/{llm_config.llm_model}",
        }
        plugin = _load_plugin_json(args.plugin_json) if args.plugin_json else None
        if plugin:
            entry["plugin"] = plugin
        with evo_path.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        print(f"[{args.domain}] appended evolution_summary row -> {evo_path}")


if __name__ == "__main__":
    main()
