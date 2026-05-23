"""Library-mode tau2 runner for the meta-evolution loop.

The tau2 CLI cannot run a candidate agent that lives outside tau2's own
package. This driver instead:
  1. imports tau2's global registry,
  2. registers a candidate agent factory under a chosen name,
  3. runs the official simulator via run_domain(),
  4. returns per-task reward.

A "candidate" is a Python module under agent_tau2/<name>/ exposing
`build_agent(tools, domain_policy, **kwargs) -> HalfDuplexAgent`.
v0 is just tau2's stock LLMAgent.

Usage:
  python tau2_runner.py --candidate v0 --domain mock --num-tasks 5
  python tau2_runner.py --candidate mh_tau2_iter1_<slug> --domain airline
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# DeepSeek official API via litellm's OpenAI-compatible path.
os.environ.setdefault("OPENAI_API_KEY", os.environ.get("DEEPSEEK_API_KEY", ""))
os.environ.setdefault("OPENAI_API_BASE", "https://api.deepseek.com")

DEFAULT_AGENT_LLM = os.environ.get("TAU2_AGENT_LLM", "openai/deepseek-v4-pro")
DEFAULT_USER_LLM = os.environ.get("TAU2_USER_LLM", "openai/deepseek-v4-pro")


def _load_candidate_factory(candidate: str):
    """Return a tau2 agent factory: agent_tau2/<candidate>/agent.py::build_agent.

    Every candidate — including v0 — is a uniform module exposing build_agent.
    """
    mod = importlib.import_module(f"agent_tau2.{candidate}.agent")
    return mod.build_agent


def run(candidate: str, domain: str, num_tasks: int | None,
        task_ids: list[str] | None, max_concurrency: int,
        save_to: str | None) -> dict:
    from tau2.registry import registry
    from tau2.data_model.simulation import TextRunConfig
    from tau2.run import run_domain

    factory = _load_candidate_factory(candidate)
    agent_name = f"cand__{candidate}"
    if agent_name not in registry.get_agents():
        registry.register_agent_factory(factory, agent_name)

    extra = {}
    if domain == "banking_knowledge":
        extra["retrieval_config"] = "bm25"

    out_path = save_to or f"tau2-runs/meta/{candidate}__{domain}.json"
    # Fresh run each eval — drop any stale checkpoint so the simulator does not
    # block on an interactive "resume?" prompt.
    stale = ROOT / "tau2-bench-src" / "data" / "simulations" / out_path
    if stale.exists():
        import shutil
        shutil.rmtree(stale, ignore_errors=True)

    # DeepSeek V4 defaults to thinking mode; in a multi-turn tool-use loop the
    # API then demands reasoning_content be threaded back through history (which
    # tau2 does not do). Disable thinking — tau2 is tool-use, not deep-reasoning.
    _ds_args = {
        "api_base": "https://api.deepseek.com",
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    config = TextRunConfig(
        domain=domain,
        agent=agent_name,
        llm_agent=DEFAULT_AGENT_LLM,
        llm_args_agent=dict(_ds_args),
        user="user_simulator",
        llm_user=DEFAULT_USER_LLM,
        llm_args_user=dict(_ds_args),
        num_trials=1,
        num_tasks=num_tasks,
        task_ids=task_ids,
        max_concurrency=max_concurrency,
        auto_resume=False,
        save_to=out_path,
        **extra,
    )
    results = run_domain(config)

    sims = results.simulations
    per_task = []
    for s in sims:
        reward = (s.reward_info.reward if s.reward_info else 0.0) or 0.0
        per_task.append({"task_id": s.task_id, "reward": reward})
    correct = sum(1 for p in per_task if p["reward"] > 0)

    # Summary jsonl in the same shape run_benchmark.py emits for GAIA, so
    # score_candidate.py can score tau2 candidates unchanged.
    summary_path = ROOT / "traces" / f"tau2_{candidate}__summary.jsonl"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        for p in per_task:
            f.write(json.dumps({
                "task_id": p["task_id"],
                "score": 1.0 if p["reward"] > 0 else 0.0,
                "reward": p["reward"],
                "domain": domain,
            }, ensure_ascii=False) + "\n")

    return {
        "candidate": candidate,
        "domain": domain,
        "n": len(per_task),
        "correct": correct,
        "acc": correct / max(len(per_task), 1),
        "summary_path": str(summary_path.relative_to(ROOT)),
        "per_task": per_task,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", default="v0")
    p.add_argument("--domain", default="mock")
    p.add_argument("--num-tasks", type=int, default=None)
    p.add_argument("--task-ids", nargs="*", default=None)
    p.add_argument("--task-ids-file", default=None,
                   help="newline-delimited task ids (overrides --task-ids)")
    p.add_argument("--max-concurrency", type=int, default=4)
    p.add_argument("--save-to", default=None)
    args = p.parse_args()

    task_ids = args.task_ids
    if args.task_ids_file:
        task_ids = [ln.strip() for ln in open(args.task_ids_file) if ln.strip()]

    out = run(args.candidate, args.domain, args.num_tasks,
              task_ids, args.max_concurrency, args.save_to)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
