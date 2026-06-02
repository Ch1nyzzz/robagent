"""Phase 0 PoC driver for ServiceNow EnterpriseOps-Gym.

Goal
----
Answer two questions before committing to a full ballast integration:

  1. Single-task wall-clock + (approx) cost — to size the proposer loop's
     frontier_set and concurrency budget.
  2. Baseline pass rate on the chosen domain/mode — to confirm we are in the
     20–60% "evolvable" band (vs. saturated or bricked).

This script does NOT touch agent/, agent_toolathlon/, or ballast/workflows/.
It just drives the upstream ``third_party/EnterpriseOps-Gym/evaluate.py`` over a
small slice of tasks and aggregates the resulting JSONs into one report.

Manual prerequisites (do these once)
------------------------------------
1. ``cd third_party/EnterpriseOps-Gym && uv sync --extra anthropic`` (or whatever
   extras match ``--provider``).
2. ``cd third_party/EnterpriseOps-Gym && unzip -n gym_dbs.zip`` (only needed if
   you plan to seed DBs from local snapshots; for HF-loaded tasks the seed file
   field is already a relative path that the upstream resolves).
3. Pull and run the MCP server for the chosen domain:

     docker pull shivakrishnareddyma225/enterpriseops-gym-mcp-calendar:latest
     docker run -d --name eog-calendar -p 8003:8003 \\
         shivakrishnareddyma225/enterpriseops-gym-mcp-calendar:latest

   Default ports (from upstream README):
     teams=8002, csm=8001, email=8004, itsm=8006, calendar=8003,
     hr=8008, drive=8009.

Run
---
    set -a && source /data/home/yuhan/robagent/.env && set +a && \\
    python ballast/scripts/enterpriseops_smoke.py \\
        --domain calendar --n-tasks 3 --provider deepseek

Output
------
A single summary block printed to stdout, plus per-task result JSONs under
``ballast/logs_components_enterpriseops_<domain>/smoke/<timestamp>/``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "third_party" / "EnterpriseOps-Gym"


# --------------------------------------------------------------------------- #
# Provider presets — map a short name to upstream's llm_config JSON shape.    #
# Phase 0 only: keep this small. Add more as needed.                          #
# --------------------------------------------------------------------------- #
PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    "deepseek": {
        # `deepseek-chat` is the non-thinking alias of deepseek-v4-flash during
        # the grace period (until 2026-07-24). We pick the non-thinking variant
        # because the thinking mode requires the client to round-trip
        # `reasoning_content`, which langchain_deepseek currently does not do —
        # the second turn of any tool-use loop 400s otherwise.
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

# Upstream's HF-row → config-file translation (see evaluate.py:286-308).
_JSON_STRING_FIELDS = {"gym_servers_config", "verifiers"}
_HF_ONLY_FIELDS = {"task_id", "domain"}


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="calendar",
                   choices=["calendar", "csm", "drive", "email",
                            "hr", "itsm", "teams", "hybrid"])
    p.add_argument("--mode", default="oracle",
                   choices=["oracle", "plus_5_tools",
                            "plus_10_tools", "plus_15_tools"])
    p.add_argument("--n-tasks", type=int, default=3,
                   help="Tasks to run; deterministic slice (sorted task_id).")
    p.add_argument("--provider", default="deepseek",
                   choices=sorted(PROVIDER_PRESETS.keys()))
    p.add_argument("--llm-config", default=None,
                   help="Override: full path to an existing upstream llm config JSON.")
    p.add_argument("--orchestrator", default="react",
                   choices=["react", "planner_react", "decomposing"])
    p.add_argument("--concurrency", type=int, default=1,
                   help="Per-task concurrency for upstream eval. Phase 0: keep 1.")
    p.add_argument("--num-runs", type=int, default=1,
                   help="Reruns per task. 1 is enough for Phase 0 sizing.")
    p.add_argument("--hf-dataset", default="ServiceNow-AI/EnterpriseOps-Gym")
    p.add_argument("--task-id", action="append", default=None,
                   help="Pin specific task_ids (repeatable). Overrides --n-tasks.")
    p.add_argument("--split-file", default=None,
                   help="Path to a txt file with one task_id per line. "
                        "Overrides --n-tasks and --task-id.")
    p.add_argument("--out-root", default=None,
                   help="Override output root (default: ballast/logs_components_enterpriseops_<domain>/smoke/<ts>)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
def ensure_llm_config(args: argparse.Namespace, workdir: Path) -> Path:
    """Either return user-supplied --llm-config, or synthesize one from env."""
    if args.llm_config:
        p = Path(args.llm_config).expanduser().resolve()
        if not p.exists():
            sys.exit(f"--llm-config not found: {p}")
        return p

    preset = PROVIDER_PRESETS[args.provider]
    api_key = os.environ.get(preset["env_key"])
    if not api_key:
        sys.exit(f"{preset['env_key']} not in env; did you source .env?")

    cfg = {
        "llm_provider": preset["llm_provider"],
        "llm_model": preset["llm_model"],
        "llm_api_key": api_key,
        "temperature": preset["temperature"],
        "max_tokens": preset["max_tokens"],
    }
    path = workdir / f"llm_{args.provider}.json"
    path.write_text(json.dumps(cfg, indent=2))
    return path


# --------------------------------------------------------------------------- #
def slice_hf_dataset(args: argparse.Namespace, configs_dir: Path) -> list[str]:
    """Load HF rows for (mode, domain), pick N by sorted task_id, write configs.

    Returns the list of task_ids actually selected.
    """
    # Late import so the script can print --help without `datasets` installed.
    from datasets import load_dataset

    ds = load_dataset(args.hf_dataset, args.mode, split=args.domain)
    rows = list(ds)
    rows.sort(key=lambda r: str(r.get("task_id", "")))

    if args.split_file:
        sf = Path(args.split_file).expanduser().resolve()
        if not sf.exists():
            sys.exit(f"--split-file not found: {sf}")
        wanted = [ln.strip() for ln in sf.read_text().splitlines() if ln.strip()]
        wanted_set = set(wanted)
        rows = [r for r in rows if r.get("task_id") in wanted_set]
        missing = wanted_set - {r.get("task_id") for r in rows}
        if missing:
            sys.exit(f"task_id(s) in split-file not found in HF split: "
                     f"{sorted(missing)[:5]}{' ...' if len(missing) > 5 else ''}")
    elif args.task_id:
        wanted = set(args.task_id)
        rows = [r for r in rows if r.get("task_id") in wanted]
        missing = wanted - {r.get("task_id") for r in rows}
        if missing:
            sys.exit(f"task_id(s) not found in split: {sorted(missing)}")
    else:
        rows = rows[: args.n_tasks]

    selected: list[str] = []
    for row in rows:
        task_id = row.get("task_id") or f"task_{id(row)}"
        selected.append(task_id)
        fname = f"{args.mode}__{args.domain}__{task_id}.json"
        task_dict: dict[str, Any] = {}
        for k, v in row.items():
            if k in _HF_ONLY_FIELDS:
                continue
            if k in _JSON_STRING_FIELDS and isinstance(v, str):
                v = json.loads(v)
            task_dict[k] = v
        (configs_dir / fname).write_text(json.dumps(task_dict))

    return selected


# --------------------------------------------------------------------------- #
def run_upstream(args: argparse.Namespace,
                 configs_dir: Path,
                 llm_config: Path,
                 results_dir: Path) -> int:
    """Invoke ``third_party/EnterpriseOps-Gym/evaluate.py --configs_folder ...``.

    We run upstream in its own cwd so its relative imports resolve.
    """
    if not UPSTREAM.exists():
        sys.exit(f"Upstream not cloned at {UPSTREAM}; "
                 f"git clone https://github.com/ServiceNow/EnterpriseOps-Gym {UPSTREAM}")

    # Upstream uses uv (pyproject.toml + extras). Prefer `uv run` so its venv is
    # used regardless of what python is invoking this smoke script. Fall back to
    # plain `python` only if uv is unavailable, with a loud warning.
    uv_bin = shutil.which("uv")
    eval_args = [
        "evaluate.py",
        "--configs_folder", str(configs_dir),
        "--llm_config", str(llm_config),
        "--output_folder", str(results_dir),
        "--orchestrator", args.orchestrator,
        "--concurrency", str(args.concurrency),
        "--num_runs", str(args.num_runs),
    ]
    if uv_bin:
        cmd = [uv_bin, "run", "python", *eval_args]
    else:
        print("[smoke] WARNING: `uv` not on PATH; using host python. Upstream "
              "deps must already be importable.", flush=True)
        cmd = [sys.executable, *eval_args]

    print(f"[smoke] launching upstream (cwd={UPSTREAM}):\n  {' '.join(cmd)}",
          flush=True)
    proc = subprocess.run(cmd, cwd=str(UPSTREAM))
    return proc.returncode


# --------------------------------------------------------------------------- #
def collect_results(results_dir: Path,
                    selected_ids: Iterable[str]) -> list[dict[str, Any]]:
    """Walk results_dir/run_*/results_*.json and pull a flat per-task summary."""
    summaries: list[dict[str, Any]] = []
    selected = set(selected_ids)
    for run_dir in sorted(results_dir.glob("run_*")):
        for jf in sorted(run_dir.glob("results_*.json")):
            try:
                data = json.loads(jf.read_text())
            except json.JSONDecodeError as exc:
                summaries.append({
                    "task_id": jf.stem, "run_dir": run_dir.name,
                    "error": f"json_decode: {exc}", "passed": False,
                    "wall_ms": None, "n_steps": None, "verifier_pass_rate": None,
                })
                continue

            # task_id is encoded in the filename: results_<mode>__<domain>__<task_id>.json
            stem = jf.stem.removeprefix("results_")
            parts = stem.split("__", 2)
            task_id = parts[2] if len(parts) == 3 else stem
            if selected and task_id not in selected:
                continue

            stats = data.get("statistics") or {}
            runs = data.get("runs") or []
            first = runs[0] if runs else {}
            verifier_summary = first.get("verification_summary") or {}
            summaries.append({
                "task_id": task_id,
                "run_dir": run_dir.name,
                "passed": bool(first.get("overall_success")),
                "error": first.get("error"),
                "wall_ms": first.get("execution_time_ms"),
                "n_tools_called": len(first.get("tools_used") or []),
                "verifier_pass_rate": verifier_summary.get("pass_rate"),
                "verifier_total": verifier_summary.get("total"),
                "verifier_passed": verifier_summary.get("passed"),
                "n_steps_proxy": len(first.get("conversation_flow") or []),
                "mean_wall_ms_all_runs": stats.get("mean_execution_time_ms"),
            })
    return summaries


# --------------------------------------------------------------------------- #
def print_report(args: argparse.Namespace,
                 summaries: list[dict[str, Any]],
                 wall_total_s: float) -> None:
    n = len(summaries)
    if n == 0:
        print("\n[smoke] no per-task results were collected; check upstream logs.")
        return

    walls = [s["wall_ms"] for s in summaries if s.get("wall_ms")]
    steps = [s["n_steps_proxy"] for s in summaries if s.get("n_steps_proxy")]
    tools = [s["n_tools_called"] for s in summaries if s.get("n_tools_called") is not None]
    vrates = [s["verifier_pass_rate"] for s in summaries
              if s.get("verifier_pass_rate") is not None]
    n_pass = sum(1 for s in summaries if s.get("passed"))
    n_err = sum(1 for s in summaries if s.get("error"))

    print("\n" + "=" * 72)
    print(f" EnterpriseOps-Gym SMOKE — {args.domain}/{args.mode}  "
          f"({args.provider}, {args.orchestrator})")
    print("=" * 72)
    print(f" tasks selected           : {n}")
    print(f" tasks passed (all verif) : {n_pass}/{n}  "
          f"({100 * n_pass / n:.1f}%)")
    print(f" tasks with hard error    : {n_err}")
    if vrates:
        print(f" verifier pass rate (mean): {mean(vrates):.3f}  "
              f"(median {median(vrates):.3f})")
    if walls:
        print(f" wall per task ms         : "
              f"median {median(walls):.0f}  "
              f"mean {mean(walls):.0f}  "
              f"max {max(walls)}")
    if steps:
        print(f" conv steps (proxy)       : "
              f"median {median(steps):.0f}  max {max(steps)}")
    if tools:
        print(f" tools invoked / task     : "
              f"median {median(tools):.0f}  max {max(tools)}")
    print(f" total smoke wall (s)     : {wall_total_s:.1f}")
    print("=" * 72)
    print(" per-task:")
    for s in summaries:
        flag = "PASS" if s["passed"] else ("ERR " if s["error"] else "FAIL")
        wall = f"{(s['wall_ms'] or 0) / 1000:6.1f}s"
        vp = (f"{s['verifier_passed']}/{s['verifier_total']}"
              if s.get("verifier_total") is not None else "  -")
        err = f"  err={s['error'][:60]}" if s.get("error") else ""
        print(f"  [{flag}] {s['task_id']:42s} {wall}  verif={vp}{err}")
    print("=" * 72)

    # Phase 0 decision rule of thumb
    if n >= 3 and walls:
        med_s = median(walls) / 1000
        pass_rate = n_pass / n
        print("\n Phase 0 decision hints:")
        if med_s > 300:
            print(f"  ! median wall {med_s:.0f}s/task > 300s  "
                  f"→ frontier_set should stay ≤ 20 tasks.")
        elif med_s > 120:
            print(f"  · median wall {med_s:.0f}s/task ∈ (120, 300]  "
                  f"→ frontier_set ≤ 30 tasks.")
        else:
            print(f"  ✓ median wall {med_s:.0f}s/task ≤ 120s  "
                  f"→ frontier_set up to 50 is feasible.")

        if not (0.10 <= pass_rate <= 0.80):
            print(f"  ! pass_rate {pass_rate:.0%} outside [10%, 80%] sweet spot "
                  f"→ try a different domain or model before committing.")
        else:
            print(f"  ✓ pass_rate {pass_rate:.0%} in evolvable band.")


# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = (Path(args.out_root) if args.out_root
                else ROOT / "ballast"
                / f"logs_components_enterpriseops_{args.domain}"
                / "smoke" / ts)
    out_root.mkdir(parents=True, exist_ok=True)
    configs_dir = out_root / "configs"
    results_dir = out_root / "results"
    configs_dir.mkdir()
    results_dir.mkdir()

    llm_config = ensure_llm_config(args, out_root)
    print(f"[smoke] llm_config -> {llm_config}", flush=True)

    selected = slice_hf_dataset(args, configs_dir)
    print(f"[smoke] selected {len(selected)} task(s): "
          f"{', '.join(selected[:5])}{' …' if len(selected) > 5 else ''}",
          flush=True)
    if not selected:
        sys.exit("[smoke] no tasks selected — exiting.")

    t0 = time.time()
    rc = run_upstream(args, configs_dir, llm_config, results_dir)
    wall_total = time.time() - t0

    if rc != 0:
        print(f"[smoke] upstream returncode={rc} (continuing to harvest "
              f"any partial results)", flush=True)

    summaries = collect_results(results_dir, selected)
    (out_root / "smoke_summary.json").write_text(json.dumps(
        {
            "args": vars(args),
            "selected_task_ids": selected,
            "wall_total_s": wall_total,
            "upstream_returncode": rc,
            "per_task": summaries,
        },
        indent=2,
        default=str,
    ))
    print_report(args, summaries, wall_total)
    print(f"\n[smoke] artifacts: {out_root}")

    # Phase 0 contract: non-zero exit if upstream itself crashed.
    sys.exit(0 if rc == 0 else rc)


if __name__ == "__main__":
    main()
