"""Outer evolution loop for GAIA harness search.

Each iteration:
  1. Spawn a headless Claude (`claude -p`) with the meta-harness-gaia skill.
     The proposer reads frontier/evolution state + failed traces, writes a new
     candidate agent under `agent/mh_iter<N>_<slug>/`, and emits
     `meta_harness/logs/pending_eval.json`.
  2. Read `pending_eval.json`, evaluate the candidate on the train-30 subset
     via `run_benchmark.py`.
  3. Score on train-30, update `frontier_val.json` + append to
     `evolution_summary.jsonl`.

After all iterations, optionally re-score the frontier on test-135.

Usage:
  python meta_harness/meta_harness.py --iterations 1 --train-parallel 4
  python meta_harness/meta_harness.py --iterations 3 --final-test
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))

import claude_wrapper  # noqa: E402  (vendored helper)

DEFAULT_LOGS = THIS_DIR / "logs"
TRAIN_IDS_FILE = THIS_DIR / "train_task_ids.txt"
TEST_IDS_FILE = THIS_DIR / "test_task_ids.txt"
SKILLS_PARENT = THIS_DIR / ".claude" / "skills"
DEFAULT_SKILL_NAME = "meta-harness-gaia"

# These are set in main() based on --logs-dir.
LOGS: Path  # type: ignore[assignment]
PENDING_EVAL: Path  # type: ignore[assignment]
FRONTIER_VAL: Path  # type: ignore[assignment]
EVOLUTION_SUMMARY: Path  # type: ignore[assignment]

PROPOSER_MODEL = os.environ.get("MH_PROPOSER_MODEL", "sonnet")
PROPOSER_TOOLS = ["Read", "Glob", "Grep", "Edit", "Write", "Bash"]


def _current_iteration() -> int:
    if not EVOLUTION_SUMMARY.exists():
        return 1
    last = 0
    with EVOLUTION_SUMMARY.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                last = max(last, int(rec.get("iteration", 0)))
            except json.JSONDecodeError:
                continue
    return last + 1


def _proposer_prompt(iteration: int, skill_name: str, slug_prefix: str) -> str:
    frontier_path = FRONTIER_VAL.relative_to(ROOT)
    summary_path = EVOLUTION_SUMMARY.relative_to(ROOT)
    pending_path = PENDING_EVAL.relative_to(ROOT)
    name_pattern = (
        f"mh_iter{iteration}_{slug_prefix}_<slug>" if slug_prefix
        else f"mh_iter{iteration}_<slug>"
    )
    return f"""You are a GAIA harness proposer. Follow the SKILL.md ({skill_name}) you were given.

Iteration: {iteration}
Working directory: {ROOT}

State files (relative to working directory):
  - Frontier per-task best:   {frontier_path}
  - Prior iteration summary:  {summary_path}
  - Output pending eval to:   {pending_path}
  - Train task ids:           meta_harness/train_task_ids.txt (30 tasks)
  - Baseline agent (v0):      agent/base.py
  - Prior candidates:         agent/mh_iter*/  (if any)
  - Train traces:             traces/runs/gaia__<task_id>__*.jsonl

Steps:
  1. Read frontier_val.json and the last few entries of evolution_summary.jsonl.
  2. Pick 4-6 task_ids from train-30 where the frontier is wrong (score == 0).
     Sample their latest trace under traces/runs/.
  3. Form ONE hypothesis tied to a mechanism present in >=3 of those traces.
  4. Create agent/{name_pattern}/{{__init__.py, base.py}}. base.py must
     export run_task() with the same signature as agent/base.py.
  5. Validate the import: `python -c "from agent.{name_pattern}.base import run_task; print('ok')"`.
  6. Write {pending_path} with the JSON schema from SKILL.md.
  7. Print one final line: CANDIDATE: <candidate_name>

CRITICAL:
  - Do NOT run benchmarks. Do NOT modify run_benchmark.py, claude_wrapper.py,
    meta_harness.py, agent/base.py, or any existing agent/v*/ or agent/mh_*/.
  - No task-specific hardcoding. No entity names in code.
"""


def run_proposer(iteration: int, log_dir: Path, skill_name: str, slug_prefix: str) -> dict:
    prompt = _proposer_prompt(iteration, skill_name, slug_prefix)
    print(f"\n=== iter {iteration}: proposer starting (skill={skill_name}) ===", flush=True)
    result = claude_wrapper.run(
        prompt=prompt,
        model=PROPOSER_MODEL,
        allowed_tools=PROPOSER_TOOLS,
        cwd=str(ROOT),
        log_dir=str(log_dir),
        name=f"iter{iteration}_proposer_{skill_name}",
        skills=[skill_name],
        skill_dir=str(SKILLS_PARENT),
        disable_skills=True,
        disable_mcp=True,
        progress=True,
    )
    print(f"  exit_code={result.exit_code}  "
          f"tokens={result.token_usage}  cost=${result.cost_usd:.4f}  "
          f"duration={result.duration_seconds:.1f}s", flush=True)
    if result.exit_code != 0:
        print(f"  STDERR: {(result.stderr or '')[:500]}", flush=True)
        raise SystemExit(f"proposer failed in iter {iteration}")
    if not PENDING_EVAL.exists():
        raise SystemExit(f"proposer did not write {PENDING_EVAL}")
    pending = json.loads(PENDING_EVAL.read_text())
    print(f"  CANDIDATE: {pending['candidate']['name']}", flush=True)
    return pending


def run_eval(agent_version: str, ids_file: Path, parallel: int, label: str) -> Path:
    print(f"\n=== eval {agent_version} on {label} (parallel={parallel}) ===", flush=True)
    summary = ROOT / "traces" / f"gaia_{agent_version}__summary.jsonl"
    cmd = [
        sys.executable,
        str(ROOT / "run_benchmark.py"),
        "gaia",
        "--agent-version", agent_version,
        "--task-ids-file", str(ids_file),
        "--parallel", str(parallel),
    ]
    started = time.time()
    ret = subprocess.run(cmd, cwd=ROOT)
    print(f"  exit={ret.returncode}  elapsed={time.time() - started:.1f}s", flush=True)
    if ret.returncode != 0:
        raise SystemExit(f"run_benchmark.py exited with {ret.returncode}")
    if not summary.exists():
        raise SystemExit(f"expected summary not found: {summary}")
    return summary


def score_and_update(iteration: int, pending: dict, summary_path: Path) -> None:
    cand = pending["candidate"]
    cmd = [
        sys.executable,
        str(THIS_DIR / "scripts" / "score_candidate.py"),
        "--agent-name", cand["name"],
        "--summary-path", str(summary_path),
        "--iteration", str(iteration),
        "--hypothesis", cand.get("hypothesis", ""),
        "--changes", cand.get("changes", ""),
        "--logs-dir", str(LOGS),
    ]
    # Robust skill emits a plugin manifest; the baseline skill does not. Pass it
    # through verbatim when present so the durability axis is recorded; absent
    # for the baseline, leaving its scoring path untouched.
    plugin = cand.get("plugin")
    if plugin:
        cmd += ["--plugin-json", json.dumps(plugin)]
    ret = subprocess.run(cmd, cwd=ROOT)
    if ret.returncode != 0:
        raise SystemExit(f"score_candidate.py exited with {ret.returncode}")
    PENDING_EVAL.unlink()


def final_test_eval(parallel: int) -> None:
    global EVOLUTION_SUMMARY
    if not EVOLUTION_SUMMARY.exists():
        print("no evolution summary yet; skipping final test eval"); return
    # Champion = the candidate with the highest train-30 score (ties broken by
    # latest iteration). A single deployable agent, not the patchwork frontier.
    best = None
    best_key = (-1, -1)  # (train_correct, iteration)
    for line in EVOLUTION_SUMMARY.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        agent = rec.get("agent", "")
        if not agent.startswith("mh_iter"):
            continue
        correct = (rec.get("train_score") or {}).get("correct", 0)
        key = (correct, int(rec.get("iteration", 0)))
        if key > best_key:
            best_key = key
            best = agent
    if best is None:
        print("no mh candidate scored yet; skipping test eval"); return
    print(f"\n=== final test eval: {best} (train {best_key[0]}/30, iter {best_key[1]}) on test-135 ===", flush=True)
    summary = run_eval(best, TEST_IDS_FILE, parallel, "test-135")
    rows = [json.loads(l) for l in summary.read_text().splitlines() if l.strip()]
    n = len(rows); correct = sum(1 for r in rows if (r.get("score") or 0) > 0)
    print(f"\nTEST RESULT: {best} → {correct}/{n} = {correct / max(n, 1):.4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--train-parallel", type=int, default=4)
    p.add_argument("--test-parallel", type=int, default=4)
    p.add_argument("--final-test", action="store_true",
                   help="after iterations, run frontier candidate on test-135")
    p.add_argument("--proposer-only", action="store_true",
                   help="run proposer for 1 iter but skip eval (for skill smoke-test)")
    p.add_argument("--skill", default=DEFAULT_SKILL_NAME,
                   help=f"skill name under {SKILLS_PARENT.relative_to(ROOT)}/ (default: {DEFAULT_SKILL_NAME})")
    p.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS,
                   help="state dir for frontier_val.json / evolution_summary.jsonl / pending_eval.json / sessions")
    p.add_argument("--candidate-slug-prefix", default="",
                   help="inject prefix into candidate name (e.g. 'robust' -> mh_iter{N}_robust_<slug>) for parallel A/B runs")
    args = p.parse_args()

    skill_dir = SKILLS_PARENT / args.skill
    if not (skill_dir / "SKILL.md").exists():
        raise SystemExit(f"skill not found: {skill_dir}/SKILL.md")

    global LOGS, PENDING_EVAL, FRONTIER_VAL, EVOLUTION_SUMMARY
    LOGS = args.logs_dir.resolve()
    LOGS.mkdir(parents=True, exist_ok=True)
    PENDING_EVAL = LOGS / "pending_eval.json"
    FRONTIER_VAL = LOGS / "frontier_val.json"
    EVOLUTION_SUMMARY = LOGS / "evolution_summary.jsonl"

    proposer_log_dir = LOGS / "proposer_sessions"
    proposer_log_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.iterations):
        iteration = _current_iteration()
        if PENDING_EVAL.exists():
            print(f"[warn] {PENDING_EVAL} already exists; will be overwritten by proposer")
        pending = run_proposer(iteration, proposer_log_dir, args.skill, args.candidate_slug_prefix)
        if args.proposer_only:
            print("--proposer-only set; stopping after proposer")
            return
        cand_name = pending["candidate"]["agent_version_arg"]
        summary = run_eval(cand_name, TRAIN_IDS_FILE, args.train_parallel, "train-30")
        score_and_update(iteration, pending, summary)

    if args.final_test:
        final_test_eval(args.test_parallel)


if __name__ == "__main__":
    main()
