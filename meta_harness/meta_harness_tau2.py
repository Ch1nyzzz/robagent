"""Outer evolution loop for tau2-bench harness search (retail domain).

Sibling of meta_harness.py, wired for tau2 instead of GAIA. Each iteration:
  1. Spawn a headless coding agent with the robust-harness-tau2 skill. It reads
     frontier/evolution state + failed tau2 simulations, writes a new candidate
     under agent_tau2/mh_tau2_iter<N>_<slug>/, emits pending_eval.json.
  2. Evaluate the candidate on the tau2 retail train-30 subset via tau2_runner.py
     (library-mode official tau2 simulator).
  3. Score, update frontier_val.json, append to evolution_summary.jsonl.

After all iterations, optionally re-score the champion on the test-84 split.

Usage:
  python meta_harness/meta_harness_tau2.py --iterations 5
  python meta_harness/meta_harness_tau2.py --iterations 30 --final-test
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

import claude_wrapper  # noqa: E402

DEFAULT_LOGS = THIS_DIR / "logs_tau2"
TRAIN_IDS = THIS_DIR / "tau2_train_task_ids.txt"
TEST_IDS = THIS_DIR / "tau2_test_task_ids.txt"
SKILLS_PARENT = THIS_DIR / ".claude" / "skills"
DEFAULT_SKILL = "robust-harness-tau2"
DOMAIN = "banking_knowledge"

# Set in main() from --logs-dir.
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
    for line in EVOLUTION_SUMMARY.read_text().splitlines():
        try:
            last = max(last, int(json.loads(line).get("iteration", 0)))
        except json.JSONDecodeError:
            continue
    return last + 1


def _proposer_prompt(iteration: int, skill_name: str, slug_prefix: str) -> str:
    name_pattern = (
        f"mh_tau2_iter{iteration}_{slug_prefix}_<slug>" if slug_prefix
        else f"mh_tau2_iter{iteration}_<slug>"
    )
    return f"""You are a tau2-bench harness proposer. Follow the SKILL.md ({skill_name}).

Iteration: {iteration}
Working directory: {ROOT}

State files (relative to working directory):
  - Frontier per-task best:   {FRONTIER_VAL.relative_to(ROOT)}
  - Prior iteration summary:  {EVOLUTION_SUMMARY.relative_to(ROOT)}
  - Output pending eval to:   {PENDING_EVAL.relative_to(ROOT)}
  - Train task ids:           meta_harness/tau2_train_task_ids.txt (30 banking_knowledge tasks)
  - Baseline candidate (v0):  agent_tau2/v0/agent.py  (stock tau2 LLMAgent)
  - Prior candidates:         agent_tau2/mh_tau2_iter*/  (if any)
  - tau2 simulations:         tau2-runs/meta/<candidate>__{DOMAIN}.json/results.json

Steps:
  1. Read frontier_val.json and the last few entries of evolution_summary.jsonl.
  2. Pick 4-6 train task_ids the frontier still fails (score == 0). Read their
     newest tau2 simulation (messages + reward_info) under tau2-runs/meta/.
  3. Form ONE hypothesis tied to a mechanism present in >=3 of those sims.
  4. Create agent_tau2/{name_pattern}/{{__init__.py, agent.py, +helper modules}}.
     agent.py must export build_agent(tools, domain_policy, **kwargs); put
     deterministic logic (calculators, filters, policy engines) in helper
     modules that agent.py imports.
  5. Validate: `python -c "from agent_tau2.{name_pattern}.agent import build_agent; print('ok')"`.
  6. Write {PENDING_EVAL.relative_to(ROOT)} with the JSON schema from SKILL.md.
  7. Print one final line: CANDIDATE: <candidate_name>

CRITICAL:
  - Do NOT run benchmarks. Do NOT modify tau2-bench-src/, tau2_runner.py,
    meta_harness/*, agent_tau2/v0/, or any earlier agent_tau2/mh_tau2_iter*/.
  - No task-specific hardcoding. No customer names / order ids in code.
"""


def run_proposer(iteration: int, log_dir: Path, skill_name: str, slug_prefix: str) -> dict:
    print(f"\n=== tau2 iter {iteration}: proposer starting (skill={skill_name}) ===", flush=True)
    # The proposer (Claude Opus) occasionally hits a transient API 529/overload.
    # Retry with backoff instead of aborting the whole multi-hour run.
    max_attempts = 6
    for attempt in range(1, max_attempts + 1):
        result = claude_wrapper.run(
            prompt=_proposer_prompt(iteration, skill_name, slug_prefix),
            model=PROPOSER_MODEL,
            allowed_tools=PROPOSER_TOOLS,
            cwd=str(ROOT),
            log_dir=str(log_dir),
            name=f"tau2_iter{iteration}_proposer_{skill_name}",
            skills=[skill_name],
            skill_dir=str(SKILLS_PARENT),
            disable_skills=True,
            disable_mcp=True,
            progress=True,
        )
        print(f"  attempt {attempt}/{max_attempts}: exit={result.exit_code} "
              f"cost=${result.cost_usd:.4f} dur={result.duration_seconds:.0f}s", flush=True)
        if result.exit_code == 0 and PENDING_EVAL.exists():
            pending = json.loads(PENDING_EVAL.read_text())
            print(f"  CANDIDATE: {pending['candidate']['name']}", flush=True)
            return pending
        if attempt < max_attempts:
            wait = min(120 * attempt, 600)
            print(f"  proposer attempt {attempt} failed (transient API error?); "
                  f"retry in {wait}s", flush=True)
            time.sleep(wait)
    raise SystemExit(f"proposer failed in iter {iteration} after {max_attempts} attempts")


def run_eval(candidate: str, ids_file: Path, parallel: int, label: str) -> Path:
    print(f"\n=== tau2 eval {candidate} on {label} ===", flush=True)
    env = os.environ.copy()
    env.setdefault("TAU2_DATA_ROOT", str(ROOT / "tau2-bench-src" / "data" / "tau2"))
    cmd = [
        sys.executable, str(ROOT / "tau2_runner.py"),
        "--candidate", candidate,
        "--domain", DOMAIN,
        "--task-ids-file", str(ids_file),
        "--max-concurrency", str(parallel),
    ]
    started = time.time()
    ret = subprocess.run(cmd, cwd=ROOT, env=env)
    print(f"  exit={ret.returncode} elapsed={time.time()-started:.0f}s", flush=True)
    if ret.returncode != 0:
        raise SystemExit(f"tau2_runner exited {ret.returncode}")
    summary = ROOT / "traces" / f"tau2_{candidate}__summary.jsonl"
    if not summary.exists():
        raise SystemExit(f"expected summary missing: {summary}")
    return summary


def score_and_update(iteration: int, pending: dict, summary_path: Path) -> None:
    cand = pending["candidate"]
    cmd = [
        sys.executable, str(THIS_DIR / "scripts" / "score_candidate.py"),
        "--agent-name", cand["name"],
        "--summary-path", str(summary_path),
        "--iteration", str(iteration),
        "--hypothesis", cand.get("hypothesis", ""),
        "--changes", cand.get("changes", ""),
        "--logs-dir", str(LOGS),
        "--train-file", str(TRAIN_IDS),
    ]
    # robust-harness-tau2 emits a plugin manifest (durability axis); forward it
    # verbatim. Absent for the baseline skill — leaves its scoring untouched.
    plugin = cand.get("plugin")
    if plugin:
        cmd += ["--plugin-json", json.dumps(plugin)]
    if subprocess.run(cmd, cwd=ROOT).returncode != 0:
        raise SystemExit("score_candidate.py failed")
    PENDING_EVAL.unlink()


def final_test_eval(parallel: int) -> None:
    if not EVOLUTION_SUMMARY.exists():
        print("no evolution summary; skipping test eval"); return
    best, best_key = None, (-1, -1)
    for line in EVOLUTION_SUMMARY.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        agent = rec.get("agent", "")
        if not agent.startswith("mh_tau2_iter"):
            continue
        key = ((rec.get("train_score") or {}).get("correct", 0), int(rec.get("iteration", 0)))
        if key > best_key:
            best_key, best = key, agent
    if best is None:
        print("frontier still v0; skipping test eval"); return
    print(f"\n=== tau2 final test: {best} (train {best_key[0]}/30) on test ===", flush=True)
    summary = run_eval(best, TEST_IDS, parallel, "test")
    rows = [json.loads(l) for l in summary.read_text().splitlines() if l.strip()]
    n = len(rows)
    correct = sum(1 for r in rows if (r.get("score") or 0) > 0)
    print(f"\nTEST RESULT: {best} -> {correct}/{n} = {correct/max(n,1):.4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--train-parallel", type=int, default=8)
    p.add_argument("--test-parallel", type=int, default=8)
    p.add_argument("--final-test", action="store_true")
    p.add_argument("--proposer-only", action="store_true")
    p.add_argument("--skill", default=DEFAULT_SKILL,
                   help=f"skill under {SKILLS_PARENT.relative_to(ROOT)}/ (default: {DEFAULT_SKILL})")
    p.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS,
                   help="state dir for frontier_val / evolution_summary / pending_eval / sessions")
    p.add_argument("--candidate-slug-prefix", default="",
                   help="inject prefix into candidate name for parallel A/B runs")
    args = p.parse_args()

    if not (SKILLS_PARENT / args.skill / "SKILL.md").exists():
        raise SystemExit(f"skill not found: {SKILLS_PARENT / args.skill}/SKILL.md")

    global LOGS, PENDING_EVAL, FRONTIER_VAL, EVOLUTION_SUMMARY
    LOGS = args.logs_dir.resolve()
    LOGS.mkdir(parents=True, exist_ok=True)
    PENDING_EVAL = LOGS / "pending_eval.json"
    FRONTIER_VAL = LOGS / "frontier_val.json"
    EVOLUTION_SUMMARY = LOGS / "evolution_summary.jsonl"

    proposer_log_dir = LOGS / "proposer_sessions"
    proposer_log_dir.mkdir(parents=True, exist_ok=True)

    for _ in range(args.iterations):
        iteration = _current_iteration()
        pending = run_proposer(iteration, proposer_log_dir, args.skill, args.candidate_slug_prefix)
        if args.proposer_only:
            print("--proposer-only; stopping"); return
        summary = run_eval(pending["candidate"]["name"], TRAIN_IDS,
                           args.train_parallel, "train-30")
        score_and_update(iteration, pending, summary)

    if args.final_test:
        final_test_eval(args.test_parallel)


if __name__ == "__main__":
    main()
