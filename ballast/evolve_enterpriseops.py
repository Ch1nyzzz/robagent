"""Outer evolution loop for enterpriseops (domain = calendar | itsm).

Directory-as-frontier model
---------------------------
The agent + its components live entirely under one directory:

    agent/enterpriseops/
        v0/                       (frozen control, never modified after init)
            agent.py
            runtime/              (dispatcher, types, policy)
            components_calendar/
            components_itsm/
        v_calendar_1/             (one per accepted calendar iter)
        v_itsm_1/                 (one per accepted itsm iter)
        current_calendar → v0 | v_calendar_N   (symlink to latest accepted)
        current_itsm     → v0 | v_itsm_N

Per-iteration flow for ONE domain:
  1. Resolve current frontier dir = readlink(current_<domain>).
  2. Pick next iteration N; clone via `cp -r current_dir agent/enterpriseops/v_<domain>_<N>`.
  3. Run proposer; it edits anything inside the new dir (agent.py, runtime/,
     components_<domain>/, new tools/, …). Outputs pending_eval.json.
  4. Run candidate on train via run_enterpriseops_baseline.py --agent-dir <new_dir>.
  5. Champion-gate: candidate train n_correct ≥ last accepted n_correct?
       accept → repoint symlink, snapshot, commit
       reject → rmtree new_dir; symlink unchanged
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(ROOT))

import claude_wrapper  # noqa: E402

SKILLS_PARENT  = THIS_DIR / ".claude" / "skills"
DEFAULT_SKILL  = "component-harness-enterpriseops"

# Upstream EnterpriseOps-Gym venv. The baseline runner imports
# langchain_deepseek / datasets which only live in this venv.
UPSTREAM_DIR = ROOT / "third_party" / "EnterpriseOps-Gym"
UPSTREAM_PY = Path(os.environ.get(
    "ENTERPRISEOPS_UPSTREAM_PY",
    str(UPSTREAM_DIR / ".venv" / "bin" / "python"),
))

# Agent dir layout: agent/enterpriseops/{v0, v_<domain>_<N>, current_<domain>}
AGENT_PARENT = ROOT / "agent" / "enterpriseops"


def _train_split_for(domain: str) -> Path:
    return THIS_DIR / f"enterpriseops_{domain}_train_task_ids.txt"


def _test_split_for(domain: str) -> Path:
    return THIS_DIR / f"enterpriseops_{domain}_test_task_ids.txt"


def _current_symlink(domain: str) -> Path:
    return AGENT_PARENT / f"current_{domain}"


def _v_dir(domain: str, n: int) -> Path:
    """Path to v_<domain>_<N>. N=0 collapses to the shared v0/."""
    if n == 0:
        return AGENT_PARENT / "v0"
    return AGENT_PARENT / f"v_{domain}_{n}"


PROPOSER_MODEL = os.environ.get("MH_PROPOSER_MODEL", "opus")
PROPOSER_TOOLS = ["Read", "Glob", "Grep", "Edit", "Write", "Bash"]

# Set by main() per domain:
LOGS: Path
PENDING_EVAL: Path
FRONTIER_VAL: Path
EVOLUTION_SUMMARY: Path
DOMAIN: str
TRAIN_PARALLEL: int
PROVIDER: str


def _logs_for(domain: str) -> Path:
    d = THIS_DIR / f"logs_components_enterpriseops_{domain}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _current_iteration() -> int:
    """Next iter number. Reads max iteration in evolution_summary.jsonl, +1."""
    if not EVOLUTION_SUMMARY.exists():
        return 1
    last = 0
    for line in EVOLUTION_SUMMARY.read_text().splitlines():
        try:
            last = max(last, int(json.loads(line).get("iteration", 0)))
        except json.JSONDecodeError:
            continue
    return last + 1


# ----------------------------------------------------------------------------
# Frontier directory load / save.
# ----------------------------------------------------------------------------


def _resolve_frontier_dir(domain: str) -> Path:
    """Return the absolute path the `current_<domain>` symlink points at.

    If the symlink is missing (first run on a fresh checkout), point it at
    v0 lazily.
    """
    link = _current_symlink(domain)
    if not link.is_symlink() and not link.exists():
        link.symlink_to("v0")
    target = (link.parent / os.readlink(link)).resolve()
    if not target.exists():
        raise SystemExit(
            f"current_{domain} symlink points at {target} which does not exist"
        )
    return target


def _repoint_frontier(domain: str, new_dir: Path) -> None:
    link = _current_symlink(domain)
    if link.is_symlink() or link.exists():
        link.unlink()
    # Store as relative symlink (target is sibling of the symlink).
    link.symlink_to(new_dir.name)


# ----------------------------------------------------------------------------
# Proposer.
# ----------------------------------------------------------------------------


def _proposer_prompt(iteration: int, skill_name: str, next_dir: Path,
                     prev_dir: Path) -> str:
    next_rel = next_dir.relative_to(ROOT)
    prev_rel = prev_dir.relative_to(ROOT)
    train_split = _train_split_for(DOMAIN).relative_to(ROOT)
    return f"""You are an EnterpriseOps agent proposer. Follow the SKILL.md ({skill_name}).

Iteration: {iteration}
Domain: {DOMAIN}
Working directory: {ROOT}

Directory-as-frontier model
---------------------------
The previous frontier lives at:
    {prev_rel}/

A pristine copy of it has already been made at:
    {next_rel}/

This is YOUR working directory for this iteration. Anything you change inside
{next_rel}/ is the candidate. If accepted, the symlink agent/enterpriseops/current_{DOMAIN}
gets repointed to it; if rejected, the entire directory is deleted.

You may edit ANY file in {next_rel}/:
  - {next_rel}/agent.py            (capability: SYSTEM_PROMPT, tool registration,
                                    orchestrator subclass, run_task body)
  - {next_rel}/runtime/            (dispatcher/policy/types — usually leave alone)
  - {next_rel}/components_{DOMAIN}/ (stabilization hooks: add/edit/delete .py files)

State files (relative to working directory):
  - Frontier per-task best:           ballast/logs_components_enterpriseops_{DOMAIN}/frontier_val.json
  - Prior iteration summary:          ballast/logs_components_enterpriseops_{DOMAIN}/evolution_summary.jsonl
  - Output pending eval to:           {PENDING_EVAL.relative_to(ROOT)}
  - Train task ids:                   {train_split}
  - Per-task baselines:               ballast/logs_components_enterpriseops_{DOMAIN}/v0__train__results.json
  - Component fire trace (per iter):  .component-state-enterpriseops/<run_tag>/fired.jsonl

Steps:
  1. Read frontier_val.json's per_task block to find tasks the frontier scores 0 on.
  2. Read the v0 per-task traces under ballast/logs_components_enterpriseops_{DOMAIN}/v0__train__traces/
     to identify failure modes (verifier check name, tool-call shape, missing SQL action).
  3. Form ONE hypothesis tied to a mechanism present in >=3 of those failures.
  4. Decide path:
       capability path  → edit {next_rel}/agent.py
                          (e.g. extend SYSTEM_PROMPT with an API cheatsheet,
                          register a new tool, alter the orchestrator).
       stabilization path → add/edit/delete files under
                          {next_rel}/components_{DOMAIN}/ (your usual hook).
     A single iteration MAY combine both if the hypothesis logically requires it.
  5. Validate component loadability (if you wrote one):
       python -c "
       import importlib; m = importlib.import_module('{ '.'.join(next_rel.parts) }.runtime')
       comps = m.load_components_from_dir('{next_rel}/components_{DOMAIN}')
       print(sorted(c.name for c in comps))
       "
  6. Write {PENDING_EVAL.relative_to(ROOT)}:
       {{
         "candidate": {{
           "name": "candidate_iter{iteration}_<slug>",
           "hypothesis": "...",
           "changes": "...",                        # natural language summary
           "edited_files": ["{next_rel}/agent.py",  # explicit list, audit trail
                            "{next_rel}/components_{DOMAIN}/iter{iteration}_<slug>.py"],
           "trust": {{
             "evidence_anchor": "...",
             "blast_radius": "local|workflow|agent",  # agent = touched agent.py
             "rollback_when": "...",
             "out_of_evidence_probe": "..."
           }}
         }}
       }}
  7. Print one final line: CANDIDATE: <candidate_name>

CRITICAL:
  - Do NOT run benchmarks. The outer loop scores.
  - Do NOT modify anything outside {next_rel}/. In particular, the
    frozen v0 dir and other v_{DOMAIN}_<M> dirs are read-only.
  - Do NOT modify third_party/EnterpriseOps-Gym/, ballast/, or
    other domains' v_<OTHER>_<M> dirs / components_<other>/.
  - **The target inference model is LOCKED** to the configured provider
    (default deepseek-chat) via agent.llm.chat. Components may call
    ctx.chat(...) for a same-model sub-LLM; do NOT call any other API.
  - No task-specific hardcoding. No entity names or test-set IDs in code.

Reminders about the component runtime (unchanged from prior iterations):
  - Component file naming: any `*.py` (excluding `_*.py`) in components_{DOMAIN}/
    is loaded as active. Replace by overwriting the same filename; disable by deleting.
  - Lifecycle events (Tier-1, all fire in enterpriseops): session_start,
    task_received, pre_prompt_build, pre_context_build, pre_agent_construct,
    pre_llm_turn, pre_llm_request, post_llm_response, post_llm_response_raw,
    on_length_truncation, on_empty_response, on_no_tool_call_emitted,
    pre_tool_arg_validation, pre_tool_use, post_tool_use, post_tool_result_raw,
    on_tool_error, on_explicit_terminate, pre_final_emit, session_end.
  - ctx.chat(messages, max_tokens=..., temperature=...) → sub-LLM (locked model).
  - ctx.emit("name") / ctx.emit_upstream(...) → publish/subscribe between components.
"""


def run_proposer(iteration: int, log_dir: Path, skill_name: str,
                 next_dir: Path, prev_dir: Path) -> dict:
    print(f"\n=== enterpriseops/{DOMAIN} iter {iteration}: proposer starting "
          f"(skill={skill_name}, working_dir={next_dir.relative_to(ROOT)}) ===",
          flush=True)
    max_attempts = 6
    for attempt in range(1, max_attempts + 1):
        result = claude_wrapper.run(
            prompt=_proposer_prompt(iteration, skill_name, next_dir, prev_dir),
            model=PROPOSER_MODEL,
            allowed_tools=PROPOSER_TOOLS,
            cwd=str(ROOT),
            log_dir=str(log_dir),
            name=f"enterpriseops_{DOMAIN}_iter{iteration}_proposer",
            skills=[skill_name],
            skill_dir=str(SKILLS_PARENT),
            disable_skills=True,
            disable_mcp=True,
            progress=True,
            docker_skill=skill_name,
            docker_domain=DOMAIN,
            docker_container_name=f"ballast-proposer-enterpriseops-{DOMAIN}-"
                                  f"iter{iteration}-{int(time.time())}",
        )
        print(f"  attempt {attempt}/{max_attempts}: exit={result.exit_code} "
              f"cost=${result.cost_usd:.4f} dur={result.duration_seconds:.0f}s",
              flush=True)
        if result.exit_code == 0 and PENDING_EVAL.exists():
            pending = json.loads(PENDING_EVAL.read_text())
            print(f"  CANDIDATE: {pending['candidate']['name']}", flush=True)
            return pending
        if attempt < max_attempts:
            wait = min(120 * attempt, 600)
            print(f"  proposer attempt {attempt} failed; retry in {wait}s",
                  flush=True)
            time.sleep(wait)
    raise SystemExit(f"proposer failed in iter {iteration} after {max_attempts} attempts")


# ----------------------------------------------------------------------------
# Eval + scoring.
# ----------------------------------------------------------------------------


def _baseline_subprocess_cmd(*, agent_name: str, iteration: int,
                             agent_dir: Path, subset: str,
                             hypothesis: str, changes: str,
                             plugin_payload: dict | None) -> list[str]:
    if not UPSTREAM_PY.exists():
        raise SystemExit(
            f"upstream venv python not found at {UPSTREAM_PY}. "
            f"Did you `cd third_party/EnterpriseOps-Gym && uv sync --extra all`? "
            f"Or set ENTERPRISEOPS_UPSTREAM_PY=... to point at a working interp."
        )
    cmd: list[str] = [
        str(UPSTREAM_PY), "-u",
        str(THIS_DIR / "scripts" / "run_enterpriseops_baseline.py"),
        "--domain", DOMAIN,
        "--agent-name", agent_name,
        "--iteration", str(iteration),
        "--train-split", str(_train_split_for(DOMAIN)),
        "--test-split", str(_test_split_for(DOMAIN)),
        "--subsets", subset,
        "--agent-dir", str(agent_dir),
        "--concurrency", str(TRAIN_PARALLEL),
        "--provider", PROVIDER,
        "--hypothesis", hypothesis,
        "--changes", changes,
    ]
    if plugin_payload is not None:
        cmd += ["--plugin-json", json.dumps(plugin_payload)]
    return cmd


def _baseline_env() -> dict[str, str]:
    """Inherit current env + augment PYTHONPATH to include the upstream tree."""
    env = os.environ.copy()
    extra_paths = [str(ROOT), str(UPSTREAM_DIR)]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(p for p in [*extra_paths, existing] if p)
    return env


def run_eval(agent_dir: Path, iteration: int, agent_name: str,
             hypothesis: str, changes: str, plugin_payload: dict) -> dict:
    train_split = _train_split_for(DOMAIN)
    n_train = sum(1 for ln in train_split.read_text().splitlines() if ln.strip())
    comp_dir = agent_dir / f"components_{DOMAIN}"
    active = sorted(p.stem for p in comp_dir.glob("*.py")
                    if not p.name.startswith("_")) if comp_dir.exists() else []
    print(f"\n=== enterpriseops eval iter {iteration} agent_dir={agent_dir.relative_to(ROOT)} "
          f"active_components={active} on {DOMAIN} train-{n_train} ===", flush=True)
    cmd = _baseline_subprocess_cmd(
        agent_name=agent_name,
        iteration=iteration,
        agent_dir=agent_dir,
        subset="train",
        hypothesis=hypothesis,
        changes=changes,
        plugin_payload=plugin_payload,
    )
    started = time.time()
    ret = subprocess.run(cmd, cwd=ROOT, env=_baseline_env())
    print(f"  exit={ret.returncode} elapsed={time.time()-started:.0f}s",
          flush=True)
    if ret.returncode != 0:
        raise SystemExit(f"run_enterpriseops_baseline.py exited {ret.returncode}")

    last_line = ""
    for line in reversed(EVOLUTION_SUMMARY.read_text().splitlines()):
        if line.strip():
            last_line = line
            break
    if not last_line:
        raise SystemExit("evolution_summary.jsonl missing/empty after eval")
    return json.loads(last_line)


def _previous_accepted_correct() -> int:
    """Champion-gate: train_score.n_correct of the most recent accepted entry."""
    if not EVOLUTION_SUMMARY.exists():
        return 0
    for line in reversed(EVOLUTION_SUMMARY.read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("accepted") is True or int(row.get("iteration", -1)) == 0:
            return int(row.get("train_score", {}).get("n_correct", 0))
    return 0


def _score_and_update(iteration: int, candidate_row: dict,
                      prev_dir: Path, next_dir: Path) -> bool:
    prev_correct = _previous_accepted_correct()
    cand_correct = int(candidate_row.get("train_score", {}).get("n_correct", 0))
    accepted = cand_correct >= prev_correct

    lines = [ln for ln in EVOLUTION_SUMMARY.read_text().splitlines() if ln.strip()]
    last = json.loads(lines[-1])
    last["accepted"] = accepted
    last["agent_dir"] = str(next_dir.relative_to(ROOT))
    last["agent_dir_after"] = str(
        (next_dir if accepted else prev_dir).relative_to(ROOT)
    )
    last["previous_accepted_correct"] = prev_correct
    lines[-1] = json.dumps(last, default=str)
    EVOLUTION_SUMMARY.write_text("\n".join(lines) + "\n")

    if accepted:
        _repoint_frontier(DOMAIN, next_dir)
        print(f"  accepted (train {cand_correct} ≥ {prev_correct}); "
              f"frontier → {next_dir.relative_to(ROOT)}", flush=True)
    else:
        shutil.rmtree(next_dir)
        print(f"  REJECTED (train {cand_correct} < {prev_correct}); "
              f"removed {next_dir.relative_to(ROOT)}", flush=True)
    return accepted


def final_test_eval() -> None:
    """Run the accepted frontier dir on the held-out test split once."""
    frontier_dir = _resolve_frontier_dir(DOMAIN)
    print(f"\n=== final test eval on {DOMAIN}: agent_dir="
          f"{frontier_dir.relative_to(ROOT)} ===", flush=True)
    iteration = _current_iteration() - 1
    cmd = _baseline_subprocess_cmd(
        agent_name=f"frontier_iter{iteration}",
        iteration=iteration,
        agent_dir=frontier_dir,
        subset="test",
        hypothesis="final test eval of accepted frontier",
        changes="",
        plugin_payload=None,
    )
    started = time.time()
    ret = subprocess.run(cmd, cwd=ROOT, env=_baseline_env())
    print(f"  exit={ret.returncode} elapsed={time.time()-started:.0f}s",
          flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True, choices=["calendar", "itsm"])
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--train-parallel", type=int, default=4,
                   help="upstream baseline --concurrency")
    p.add_argument("--provider", default="deepseek",
                   choices=["deepseek", "openai", "anthropic"])
    p.add_argument("--final-test", action="store_true")
    p.add_argument("--proposer-only", action="store_true")
    p.add_argument("--skill", default=DEFAULT_SKILL)
    args = p.parse_args()

    global LOGS, PENDING_EVAL, FRONTIER_VAL, EVOLUTION_SUMMARY
    global DOMAIN, TRAIN_PARALLEL, PROVIDER
    DOMAIN = args.domain
    TRAIN_PARALLEL = args.train_parallel
    PROVIDER = args.provider
    LOGS = _logs_for(DOMAIN).resolve()
    PENDING_EVAL = LOGS / "pending_eval.json"
    FRONTIER_VAL = LOGS / "frontier_val.json"
    EVOLUTION_SUMMARY = LOGS / "evolution_summary.jsonl"

    if not (SKILLS_PARENT / args.skill / "SKILL.md").exists():
        raise SystemExit(f"skill not found: {SKILLS_PARENT / args.skill}/SKILL.md")
    if not _train_split_for(DOMAIN).exists():
        raise SystemExit(f"train split missing: {_train_split_for(DOMAIN)}")

    proposer_log_dir = LOGS / "proposer_sessions"
    proposer_log_dir.mkdir(parents=True, exist_ok=True)

    for _ in range(args.iterations):
        iteration = _current_iteration()
        prev_dir = _resolve_frontier_dir(DOMAIN)
        next_dir = _v_dir(DOMAIN, iteration)
        if next_dir.exists():
            raise SystemExit(
                f"next iter dir already exists: {next_dir}; clean it up first"
            )
        print(f"  cloning {prev_dir.relative_to(ROOT)} → "
              f"{next_dir.relative_to(ROOT)}", flush=True)
        shutil.copytree(prev_dir, next_dir, symlinks=True)

        try:
            pending = run_proposer(iteration, proposer_log_dir, args.skill,
                                   next_dir, prev_dir)
        except BaseException:
            shutil.rmtree(next_dir, ignore_errors=True)
            raise
        if args.proposer_only:
            print("--proposer-only; stopping (leaving next_dir for inspection)")
            return

        cand = pending["candidate"]
        plugin_payload = {
            "candidate": cand,
            "agent_dir": str(next_dir.relative_to(ROOT)),
        }
        candidate_row = run_eval(
            next_dir, iteration, cand["name"],
            cand.get("hypothesis", ""), cand.get("changes", ""),
            plugin_payload,
        )
        _score_and_update(iteration, candidate_row, prev_dir, next_dir)
        if PENDING_EVAL.exists():
            PENDING_EVAL.unlink()

    if args.final_test:
        final_test_eval()


if __name__ == "__main__":
    main()
