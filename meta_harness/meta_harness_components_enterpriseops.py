"""Outer evolution loop for component-harness-enterpriseops.

Per-iteration flow for ONE domain (calendar | itsm):
  1. Load current frontier workflow (meta_harness/workflows/enterpriseops_<domain>.yaml).
  2. Run proposer (Claude Code subagent w/ component-harness-enterpriseops skill).
     The proposer writes a single component file to
     agent/components_enterpriseops_<domain>/ and a `pending_eval.json`
     describing the workflow_patch + component metadata.
  3. Apply the patch → next_workflow; write the yaml.
  4. Run candidate on the train subset of the domain via run_enterpriseops_baseline.py
     (with --workflow / --components-dir CLI args pinned; train ids come from
     a per-domain txt file).
  5. Champion-gate: candidate.train_score.n_correct ≥ last_accepted.n_correct?
       accept → snapshot frontier_workflow.json + commit
       reject → rollback workflow yaml; delete added file or restore .bak
  6. Update last evolution_summary.jsonl row with accepted bool + workflow_after.

State per domain lives under
`meta_harness/logs_components_enterpriseops_<domain>/`.

Ported from `meta_harness_components_sopbench.py` (Phase 0 of the
event-based runtime migration). Differences from the sopbench loop:
  - Baseline subprocess uses the upstream EnterpriseOps-Gym uv-managed
    venv interpreter and sets PYTHONPATH to include the upstream tree.
  - Subprocess args are CLI (`--workflow`, `--components-dir`,
    `--train-split`) rather than env vars (`SOPBENCH_COMPONENT_*`).
  - Iteration `n_correct` is read from the evolution_summary.jsonl row
    the baseline script appends (same shape as sopbench:
    `train_score.n_correct`).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(ROOT))

import claude_wrapper  # noqa: E402

from agent.component_runtime_enterpriseops import (  # noqa: E402
    FrontierSnapshot,
    Patch,
    PatchOp,
    Workflow,
    apply_patch,
)

WORKFLOWS_DIR  = THIS_DIR / "workflows"
SKILLS_PARENT  = THIS_DIR / ".claude" / "skills"
DEFAULT_SKILL  = "component-harness-enterpriseops"

# Upstream EnterpriseOps-Gym venv. The baseline runner imports
# langchain_deepseek / datasets which only live in this venv.
UPSTREAM_DIR = ROOT / "third_party" / "EnterpriseOps-Gym"
UPSTREAM_PY = Path(os.environ.get(
    "ENTERPRISEOPS_UPSTREAM_PY",
    str(UPSTREAM_DIR / ".venv" / "bin" / "python"),
))


def _components_dir_for(domain: str) -> Path:
    """Per-domain components dir so parallel domain runs don't share files."""
    return ROOT / "agent" / f"components_enterpriseops_{domain}"


def _train_split_for(domain: str) -> Path:
    return THIS_DIR / f"enterpriseops_{domain}_train_task_ids.txt"


def _test_split_for(domain: str) -> Path:
    return THIS_DIR / f"enterpriseops_{domain}_test_task_ids.txt"


PROPOSER_MODEL = os.environ.get("MH_PROPOSER_MODEL", "opus")
PROPOSER_TOOLS = ["Read", "Glob", "Grep", "Edit", "Write", "Bash"]

# Set by main() per domain:
LOGS: Path
WORKFLOW_YAML: Path
PENDING_EVAL: Path
FRONTIER_VAL: Path
FRONTIER_WORKFLOW: Path
EVOLUTION_SUMMARY: Path
DOMAIN: str
TRAIN_PARALLEL: int
PROVIDER: str


def _logs_for(domain: str) -> Path:
    d = THIS_DIR / f"logs_components_enterpriseops_{domain}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _workflow_yaml_for(domain: str) -> Path:
    p = WORKFLOWS_DIR / f"enterpriseops_{domain}.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


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


# ----------------------------------------------------------------------------
# Frontier workflow load / save.
# ----------------------------------------------------------------------------


def _load_frontier_workflow() -> Workflow:
    if not WORKFLOW_YAML.exists():
        return Workflow()
    return Workflow.from_yaml(WORKFLOW_YAML)


def _save_frontier_snapshot(wf: Workflow, iteration: int) -> None:
    yaml_text = WORKFLOW_YAML.read_text() if WORKFLOW_YAML.exists() else ""
    snap = FrontierSnapshot(
        workflow_yaml_at_accept=yaml_text,
        active_names=list(wf.active_nodes()),
        accepted_at_iteration=iteration,
        accepted_at=dt.datetime.now(dt.timezone.utc).isoformat(),
    )
    snap.to_json(FRONTIER_WORKFLOW)


# ----------------------------------------------------------------------------
# Proposer.
# ----------------------------------------------------------------------------


def _proposer_prompt(iteration: int, skill_name: str) -> str:
    name_pattern = (
        f"component_enterpriseops_{DOMAIN}_iter{iteration}_<slug>"
    )
    comp_dir = _components_dir_for(DOMAIN).relative_to(ROOT)
    train_split = _train_split_for(DOMAIN).relative_to(ROOT)
    return f"""You are an EnterpriseOps component proposer. Follow the SKILL.md ({skill_name}).

Iteration: {iteration}
Domain: {DOMAIN}
Working directory: {ROOT}

State files (relative to working directory):
  - Domain workflow YAML (frontier):  {WORKFLOW_YAML.relative_to(ROOT)}
  - Frontier per-task best:           {FRONTIER_VAL.relative_to(ROOT)}
  - Frontier snapshot:                {FRONTIER_WORKFLOW.relative_to(ROOT)}
  - Prior iteration summary:          {EVOLUTION_SUMMARY.relative_to(ROOT)}
  - Output pending eval to:           {PENDING_EVAL.relative_to(ROOT)}
  - EnterpriseOpsAgent (READ-ONLY):   agent/enterpriseops_agent.py
  - Component runtime (READ-ONLY):    agent/component_runtime_enterpriseops/
  - THIS DOMAIN's components dir:     {comp_dir}/  (READ-ONLY; modify by
                                       reusing COMPONENT.name = replace_node)
  - Train task ids:                   {train_split}
  - Per-task baselines:               meta_harness/logs_components_enterpriseops_{DOMAIN}/v0__train__results.json
                                       (per-task verifier scores from the v0 run)
  - Component fire trace:             .component-state-enterpriseops/{DOMAIN}_iter<N-1>/fired.jsonl (absent on iter 1)

IMPORTANT: other domains' components live in their own dirs
(`components_enterpriseops_<other_domain>/`). Do NOT read them; isolation
is enforced via per-domain dirs + the docker proposer mount whitelist.
You only write to this domain's dir: `{comp_dir}/`.

Steps:
  1. Read workflows/enterpriseops_{DOMAIN}.yaml, frontier_workflow.json,
     frontier_val.json, and the last few entries of evolution_summary.jsonl.
  2. Pick 4-6 train task_ids where the current frontier scores 0 (look at
     `per_task` in frontier_val.json). Read their stored per-task result
     trace under logs_components_enterpriseops_{DOMAIN}/v0__train__traces/
     to identify the failure mode (verifier check name, tool-call shape,
     missing SQL action, etc.).
  3. Form ONE hypothesis tied to a mechanism present in >=3 of those failures.
  4. Choose ONE patch op: add_node | replace_node | disable_node.
       * add_node: create exactly one file at
         {comp_dir}/{name_pattern}.py exporting `COMPONENT: Component`.
       * replace_node: BEFORE editing, run
           cp {comp_dir}/<existing>.py {comp_dir}/<existing>.py.bak_iter{iteration}
         then overwrite (keep COMPONENT.name unchanged).
       * disable_node: write no file; reference the existing component
         name in the workflow_patch block.
  5. Validate registration + trust:
       python -c "
       from agent.component_runtime_enterpriseops import load_components_from_dir
       comps = load_components_from_dir('{comp_dir}', only=['<COMPONENT.name>'])
       assert any(c.name == '<COMPONENT.name>' for cs in comps.values() for c in cs)
       print('ok')
       "
  6. Write {PENDING_EVAL.relative_to(ROOT)} with this JSON schema:
       {{
         "candidate": {{
           "name": "mh_iter{iteration}_<slug>",
           "hypothesis": "...",
           "changes": "...",
           "component": {{
             "name": "...", "cls": "...", "mount": "...",
             "file": "{comp_dir}/...py",
             "trust": {{...}}
           }},
           "workflow_patch": {{
             "op": "add_node|replace_node|disable_node",
             "name": "<COMPONENT.name>",
             "file": "{comp_dir}/...py",
             "edges_in": [], "edges_out": []
           }}
         }}
       }}
  7. Print one final line: CANDIDATE: <candidate_name>

CRITICAL:
  - Do NOT run benchmarks. Do NOT modify third_party/EnterpriseOps-Gym/,
    meta_harness/scripts/run_enterpriseops_baseline.py,
    agent/enterpriseops_agent.py, agent/component_runtime_enterpriseops/.
  - **The target inference model is LOCKED** to the configured provider
    (default deepseek-chat). Do NOT call any other model API from a
    component; the upstream LangChain ReAct loop binds the model once
    per task and components only intercept around it.
  - **Per-domain dirs (no cross-domain reads)**: other domains have
    their own component dirs and logs. Do NOT read files under
    components_enterpriseops_<other_domain>/,
    logs_components_enterpriseops_<other_domain>/, or
    workflows/enterpriseops_<other_domain>.yaml. Stick to YOUR domain only.
  - No task-specific hardcoding. No entity names or test-set IDs in code.
"""


def run_proposer(iteration: int, log_dir: Path, skill_name: str) -> dict:
    print(f"\n=== enterpriseops/{DOMAIN} iter {iteration}: proposer starting "
          f"(skill={skill_name}) ===", flush=True)
    max_attempts = 6
    for attempt in range(1, max_attempts + 1):
        result = claude_wrapper.run(
            prompt=_proposer_prompt(iteration, skill_name),
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
            docker_container_name=f"robagent-proposer-enterpriseops-{DOMAIN}-"
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
                             workflow_path: Path, components_dir: Path,
                             subset: str,
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
        "--workflow", str(workflow_path),
        "--components-dir", str(components_dir),
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


def run_eval(workflow: Workflow, iteration: int, agent_name: str,
             hypothesis: str, changes: str, plugin_payload: dict) -> dict:
    active = list(workflow.active_nodes())
    train_split = _train_split_for(DOMAIN)
    n_train = sum(1 for ln in train_split.read_text().splitlines() if ln.strip())
    print(f"\n=== enterpriseops eval iter {iteration} active={active} on "
          f"{DOMAIN} train-{n_train} ===", flush=True)
    cmd = _baseline_subprocess_cmd(
        agent_name=agent_name,
        iteration=iteration,
        workflow_path=WORKFLOW_YAML,
        components_dir=_components_dir_for(DOMAIN),
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
    """Champion-gate: train_score.n_correct of the most recent accepted entry.

    Iter 0 (v0 seed) counts as accepted. Skips rejected candidates so their
    train_score never inflates the gate the next candidate must clear.
    """
    if not EVOLUTION_SUMMARY.exists():
        return 0
    for line in reversed(EVOLUTION_SUMMARY.read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("accepted") is True or int(row.get("iteration", -1)) == 0:
            return int(row.get("train_score", {}).get("n_correct", 0))
    return 0


def _backup_replace_file(patch: Patch, iteration: int) -> tuple[Path, Path] | None:
    if patch.op is not PatchOp.REPLACE_NODE or not patch.file:
        return None
    live = ROOT / patch.file
    bak = live.with_suffix(live.suffix + f".bak_iter{iteration}")
    if bak.exists():
        return live, bak
    return None


def _rollback_patch(prev_workflow: Workflow, patch: Patch,
                    backup: tuple[Path, Path] | None) -> None:
    prev_workflow.to_yaml(WORKFLOW_YAML)
    if patch.op is PatchOp.ADD_NODE and patch.file:
        f = ROOT / patch.file
        if f.exists():
            f.unlink()
            print(f"  cleaned up rejected component file: {patch.file}", flush=True)
    elif patch.op is PatchOp.REPLACE_NODE:
        if backup is not None:
            live, bak = backup
            live.write_text(bak.read_text())
            bak.unlink()
            print(f"  restored {live.relative_to(ROOT)} from .bak", flush=True)
        else:
            print(f"  WARN: replace_node rejected but no .bak found at "
                  f"{patch.file}.bak_iter<N>; file remains as-is", flush=True)


def _commit_patch(patch: Patch, backup: tuple[Path, Path] | None) -> None:
    if patch.op is PatchOp.REPLACE_NODE and backup is not None:
        _, bak = backup
        if bak.exists():
            bak.unlink()


def _score_and_update(iteration: int, candidate_row: dict,
                      prev_workflow: Workflow, next_workflow: Workflow,
                      patch: Patch, backup: tuple[Path, Path] | None) -> bool:
    prev_correct = _previous_accepted_correct()
    cand_correct = int(candidate_row.get("train_score", {}).get("n_correct", 0))
    accepted = cand_correct >= prev_correct

    # Re-read all lines, update last
    lines = [ln for ln in EVOLUTION_SUMMARY.read_text().splitlines() if ln.strip()]
    last = json.loads(lines[-1])
    last["accepted"] = accepted
    last["workflow_after"] = list(
        (next_workflow if accepted else prev_workflow).active_nodes()
    )
    last["previous_accepted_correct"] = prev_correct
    lines[-1] = json.dumps(last, default=str)
    EVOLUTION_SUMMARY.write_text("\n".join(lines) + "\n")

    if accepted:
        _commit_patch(patch, backup)
        _save_frontier_snapshot(next_workflow, iteration)
        print(f"  accepted (train {cand_correct} ≥ {prev_correct}); "
              f"frontier → {list(next_workflow.active_nodes())}", flush=True)
    else:
        print(f"  REJECTED (train {cand_correct} < {prev_correct}); "
              f"rolling back patch", flush=True)
        _rollback_patch(prev_workflow, patch, backup)
    return accepted


def final_test_eval() -> None:
    """Run the accepted frontier workflow on the held-out test split once."""
    wf = _load_frontier_workflow()
    print(f"\n=== final test eval on {DOMAIN}: workflow active="
          f"{list(wf.active_nodes())} ===", flush=True)
    iteration = _current_iteration() - 1
    cmd = _baseline_subprocess_cmd(
        agent_name=f"frontier_iter{iteration}",
        iteration=iteration,
        workflow_path=WORKFLOW_YAML,
        components_dir=_components_dir_for(DOMAIN),
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

    global LOGS, WORKFLOW_YAML, PENDING_EVAL, FRONTIER_VAL, FRONTIER_WORKFLOW
    global EVOLUTION_SUMMARY, DOMAIN, TRAIN_PARALLEL, PROVIDER
    DOMAIN = args.domain
    TRAIN_PARALLEL = args.train_parallel
    PROVIDER = args.provider
    LOGS = _logs_for(DOMAIN).resolve()
    WORKFLOW_YAML = _workflow_yaml_for(DOMAIN).resolve()
    PENDING_EVAL = LOGS / "pending_eval.json"
    FRONTIER_VAL = LOGS / "frontier_val.json"
    FRONTIER_WORKFLOW = LOGS / "frontier_workflow.json"
    EVOLUTION_SUMMARY = LOGS / "evolution_summary.jsonl"

    if not (SKILLS_PARENT / args.skill / "SKILL.md").exists():
        raise SystemExit(f"skill not found: {SKILLS_PARENT / args.skill}/SKILL.md")
    if not _train_split_for(DOMAIN).exists():
        raise SystemExit(f"train split missing: {_train_split_for(DOMAIN)}")

    if not WORKFLOW_YAML.exists():
        Workflow().to_yaml(WORKFLOW_YAML)
    if not FRONTIER_WORKFLOW.exists():
        _save_frontier_snapshot(_load_frontier_workflow(), 0)

    proposer_log_dir = LOGS / "proposer_sessions"
    proposer_log_dir.mkdir(parents=True, exist_ok=True)

    for _ in range(args.iterations):
        iteration = _current_iteration()
        prev_workflow = _load_frontier_workflow()

        pending = run_proposer(iteration, proposer_log_dir, args.skill)
        if args.proposer_only:
            print("--proposer-only; stopping")
            return

        cand = pending["candidate"]
        patch = Patch.from_dict(cand["workflow_patch"])
        backup = _backup_replace_file(patch, iteration)
        next_workflow = apply_patch(prev_workflow, patch)
        next_workflow.to_yaml(WORKFLOW_YAML)

        plugin_payload = {
            "component": cand.get("component"),
            "workflow_patch": cand.get("workflow_patch"),
        }
        candidate_row = run_eval(
            next_workflow, iteration, cand["name"],
            cand.get("hypothesis", ""), cand.get("changes", ""),
            plugin_payload,
        )
        _score_and_update(iteration, candidate_row,
                          prev_workflow, next_workflow, patch, backup)
        if PENDING_EVAL.exists():
            PENDING_EVAL.unlink()

    if args.final_test:
        final_test_eval()


if __name__ == "__main__":
    main()
