"""Outer evolution loop for component-harness-sopbench.

Per-iteration flow for ONE domain:
  1. Load current frontier workflow (meta_harness/workflows/sopbench_<domain>.yaml)
  2. Isolate stale paths so the proposer can't read other domains' state.
  3. Run proposer (Claude Code subagent w/ component-harness-sopbench skill).
     The proposer writes a single component file to
     agent/components_sopbench/ and a `pending_eval.json` describing the
     workflow_patch + component metadata.
  4. Apply the patch → next_workflow; write the yaml.
  5. Run candidate on the train subset of the domain via run_sopbench_baseline.py
     (with SOPBENCH_COMPONENT_WORKFLOW / SOPBENCH_COMPONENT_NAMES pinned).
  6. Champion-gate: candidate.train_score.n_correct ≥ last_accepted.n_correct?
       accept → snapshot frontier_workflow.json + commit
       reject → rollback workflow yaml; delete added file or restore .bak
  7. Update last evolution_summary.jsonl row with accepted bool + workflow_after.

State per domain lives under `meta_harness/logs_components_sopbench_<domain>/`.
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

from agent.component_runtime_sopbench import (  # noqa: E402
    FrontierSnapshot,
    Patch,
    PatchOp,
    Workflow,
    apply_patch,
)

WORKFLOWS_DIR  = THIS_DIR / "workflows"
SKILLS_PARENT  = THIS_DIR / ".claude" / "skills"
DEFAULT_SKILL  = "component-harness-sopbench"


def _components_dir_for(domain: str) -> Path:
    """Per-domain components dir so parallel domain runs don't share files."""
    return ROOT / "agent" / f"components_sopbench_{domain}"

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
TRAIN_SIZE: int
TRAIN_PARALLEL: int


def _logs_for(domain: str) -> Path:
    d = THIS_DIR / f"logs_components_sopbench_{domain}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _workflow_yaml_for(domain: str) -> Path:
    p = WORKFLOWS_DIR / f"sopbench_{domain}.yaml"
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
# Isolation: hide the OTHER domains' logs + components from the proposer so
# it stays focused on this domain's history. We do not hide GAIA / tau2 dirs
# because those are different runtimes the proposer is told to ignore.
# ----------------------------------------------------------------------------


def _other_domains_logs(this_domain: str) -> list[Path]:
    out: list[Path] = []
    for p in sorted(THIS_DIR.glob("logs_components_sopbench_*")):
        if p.name == f"logs_components_sopbench_{this_domain}":
            continue
        out.append(p)
    return out


def _other_domains_workflows(this_domain: str) -> list[Path]:
    out: list[Path] = []
    for p in sorted(WORKFLOWS_DIR.glob("sopbench_*.yaml")):
        if p.name == f"sopbench_{this_domain}.yaml":
            continue
        out.append(p)
    return out


# Proposer isolation is now enforced by running claude inside a docker
# container with a per-skill+per-domain whitelist of bind mounts (see
# meta_harness/_proposer_docker.py).  sopbench's previous "no-op
# contextmanager" relied on each domain having its own components dir /
# workflow yaml / logs dir; that file-layout invariant is still in force,
# but the container now adds a kernel-enforced filesystem boundary on top.


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
        f"component_sopbench_{DOMAIN}_iter{iteration}_<slug>"
    )
    comp_dir = _components_dir_for(DOMAIN).relative_to(ROOT)
    return f"""You are a SOP-Bench component proposer. Follow the SKILL.md ({skill_name}).

Iteration: {iteration}
Domain: {DOMAIN}
Working directory: {ROOT}

State files (relative to working directory):
  - Domain workflow YAML (frontier):  {WORKFLOW_YAML.relative_to(ROOT)}
  - Frontier per-task best:           {FRONTIER_VAL.relative_to(ROOT)}
  - Frontier snapshot:                {FRONTIER_WORKFLOW.relative_to(ROOT)}
  - Prior iteration summary:          {EVOLUTION_SUMMARY.relative_to(ROOT)}
  - Output pending eval to:           {PENDING_EVAL.relative_to(ROOT)}
  - SopBenchAgent (READ-ONLY):        agent/sopbench_agent.py
  - Component runtime (READ-ONLY):    agent/component_runtime_sopbench/
  - THIS DOMAIN's components dir:     {comp_dir}/  (READ-ONLY; modify by
                                       reusing COMPONENT.name = replace_node)
  - Per-task baselines:               meta_harness/logs_components_sopbench_{DOMAIN}/v0__train__results.json
  - Component fire trace:             .component-state-sopbench/{DOMAIN}_iter<N-1>/fired.jsonl (absent on iter 1)

IMPORTANT: other domains' components live in their own dirs
(`components_sopbench_<other_domain>/`). Do NOT read them; isolation is
no longer enforced because we now use per-domain dirs. You only write to
this domain's dir: `{comp_dir}/`.

Steps:
  1. Read workflows/sopbench_{DOMAIN}.yaml, frontier_workflow.json,
     frontier_val.json, and the last few entries of evolution_summary.jsonl.
  2. Pick 4-6 train task_ids where the current frontier scores 0 (look at
     `per_task` in frontier_val.json). Read their stored predicted_output
     and compare to expected_output to identify the failure mode.
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
       from agent.component_runtime_sopbench import load_components_from_dir
       comps = load_components_from_dir('{comp_dir}', only=['<COMPONENT.name>'])
       assert any(c.name == '<COMPONENT.name>' for cs in comps.values() for c in cs)
       print('ok')
       "
  6. Write {PENDING_EVAL.relative_to(ROOT)} with this JSON schema:
       {{
         "candidate": {{
           "name": "candidate_iter{iteration}_<slug>",
           "hypothesis": "...",
           "changes": "...",
           "component": {{
             "name": "...", "cls": "...", "mount": "...",
             "file": "agent/components_sopbench/...py",
             "trust": {{...}}
           }},
           "workflow_patch": {{
             "op": "add_node|replace_node|disable_node",
             "name": "<COMPONENT.name>",
             "file": "agent/components_sopbench/...py",
             "edges_in": [], "edges_out": []
           }}
         }}
       }}
  7. Print one final line: CANDIDATE: <candidate_name>

CRITICAL:
  - Do NOT run benchmarks. Do NOT modify amazon_sop_bench/, third_party/,
    bench/, run_benchmark.py, meta_harness/scripts/run_sopbench_baseline.py,
    agent/sopbench_agent.py, agent/component_runtime_sopbench/, agent/llm.py.
  - **The target inference model is LOCKED.** Do NOT pass a model= kwarg
    to chat() and do NOT call any other model API. chat() enforces this
    at call time (RuntimeError on any model override). No "fallback to a
    cheaper variant", no second-opinion calls to a different model.
  - **Per-domain dirs (no cross-domain reads)**: other domains have their
    own component dirs and logs. Do NOT read files under
    components_sopbench_<other_domain>/, logs_components_sopbench_<other_domain>/,
    or workflows/sopbench_<other_domain>.yaml. Stick to YOUR domain only.
  - No task-specific hardcoding. No entity names or test-set IDs in code.

NEW (Phase D event-runtime additions; see SKILL.md "Event runtime additions"):
  - All 15 Tier-1 events fire in sopbench. You MAY use
    `listens="on_tool_error"` / `"on_length_truncation"` /
    `"pre_tool_arg_validation"` / `"post_tool_result_raw"` /
    `"on_explicit_terminate"` etc. on your Component to subscribe to
    runtime-synthesised events directly, instead of writing matchers on
    `ctx.current_tool_success` / `ctx.finish_reason`. The `mount=` field
    is still required for policy validation (set it to the closest Mount).
  - `ctx.chat(messages, max_tokens=..., temperature=...)` is available for
    sub-LLM verifier / re-format / critic patterns — locked SUT model, no
    `model=` kwarg. Declare `capabilities=(Capability.LLM_CALL,)` if used.
  - `ctx.emit("iter<N>_<slug>_<event>")` / `ctx.emit_upstream(...)` let two
    components exchange data within one task. Declare `emits=(...)` on the
    publisher for audit / discovery.
"""


def run_proposer(iteration: int, log_dir: Path, skill_name: str) -> dict:
    print(f"\n=== sopbench/{DOMAIN} iter {iteration}: proposer starting "
          f"(skill={skill_name}) ===", flush=True)
    max_attempts = 6
    for attempt in range(1, max_attempts + 1):
        result = claude_wrapper.run(
            prompt=_proposer_prompt(iteration, skill_name),
            model=PROPOSER_MODEL,
            allowed_tools=PROPOSER_TOOLS,
            cwd=str(ROOT),
            log_dir=str(log_dir),
            name=f"sopbench_{DOMAIN}_iter{iteration}_proposer",
            skills=[skill_name],
            skill_dir=str(SKILLS_PARENT),
            disable_skills=True,
            disable_mcp=True,
            progress=True,
            docker_skill=skill_name,
            docker_domain=DOMAIN,
            docker_container_name=f"robagent-proposer-sopbench-{DOMAIN}-"
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


def run_eval(workflow: Workflow, iteration: int, agent_name: str,
             hypothesis: str, changes: str, plugin_payload: dict) -> dict:
    active = list(workflow.active_nodes())
    print(f"\n=== sopbench eval iter {iteration} active={active} on "
          f"{DOMAIN} train-{TRAIN_SIZE} ===", flush=True)
    env = os.environ.copy()
    env["SOPBENCH_COMPONENT_WORKFLOW"] = str(WORKFLOW_YAML)
    env["SOPBENCH_COMPONENT_NAMES"] = ",".join(active)
    env["SOPBENCH_COMPONENT_RUN_TAG"] = f"{DOMAIN}_iter{iteration}"
    env["SOPBENCH_COMPONENT_DIR"] = str(_components_dir_for(DOMAIN))
    env["SOPBENCH_BENCHMARK_NAME"] = DOMAIN
    cmd = [
        sys.executable, "-u",
        str(THIS_DIR / "scripts" / "run_sopbench_baseline.py"),
        "--domain", DOMAIN,
        "--agent-name", agent_name,
        "--iteration", str(iteration),
        "--train-size", str(TRAIN_SIZE),
        "--max-workers", str(TRAIN_PARALLEL),
        "--subsets", "train",
        "--hypothesis", hypothesis,
        "--changes", changes,
        "--plugin-json", json.dumps(plugin_payload),
    ]
    started = time.time()
    ret = subprocess.run(cmd, cwd=ROOT, env=env)
    print(f"  exit={ret.returncode} elapsed={time.time()-started:.0f}s",
          flush=True)
    if ret.returncode != 0:
        raise SystemExit(f"run_sopbench_baseline.py exited {ret.returncode}")

    last_line = ""
    for line in reversed(EVOLUTION_SUMMARY.read_text().splitlines()):
        if line.strip():
            last_line = line
            break
    if not last_line:
        raise SystemExit(f"evolution_summary.jsonl missing/empty after eval")
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
    env = os.environ.copy()
    env["SOPBENCH_COMPONENT_WORKFLOW"] = str(WORKFLOW_YAML)
    env["SOPBENCH_COMPONENT_NAMES"] = ",".join(wf.active_nodes())
    env["SOPBENCH_COMPONENT_RUN_TAG"] = f"{DOMAIN}_final_test"
    env["SOPBENCH_COMPONENT_DIR"] = str(_components_dir_for(DOMAIN))
    env["SOPBENCH_BENCHMARK_NAME"] = DOMAIN
    cmd = [
        sys.executable, "-u",
        str(THIS_DIR / "scripts" / "run_sopbench_baseline.py"),
        "--domain", DOMAIN,
        "--agent-name", f"frontier_iter{_current_iteration() - 1}",
        "--iteration", str(_current_iteration() - 1),
        "--train-size", str(TRAIN_SIZE),
        "--max-workers", str(TRAIN_PARALLEL),
        "--subsets", "test",
        "--hypothesis", "final test eval of accepted frontier",
    ]
    started = time.time()
    ret = subprocess.run(cmd, cwd=ROOT, env=env)
    print(f"  exit={ret.returncode} elapsed={time.time()-started:.0f}s",
          flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True,
                   choices=["dangerous_goods", "warehouse_package_inspection",
                            "traffic_spoofing_detection"])
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--train-size", type=int, default=30)
    p.add_argument("--train-parallel", type=int, default=4)
    p.add_argument("--final-test", action="store_true")
    p.add_argument("--proposer-only", action="store_true")
    p.add_argument("--skill", default=DEFAULT_SKILL)
    args = p.parse_args()

    global LOGS, WORKFLOW_YAML, PENDING_EVAL, FRONTIER_VAL, FRONTIER_WORKFLOW
    global EVOLUTION_SUMMARY, DOMAIN, TRAIN_SIZE, TRAIN_PARALLEL
    DOMAIN = args.domain
    TRAIN_SIZE = args.train_size
    TRAIN_PARALLEL = args.train_parallel
    LOGS = _logs_for(DOMAIN).resolve()
    WORKFLOW_YAML = _workflow_yaml_for(DOMAIN).resolve()
    PENDING_EVAL = LOGS / "pending_eval.json"
    FRONTIER_VAL = LOGS / "frontier_val.json"
    FRONTIER_WORKFLOW = LOGS / "frontier_workflow.json"
    EVOLUTION_SUMMARY = LOGS / "evolution_summary.jsonl"

    if not (SKILLS_PARENT / args.skill / "SKILL.md").exists():
        raise SystemExit(f"skill not found: {SKILLS_PARENT / args.skill}/SKILL.md")

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
