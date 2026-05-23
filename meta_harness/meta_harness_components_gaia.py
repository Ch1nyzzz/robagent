"""Outer evolution loop for component-harness-gaia.

Sibling of meta_harness_components.py. GAIA differences:

  * agent_version_arg is always "component_runtime"; the runtime resolves
    the active set from meta_harness/workflows/gaia_main.yaml at task time.
  * Evaluation goes through run_benchmark.py (not tau2_runner.py).
  * Score gate: per-task best update (matches the original GAIA flow);
    the workflow YAML is committed iff overall train accuracy does not
    regress vs prior frontier.
  * Final-test eval mode: re-runs the accepted workflow on test-135.

State lives under `meta_harness/logs_components_gaia/`.
"""
from __future__ import annotations

import argparse
import contextlib
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

from agent.component_runtime.workflow import (  # noqa: E402
    apply_patch,
    FrontierSnapshot,
    Patch,
    PatchOp,
    Workflow,
)

DEFAULT_LOGS = THIS_DIR / "logs_components_gaia"
WORKFLOW_YAML_DEFAULT = THIS_DIR / "workflows" / "gaia_main.yaml"
TRAIN_IDS = THIS_DIR / "train_task_ids.txt"
TEST_IDS = THIS_DIR / "test_task_ids.txt"
SKILLS_PARENT = THIS_DIR / ".claude" / "skills"
DEFAULT_SKILL = "component-harness-gaia"
COMPONENT_RUNTIME_AGENT_VERSION = "component_runtime"

LOGS: Path
WORKFLOW_YAML: Path
PENDING_EVAL: Path
FRONTIER_VAL: Path
FRONTIER_WORKFLOW: Path
EVOLUTION_SUMMARY: Path

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


# ----------------------------------------------------------------------------
# Isolation: hide prior-iter artefacts during the proposer run.
# ----------------------------------------------------------------------------

_ISOLATE_GLOBS = [
    ("agent", "mh_iter*"),                 # prior robust/baseline candidates
    ("meta_harness", "logs_*"),             # prior frontiers; keep-list below
]
_ISOLATE_KEEP_LOGS_DIR = "logs_components_gaia"

_ISOLATE_MEMORY_FILES: list[str] = []      # GAIA has no iter-viewpoint memories yet
_MEMORY_DIR = Path.home() / ".claude" / "projects" / "-Users-erv1n-robagent" / "memory"


def _isolation_targets(root: Path) -> list[Path]:
    targets: list[Path] = []
    for top, pat in _ISOLATE_GLOBS:
        for p in sorted((root / top).glob(pat)):
            if p.name == _ISOLATE_KEEP_LOGS_DIR:
                continue
            targets.append(p)
    for name in _ISOLATE_MEMORY_FILES:
        p = _MEMORY_DIR / name
        if p.exists():
            targets.append(p)
    return targets


def _hide_memory_index_lines(memory_md: Path, hidden_basenames: set[str]) -> str | None:
    if not memory_md.exists() or not hidden_basenames:
        return None
    original = memory_md.read_text()
    kept_lines = [
        line for line in original.splitlines()
        if not any(f"({name})" in line for name in hidden_basenames)
    ]
    memory_md.write_text("\n".join(kept_lines) + "\n")
    return original


@contextlib.contextmanager
def _isolate_proposer(root: Path):
    targets = _isolation_targets(root)
    ts = time.strftime("%Y%m%d_%H%M%S")
    stash = root / f".component_gaia_proposer_hidden_{ts}"
    stash.mkdir(parents=True, exist_ok=False)
    manifest_path = stash / "moved.json"

    moved: list[tuple[Path, Path]] = []
    memory_md = _MEMORY_DIR / "MEMORY.md"
    memory_original: str | None = None

    try:
        manifest: list[dict] = []
        for src in targets:
            stem = f"{abs(hash(str(src))) & 0xffff_ffff:08x}__{src.name}"
            dst = stash / stem
            src.rename(dst)
            moved.append((src, dst))
            manifest.append({"src": str(src), "dst": str(dst)})
        manifest_path.write_text(json.dumps(manifest, indent=2))

        memory_original = _hide_memory_index_lines(
            memory_md, hidden_basenames=set(_ISOLATE_MEMORY_FILES)
        )

        print(f"  isolation: hid {len(moved)} paths into {stash.name}/", flush=True)
        yield
    finally:
        for src, dst in reversed(moved):
            try:
                if dst.exists():
                    dst.rename(src)
            except Exception as e:
                print(f"  WARN: failed to restore {src}: {e}", flush=True)
        if memory_original is not None:
            try:
                memory_md.write_text(memory_original)
            except Exception as e:
                print(f"  WARN: failed to restore MEMORY.md: {e}", flush=True)
        try:
            shutil.rmtree(stash)
        except Exception as e:
            print(f"  WARN: stash {stash} not removed: {e}", flush=True)


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


def _proposer_prompt(iteration: int, skill_name: str, slug_prefix: str) -> str:
    name_pattern = (
        f"component_iter{iteration}_{slug_prefix}_<slug>" if slug_prefix
        else f"component_iter{iteration}_<slug>"
    )
    return f"""You are a GAIA component proposer. Follow the SKILL.md ({skill_name}).

Iteration: {iteration}
Working directory: {ROOT}

State files (relative to working directory):
  - Workflow graph (frontier):  {WORKFLOW_YAML.relative_to(ROOT)}
  - Frontier per-task best:     {FRONTIER_VAL.relative_to(ROOT)}
  - Frontier snapshot:          {FRONTIER_WORKFLOW.relative_to(ROOT)}
  - Prior iteration summary:    {EVOLUTION_SUMMARY.relative_to(ROOT)}
  - Output pending eval to:     {PENDING_EVAL.relative_to(ROOT)}
  - Train task ids:             meta_harness/train_task_ids.txt (30 tasks)
  - Base agent (read-only):     agent/base.py
  - Component runtime (READ-ONLY): agent/component_runtime/
  - Existing components:        agent/components/  (READ-ONLY; modify by reusing
                                                    COMPONENT.name = replace_node)
  - Train traces:               traces/runs/gaia__<task_id>__*.jsonl
  - Component fire trace:       .component-state/iter<N-1>/fired.jsonl (absent on iter 1)

Steps:
  1. Read workflows/gaia_main.yaml, frontier_workflow.json, frontier_val.json,
     and the last few entries of evolution_summary.jsonl.
  2. Pick 4-6 train task_ids where the current frontier scores 0. Read their
     newest trace under traces/runs/. Cross-reference fired.jsonl (when
     present) for which existing components fired.
  3. Form ONE hypothesis tied to a mechanism present in >=3 of those traces.
  4. Choose ONE patch op: add_node | replace_node | disable_node.
       * add_node: create exactly one file at
         agent/components/{name_pattern}.py exporting `COMPONENT: Component`.
       * replace_node: BEFORE editing, run
           cp agent/components/<existing>.py agent/components/<existing>.py.bak_iter{iteration}
         then overwrite (keep COMPONENT.name unchanged).
       * disable_node: write no file; reference the existing component
         name in the workflow_patch block.
  5. Validate registration + trust:
       python -c "
       from agent.component_runtime.registry import load_components_from_dir
       comps = load_components_from_dir(only=['<COMPONENT.name>'])
       assert any(c.name == '<COMPONENT.name>' for cs in comps.values() for c in cs)
       print('ok')
       "
  6. Write {PENDING_EVAL.relative_to(ROOT)} with the JSON schema from
     SKILL.md (component block + workflow_patch block). Note
     candidate.agent_version_arg = "component_runtime".
  7. Print one final line: CANDIDATE: <candidate_name>

CRITICAL:
  - Do NOT run benchmarks. Do NOT modify bench/, run_benchmark.py,
    meta_harness/*, agent/base.py, agent/component_runtime/, agent/llm.py,
    agent/events.py, or any earlier agent/v*/ / agent/mh_iter*/ / agent/components/
    (except via replace_node with the .bak protocol).
  - Do NOT build a new agent directory under agent/. That's robust-harness-gaia;
    this skill only writes component files + workflow_patch metadata.
  - **Isolation invariant**: legacy paths (agent/mh_iter*, meta_harness/logs_*
    except logs_components_gaia) are physically moved out of the tree before
    you start. If you discover a path that names "mh_iter" or a robust/baseline
    frontier log, the isolation has a bug — do not read it.
  - No task-specific hardcoding. No entity names or task ids in code.
"""


def run_proposer(iteration: int, log_dir: Path, skill_name: str,
                 slug_prefix: str) -> dict:
    print(f"\n=== gaia iter {iteration}: proposer starting "
          f"(skill={skill_name}) ===", flush=True)
    max_attempts = 6
    for attempt in range(1, max_attempts + 1):
        result = claude_wrapper.run(
            prompt=_proposer_prompt(iteration, skill_name, slug_prefix),
            model=PROPOSER_MODEL,
            allowed_tools=PROPOSER_TOOLS,
            cwd=str(ROOT),
            log_dir=str(log_dir),
            name=f"gaia_iter{iteration}_proposer_{skill_name}",
            skills=[skill_name],
            skill_dir=str(SKILLS_PARENT),
            disable_skills=True,
            disable_mcp=True,
            progress=True,
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


def run_eval(workflow: Workflow, iteration: int, ids_file: Path,
             parallel: int, label: str) -> Path:
    active = list(workflow.active_nodes())
    print(f"\n=== gaia eval iter {iteration} active={active} on {label} ===",
          flush=True)
    env = os.environ.copy()
    env["COMPONENT_NAMES"] = ",".join(active)
    env["COMPONENT_WORKFLOW"] = str(WORKFLOW_YAML)
    env["COMPONENT_RUN_TAG"] = f"iter{iteration}"
    cmd = [
        sys.executable, str(ROOT / "run_benchmark.py"),
        "gaia",
        "--agent-version", COMPONENT_RUNTIME_AGENT_VERSION,
        "--task-ids-file", str(ids_file),
        "--parallel", str(parallel),
    ]
    started = time.time()
    ret = subprocess.run(cmd, cwd=ROOT, env=env)
    print(f"  exit={ret.returncode} elapsed={time.time()-started:.0f}s",
          flush=True)
    if ret.returncode != 0:
        raise SystemExit(f"run_benchmark.py exited {ret.returncode}")
    summary = ROOT / "traces" / f"gaia_{COMPONENT_RUNTIME_AGENT_VERSION}__summary.jsonl"
    if not summary.exists():
        raise SystemExit(f"expected summary missing: {summary}")
    return summary


def _previous_frontier_correct() -> int:
    if not FRONTIER_VAL.exists():
        return 0
    fv = json.loads(FRONTIER_VAL.read_text())
    return int(fv.get("frontier_score", {}).get("correct", 0))


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


def score_and_update(iteration: int, pending: dict, summary_path: Path,
                     prev_workflow: Workflow, next_workflow: Workflow,
                     patch: Patch, backup: tuple[Path, Path] | None) -> None:
    cand = pending["candidate"]
    prev_correct = _previous_frontier_correct()
    cmd = [
        sys.executable, str(THIS_DIR / "scripts" / "score_candidate.py"),
        "--agent-name", cand["name"],
        "--summary-path", str(summary_path),
        "--iteration", str(iteration),
        "--hypothesis", cand.get("hypothesis", ""),
        "--changes", cand.get("changes", ""),
        "--logs-dir", str(LOGS),
    ]
    payload = {
        "component": cand.get("component"),
        "workflow_patch": cand.get("workflow_patch"),
    }
    cmd += ["--plugin-json", json.dumps(payload)]
    if subprocess.run(cmd, cwd=ROOT).returncode != 0:
        raise SystemExit("score_candidate.py failed")

    last_line = EVOLUTION_SUMMARY.read_text().splitlines()[-1]
    last = json.loads(last_line)
    cand_correct = int(last.get("train_score", {}).get("correct", 0))
    accepted = cand_correct >= prev_correct

    last["accepted"] = accepted
    last["workflow_after"] = list(
        (next_workflow if accepted else prev_workflow).active_nodes()
    )
    lines = EVOLUTION_SUMMARY.read_text().splitlines()
    lines[-1] = json.dumps(last, default=str)
    EVOLUTION_SUMMARY.write_text("\n".join(lines) + "\n")

    if accepted:
        _commit_patch(patch, backup)
        _save_frontier_snapshot(next_workflow, iteration)
        print(f"  accepted (train {cand_correct} ≥ {prev_correct}); "
              f"frontier workflow → {list(next_workflow.active_nodes())}",
              flush=True)
    else:
        print(f"  REJECTED (train {cand_correct} < {prev_correct}); "
              f"rolling back patch", flush=True)
        _rollback_patch(prev_workflow, patch, backup)
    PENDING_EVAL.unlink()


def final_test_eval(parallel: int) -> None:
    """Run the current accepted frontier workflow on test-135 once."""
    wf = _load_frontier_workflow()
    if not wf.active_nodes():
        print("frontier workflow is empty; skipping final test eval"); return
    print(f"\n=== final test eval: workflow active={list(wf.active_nodes())} "
          f"on test-135 ===", flush=True)
    summary = run_eval(wf, _current_iteration() - 1, TEST_IDS, parallel, "test-135")
    rows = [json.loads(l) for l in summary.read_text().splitlines() if l.strip()]
    n = len(rows)
    correct = sum(1 for r in rows if (r.get("score") or 0) > 0)
    print(f"\nTEST RESULT: workflow → {correct}/{n} = {correct / max(n, 1):.4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--train-parallel", type=int, default=8)
    p.add_argument("--test-parallel", type=int, default=8)
    p.add_argument("--final-test", action="store_true",
                   help="after iterations, run frontier workflow on test-135")
    p.add_argument("--proposer-only", action="store_true")
    p.add_argument("--skill", default=DEFAULT_SKILL)
    p.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS)
    p.add_argument("--workflow-yaml", type=Path, default=WORKFLOW_YAML_DEFAULT)
    p.add_argument("--candidate-slug-prefix", default="")
    args = p.parse_args()

    if not (SKILLS_PARENT / args.skill / "SKILL.md").exists():
        raise SystemExit(f"skill not found: {SKILLS_PARENT / args.skill}/SKILL.md")

    global LOGS, WORKFLOW_YAML, PENDING_EVAL, FRONTIER_VAL, FRONTIER_WORKFLOW
    global EVOLUTION_SUMMARY
    LOGS = args.logs_dir.resolve()
    LOGS.mkdir(parents=True, exist_ok=True)
    WORKFLOW_YAML = args.workflow_yaml.resolve()
    WORKFLOW_YAML.parent.mkdir(parents=True, exist_ok=True)
    PENDING_EVAL = LOGS / "pending_eval.json"
    FRONTIER_VAL = LOGS / "frontier_val.json"
    FRONTIER_WORKFLOW = LOGS / "frontier_workflow.json"
    EVOLUTION_SUMMARY = LOGS / "evolution_summary.jsonl"

    if not WORKFLOW_YAML.exists():
        Workflow().to_yaml(WORKFLOW_YAML)
    if not FRONTIER_WORKFLOW.exists():
        _save_frontier_snapshot(_load_frontier_workflow(), 0)

    proposer_log_dir = LOGS / "proposer_sessions"
    proposer_log_dir.mkdir(parents=True, exist_ok=True)

    for _ in range(args.iterations):
        iteration = _current_iteration()
        prev_workflow = _load_frontier_workflow()

        with _isolate_proposer(ROOT):
            pending = run_proposer(iteration, proposer_log_dir,
                                   args.skill, args.candidate_slug_prefix)
        if args.proposer_only:
            print("--proposer-only; stopping"); return

        patch = Patch.from_dict(pending["candidate"]["workflow_patch"])
        backup = _backup_replace_file(patch, iteration)
        next_workflow = apply_patch(prev_workflow, patch)
        next_workflow.to_yaml(WORKFLOW_YAML)

        summary = run_eval(next_workflow, iteration, TRAIN_IDS,
                           args.train_parallel, "train-30")
        score_and_update(iteration, pending, summary,
                         prev_workflow, next_workflow, patch, backup)

    if args.final_test:
        final_test_eval(args.test_parallel)


if __name__ == "__main__":
    main()
