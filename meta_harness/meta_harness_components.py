"""Outer evolution loop for component-harness-tau2.

Successor to meta_harness_hooks.py. Differences:

  * The proposer runs the `component-harness-tau2` skill, whose artefact
    is a single component file under `agent_tau2/components/` plus a
    `workflow_patch` block (one of add_node / replace_node / disable_node)
    in pending_eval.json.
  * The "candidate" evaluated by tau2_runner is always the fixed
    `component_runtime` module; the active component set is determined by
    the workflow YAML at `meta_harness/workflows/tau2_main.yaml`, pinned
    by env var `COMPONENT_NAMES` for a defensive cross-check.
  * The frontier is the workflow GRAPH itself. Each iteration applies the
    candidate patch, scores the resulting graph, and either keeps the
    patch (workflow YAML and any new file persist) or rolls back (YAML
    reverts to prev; new add_node file is deleted; replace_node restores
    from .bak_iter<N>; disable_node YAML reverts).

State files live under `meta_harness/logs_tau2_components/` so this loop
never collides with the legacy hook / robust / baseline frontiers.

Usage:
  python meta_harness/meta_harness_components.py --iterations 1 --proposer-only
  python meta_harness/meta_harness_components.py --iterations 5
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

from agent_tau2.component_runtime.workflow import (  # noqa: E402
    apply_patch,
    FrontierSnapshot,
    Patch,
    PatchOp,
    Workflow,
)

DEFAULT_LOGS = THIS_DIR / "logs_tau2_components"
WORKFLOW_YAML_DEFAULT = THIS_DIR / "workflows" / "tau2_main.yaml"
TRAIN_IDS = THIS_DIR / "tau2_train_task_ids.txt"
TEST_IDS = THIS_DIR / "tau2_test_task_ids.txt"
SKILLS_PARENT = THIS_DIR / ".claude" / "skills"
DEFAULT_SKILL = "component-harness-tau2"
DOMAIN = "banking_knowledge"
COMPONENT_RUNTIME_CANDIDATE = "component_runtime"

LOGS: Path
WORKFLOW_YAML: Path
PENDING_EVAL: Path
FRONTIER_VAL: Path
FRONTIER_WORKFLOW: Path        # JSON snapshot of accepted workflow
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
# Isolation: physically hide prior-iter artefacts during the proposer run.
# ----------------------------------------------------------------------------

_ISOLATE_GLOBS = [
    ("agent_tau2", "mh_tau2_iter*"),       # legacy LLMAgent-subclass candidates
    ("meta_harness", "logs_tau2*"),         # legacy frontiers; logs_tau2_components kept
]
_ISOLATE_KEEP_LOGS_DIR = "logs_tau2_components"

_ISOLATE_MEMORY_FILES = [
    "tau2-frontier-fail-decomposition.md",  # 100% iter / frontier viewpoint
    "hook-harness-tau2-design.md",          # legacy hook design notes
    "component-harness-tau2-design.md",     # this skill's own design notes (if any)
]
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
    if not memory_md.exists():
        return None
    original = memory_md.read_text()
    kept_lines = []
    for line in original.splitlines():
        if any(f"({name})" in line for name in hidden_basenames):
            continue
        kept_lines.append(line)
    memory_md.write_text("\n".join(kept_lines) + "\n")
    return original


@contextlib.contextmanager
def _isolate_proposer(root: Path):
    """Hide prior-iter artefacts so the proposer cannot read them.

    Each target is renamed under a timestamped stash directory; MEMORY.md
    is rewritten to drop links to hidden memory files. On exit (success or
    exception) every move is reversed and the stash is removed.
    """
    targets = _isolation_targets(root)
    ts = time.strftime("%Y%m%d_%H%M%S")
    stash = root / f".component_proposer_hidden_{ts}"
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
# Proposer prompt + invocation.
# ----------------------------------------------------------------------------


def _proposer_prompt(iteration: int, skill_name: str, slug_prefix: str) -> str:
    name_pattern = (
        f"component_iter{iteration}_{slug_prefix}_<slug>" if slug_prefix
        else f"component_iter{iteration}_<slug>"
    )
    return f"""You are a tau2-bench component proposer. Follow the SKILL.md ({skill_name}).

Iteration: {iteration}
Working directory: {ROOT}

State files (relative to working directory):
  - Workflow graph (frontier):  {WORKFLOW_YAML.relative_to(ROOT)}
  - Frontier per-task best:     {FRONTIER_VAL.relative_to(ROOT)}
  - Frontier snapshot:          {FRONTIER_WORKFLOW.relative_to(ROOT)}
  - Prior iteration summary:    {EVOLUTION_SUMMARY.relative_to(ROOT)}
  - Output pending eval to:     {PENDING_EVAL.relative_to(ROOT)}
  - Train task ids:             meta_harness/tau2_train_task_ids.txt (30 banking_knowledge tasks)
  - Base agent (read-only):     agent_tau2/v0/agent.py  (stock tau2 LLMAgent)
  - Component runtime (READ-ONLY): agent_tau2/component_runtime/
  - Existing components:        agent_tau2/components/  (READ-ONLY for everyone else;
                                  modify only by reusing COMPONENT.name = replace_node)
  - Reference examples:         agent_tau2/components/close_account_strip_optional_reason.py
                                agent_tau2/components/discoverable_audit_channel.py
  - tau2 sim records dir:       tau2-bench-src/data/simulations/tau2-runs/meta/
      * component-loop sims:    <dir>/{COMPONENT_RUNTIME_CANDIDATE}__{DOMAIN}.json/results.json
      * BOOTSTRAP (iter 1, no component sims yet): read v0's sim at
        <dir>/v0__{DOMAIN}.json/results.json — that is the stock base
        agent on the same 30 tasks and tells you which tasks v0 fails.
  - Component fire trace:       .component-state/iter<N-1>/fired.jsonl (absent on iter 1)

Steps:
  1. Read frontier_val.json, workflows/tau2_main.yaml, and the last few
     entries of evolution_summary.jsonl. If this is iter 1 (frontier just
     bootstrapped with 2 migrated components), the relevant base sim is
     v0's at tau2-bench-src/data/simulations/tau2-runs/meta/v0__{DOMAIN}.json/results.json.
  2. Pick 4-6 train task_ids the current frontier fails. Read their tau2
     simulation (messages + reward_info) under
     tau2-bench-src/data/simulations/tau2-runs/meta/. Cross-reference
     .component-state/iter<N-1>/fired.jsonl (when present) to see which
     frontier components fired on them.
  3. Form ONE hypothesis tied to a mechanism present in >=3 of those sims.
  4. Choose ONE patch op: add_node | replace_node | disable_node.
       * add_node: create exactly one file at
         agent_tau2/components/{name_pattern}.py exporting `COMPONENT: Component`.
       * replace_node: BEFORE editing, run
           cp agent_tau2/components/<existing>.py agent_tau2/components/<existing>.py.bak_iter{iteration}
         then overwrite the file (keep COMPONENT.name unchanged).
       * disable_node: write no file; just reference the existing component
         name in the workflow_patch block.
  5. Validate registration + trust:
       python -c "
       from agent_tau2.component_runtime.registry import load_components_from_dir
       comps = load_components_from_dir(only=['<COMPONENT.name>'])
       assert any(c.name == '<COMPONENT.name>' for cs in comps.values() for c in cs)
       print('ok')
       "
  6. Write {PENDING_EVAL.relative_to(ROOT)} with the JSON schema from
     SKILL.md (component block + workflow_patch block).
  7. Print one final line: CANDIDATE: <candidate_name>

CRITICAL:
  - Do NOT run benchmarks. Do NOT modify tau2-bench-src/, tau2_runner.py,
    meta_harness/*, agent_tau2/v0/, agent_tau2/component_runtime/, or any
    earlier component file (replace_node by reusing COMPONENT.name is
    allowed; in-place edit of someone else's component file is not).
  - Do NOT build a new agent directory under agent_tau2/. This skill
    only writes component files (under agent_tau2/components/) and
    workflow_patch metadata in pending_eval.json.
  - **Isolation invariant**: legacy paths (agent_tau2/mh_tau2_iter*,
    meta_harness/logs_tau2_*, except logs_tau2_components) are physically
    moved out of the tree before you start. If you discover a path that
    names "mh_tau2_iter" or a robust/baseline frontier log, the isolation
    has a bug — do not read it.
  - No task-specific hardcoding. No customer names / order ids in code.
"""


def run_proposer(iteration: int, log_dir: Path, skill_name: str,
                 slug_prefix: str) -> dict:
    print(f"\n=== component iter {iteration}: proposer starting "
          f"(skill={skill_name}) ===", flush=True)
    max_attempts = 6
    for attempt in range(1, max_attempts + 1):
        result = claude_wrapper.run(
            prompt=_proposer_prompt(iteration, skill_name, slug_prefix),
            model=PROPOSER_MODEL,
            allowed_tools=PROPOSER_TOOLS,
            cwd=str(ROOT),
            log_dir=str(log_dir),
            name=f"component_iter{iteration}_proposer_{skill_name}",
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
    raise SystemExit(f"proposer failed in iter {iteration} "
                     f"after {max_attempts} attempts")


# ----------------------------------------------------------------------------
# Eval + scoring.
# ----------------------------------------------------------------------------


def run_eval(workflow: Workflow, iteration: int, ids_file: Path,
             parallel: int, label: str) -> Path:
    active = list(workflow.active_nodes())
    print(f"\n=== component eval iter {iteration} active={active} on {label} ===",
          flush=True)
    env = os.environ.copy()
    env.setdefault("TAU2_DATA_ROOT",
                   str(ROOT / "tau2-bench-src" / "data" / "tau2"))
    env["COMPONENT_NAMES"] = ",".join(active)
    env["COMPONENT_WORKFLOW"] = str(WORKFLOW_YAML)
    env["COMPONENT_RUN_TAG"] = f"iter{iteration}"
    cmd = [
        sys.executable, str(ROOT / "tau2_runner.py"),
        "--candidate", COMPONENT_RUNTIME_CANDIDATE,
        "--domain", DOMAIN,
        "--task-ids-file", str(ids_file),
        "--max-concurrency", str(parallel),
    ]
    started = time.time()
    ret = subprocess.run(cmd, cwd=ROOT, env=env)
    print(f"  exit={ret.returncode} elapsed={time.time()-started:.0f}s",
          flush=True)
    if ret.returncode != 0:
        raise SystemExit(f"tau2_runner exited {ret.returncode}")
    summary = ROOT / "traces" / f"tau2_{COMPONENT_RUNTIME_CANDIDATE}__summary.jsonl"
    if not summary.exists():
        raise SystemExit(f"expected summary missing: {summary}")
    return summary


def _previous_frontier_correct() -> int:
    if not FRONTIER_VAL.exists():
        return 0
    fv = json.loads(FRONTIER_VAL.read_text())
    return int(fv.get("frontier_score", {}).get("correct", 0))


def _backup_replace_file(patch: Patch, iteration: int) -> tuple[Path, Path] | None:
    """For replace_node: locate the .bak_iter<N> the builder was told to make.

    Returns (live_path, bak_path) if the bak exists, else None. The outer
    loop trusts the SKILL.md-mandated `cp <file> <file>.bak_iter<N>` step.
    """
    if patch.op is not PatchOp.REPLACE_NODE or not patch.file:
        return None
    live = ROOT / patch.file
    bak = live.with_suffix(live.suffix + f".bak_iter{iteration}")
    if bak.exists():
        return live, bak
    return None


def _rollback_patch(prev_workflow: Workflow, patch: Patch,
                    backup: tuple[Path, Path] | None) -> None:
    """Reverse the patch on disk."""
    # Workflow YAML always reverts to prev on reject.
    prev_workflow.to_yaml(WORKFLOW_YAML)

    if patch.op is PatchOp.ADD_NODE and patch.file:
        f = ROOT / patch.file
        if f.exists():
            f.unlink()
            print(f"  cleaned up rejected component file: {patch.file}",
                  flush=True)
    elif patch.op is PatchOp.REPLACE_NODE:
        if backup is not None:
            live, bak = backup
            live.write_text(bak.read_text())
            bak.unlink()
            print(f"  restored {live.relative_to(ROOT)} from .bak", flush=True)
        else:
            print(f"  WARN: replace_node rejected but no .bak found at "
                  f"{patch.file}.bak_iter<N>; the new file remains in place",
                  flush=True)
    # disable_node: nothing on disk beyond the YAML revert.


def _commit_patch(patch: Patch, backup: tuple[Path, Path] | None) -> None:
    """Finalise an accepted patch (e.g. delete the .bak)."""
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
        "--train-file", str(TRAIN_IDS),
    ]
    # Pack the component + workflow_patch blocks together; score_candidate.py
    # stores --plugin-json verbatim under the `plugin` key.
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--train-parallel", type=int, default=8)
    p.add_argument("--proposer-only", action="store_true")
    p.add_argument("--skill", default=DEFAULT_SKILL,
                   help=f"skill under {SKILLS_PARENT.relative_to(ROOT)}/ "
                        f"(default: {DEFAULT_SKILL})")
    p.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS)
    p.add_argument("--workflow-yaml", type=Path, default=WORKFLOW_YAML_DEFAULT)
    p.add_argument("--candidate-slug-prefix", default="")
    args = p.parse_args()

    if not (SKILLS_PARENT / args.skill / "SKILL.md").exists():
        raise SystemExit(f"skill not found: "
                         f"{SKILLS_PARENT / args.skill}/SKILL.md")

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


if __name__ == "__main__":
    main()
