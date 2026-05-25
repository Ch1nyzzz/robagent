"""Outer evolution loop for component-harness-toolathlon.

Per-iteration flow (toolathlon has a single corpus — no per-domain split
like sopbench; we evolve ONE workflow against the train task list):

  1. Load current frontier workflow (meta_harness/workflows/toolathlon_main.yaml).
  2. Run proposer (Claude Code subagent w/ component-harness-toolathlon
     skill). The proposer writes a single component file to
     agent_toolathlon/components/ and a `pending_eval.json` describing
     the workflow_patch + component metadata.
  3. Apply the patch → next_workflow; write the yaml.
  4. Run candidate `cr` on the train subset (28 task ids) via
     toolathlon_runner.py with COMPONENT_WORKFLOW / COMPONENT_NAMES /
     COMPONENT_DIR / COMPONENT_RUN_TAG env vars set.
  5. Champion-gate: candidate.n_correct ≥ last_accepted.n_correct?
       accept → snapshot frontier_workflow.json + commit
       reject → rollback workflow yaml; delete added file or restore .bak
  6. Update last evolution_summary.jsonl row with accepted bool +
     workflow_after.

iter 0 BOOTSTRAP is mandatory (no default prev_correct=0):
  python meta_harness_components_toolathlon.py \
      --bootstrap-from traces/toolathlon_v0_full_baseline.jsonl

The bootstrap extracts the train-28 subset score from the v0 99-task
baseline trace and writes the iter 0 row to evolution_summary.jsonl
plus frontier_val.json. This guarantees iter 1's champion-gate compares
against the REAL v0 baseline, not zero.

State lives under `meta_harness/logs_components_toolathlon/`.
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

from agent_toolathlon.component_runtime import (  # noqa: E402
    workflow as _wf_mod,
)
from agent_toolathlon.component_runtime.workflow import (  # noqa: E402
    FrontierSnapshot,
    Patch,
    PatchOp,
    Workflow,
    apply_patch,
)

WORKFLOWS_DIR     = THIS_DIR / "workflows"
WORKFLOW_YAML     = (WORKFLOWS_DIR / "toolathlon_main.yaml").resolve()
COMPONENTS_DIR    = (ROOT / "agent_toolathlon" / "components").resolve()
SKILLS_PARENT     = THIS_DIR / ".claude" / "skills"
DEFAULT_SKILL     = "component-harness-toolathlon"
TRAIN_IDS_FILE    = (THIS_DIR / "toolathlon_train_task_ids.txt").resolve()
TEST_IDS_FILE     = (THIS_DIR / "toolathlon_test_task_ids.txt").resolve()
LOGS              = (THIS_DIR / "logs_components_toolathlon").resolve()
PENDING_EVAL      = LOGS / "pending_eval.json"
FRONTIER_VAL      = LOGS / "frontier_val.json"
FRONTIER_WORKFLOW = LOGS / "frontier_workflow.json"
EVOLUTION_SUMMARY = LOGS / "evolution_summary.jsonl"

PROPOSER_MODEL = os.environ.get("MH_PROPOSER_MODEL", "opus")
PROPOSER_TOOLS = ["Read", "Glob", "Grep", "Edit", "Write", "Bash"]

# Defaults configurable per invocation.
TRAIN_PARALLEL = 8


# ---------------------------------------------------------------------------
# Bookkeeping helpers
# ---------------------------------------------------------------------------


def _current_iteration() -> int:
    """Next iter = 1 + max iteration seen so far. Iter 0 is the v0
    bootstrap row written by --bootstrap-from."""
    if not EVOLUTION_SUMMARY.exists():
        raise SystemExit(
            "evolution_summary.jsonl missing. Run "
            "`--bootstrap-from traces/toolathlon_v0_full_baseline.jsonl` first."
        )
    last = -1
    for line in EVOLUTION_SUMMARY.read_text().splitlines():
        try:
            last = max(last, int(json.loads(line).get("iteration", -1)))
        except json.JSONDecodeError:
            continue
    if last < 0:
        raise SystemExit(
            "evolution_summary.jsonl has no parseable rows; rerun bootstrap."
        )
    return last + 1


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


# ---------------------------------------------------------------------------
# iter 0 bootstrap from an existing baseline summary jsonl.
# ---------------------------------------------------------------------------


def _read_train_ids() -> set[str]:
    return {
        ln.strip() for ln in TRAIN_IDS_FILE.read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    }


def bootstrap_from_baseline(baseline_jsonl: Path) -> None:
    """Build iter 0 frontier_val.json + evolution_summary row from an
    already-completed baseline jsonl (e.g. toolathlon_v0_full_baseline.jsonl).

    The baseline must contain at least the 28 train task_ids; extras are
    ignored. Each record is expected to have fields `task_id`, `tier`,
    `passed`, and `dump_dir` (the v0 dump directory the proposer will
    later read for failure analysis).
    """
    train_ids = _read_train_ids()
    if not baseline_jsonl.exists():
        raise SystemExit(f"baseline jsonl not found: {baseline_jsonl}")

    recs = []
    for line in baseline_jsonl.read_text().splitlines():
        if line.strip():
            recs.append(json.loads(line))

    by_id = {r["task_id"]: r for r in recs}
    missing = sorted(train_ids - by_id.keys())
    if missing:
        print(f"  WARN: {len(missing)} train task_ids missing from baseline: "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}", flush=True)

    in_train = [by_id[t] for t in train_ids if t in by_id]
    n_correct = sum(1 for r in in_train if r.get("passed"))
    n_total = len(in_train)

    per_task = {
        r["task_id"]: {
            "passed": bool(r.get("passed")),
            "tier": r.get("tier"),
            "dump_dir": r.get("dump_dir"),
            "error": r.get("error"),
        }
        for r in in_train
    }

    LOGS.mkdir(parents=True, exist_ok=True)

    row = {
        "iteration": 0,
        "agent_name": "v0_baseline_bootstrap",
        "hypothesis": "empty workflow / v0 baseline",
        "changes": f"bootstrapped from {baseline_jsonl.relative_to(ROOT) if baseline_jsonl.is_relative_to(ROOT) else baseline_jsonl}",
        "train_score": {"n_correct": n_correct, "n_total": n_total},
        "per_task": per_task,
        "source_jsonl": str(baseline_jsonl),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "accepted": True,
        "workflow_after": [],
    }
    EVOLUTION_SUMMARY.write_text(json.dumps(row, default=str) + "\n")

    FRONTIER_VAL.write_text(json.dumps({
        "iteration": 0,
        "n_correct": n_correct,
        "n_total": n_total,
        "per_task": per_task,
        "source_jsonl": str(baseline_jsonl),
    }, indent=2))

    if not WORKFLOW_YAML.exists():
        Workflow().to_yaml(WORKFLOW_YAML)
    _save_frontier_snapshot(_load_frontier_workflow(), 0)

    print(f"iter 0 bootstrap from {baseline_jsonl}:", flush=True)
    print(f"  train_correct = {n_correct}/{n_total}", flush=True)
    print(f"  wrote: {EVOLUTION_SUMMARY.relative_to(ROOT)}", flush=True)
    print(f"         {FRONTIER_VAL.relative_to(ROOT)}", flush=True)
    print(f"         {FRONTIER_WORKFLOW.relative_to(ROOT)}", flush=True)


# ---------------------------------------------------------------------------
# Proposer.
# ---------------------------------------------------------------------------


def _proposer_prompt(iteration: int, skill_name: str,
                     candidate_slug_prefix: str) -> str:
    return f"""You are a Toolathlon component proposer. Follow the SKILL.md ({skill_name}).

Iteration: {iteration}
Working directory: {ROOT}

State files (relative to working directory):
  - Frontier workflow YAML:      {WORKFLOW_YAML.relative_to(ROOT)}
  - Frontier per-task best:      {FRONTIER_VAL.relative_to(ROOT)}
  - Frontier snapshot:           {FRONTIER_WORKFLOW.relative_to(ROOT)}
  - Prior iteration summary:     {EVOLUTION_SUMMARY.relative_to(ROOT)}
  - Output pending eval to:      {PENDING_EVAL.relative_to(ROOT)}
  - Toolathlon TaskAgent (READ-ONLY):  agent_toolathlon/component_runtime/task_agent.py
  - Component runtime (READ-ONLY):     agent_toolathlon/component_runtime/
  - Components dir:              {COMPONENTS_DIR.relative_to(ROOT)}/
                                 (you write here; modify by reusing
                                 COMPONENT.name = replace_node)
  - Component fire trace:        .component-state-toolathlon/toolathlon_iter<N-1>/fired.jsonl
                                 (absent on iter 1)

Per-task trace locations (all preserved across iterations; nothing is overwritten):
  CURRENT FRONTIER (= last accepted candidate; usually v0 until a candidate is admitted):
    frontier_val.json's per_task[<tid>].dump_dir
      = Toolathlon-runs/v0/finalpool/<tid>/                          (iter 0 / v0 baseline)
      OR Toolathlon-runs/cr/iter<K>/finalpool/<tid>/                 (iter K, if accepted)
  ANY PRIOR CANDIDATE (accepted or rejected; one row per iter ≥ 1):
    evolution_summary.jsonl[row_for_iter_K].per_task[<tid>].dump_dir
      = Toolathlon-runs/cr/iter<K>/finalpool/<tid>/
  Each <tid>/ directory holds:
    <dump_dir>/traj_log.json    (messages + tool calls + final status)
    <dump_dir>/eval_res.json    (pass/fail + verifier details)
    <dump_dir>/host_loop.log    (pretty-printed run trace)
  Component-fire records are also split per iter:
    .component-state-toolathlon/toolathlon_iter<K>/fired.jsonl

Steps:
  1. Read frontier_workflow.json, frontier_val.json, workflows/toolathlon_main.yaml,
     and the last few entries of evolution_summary.jsonl.
  2. Pick 4-6 train task_ids the current frontier FAILS (per_task[*].passed=false).
     Read their traj_log.json + eval_res.json + host_loop.log to identify
     the failure mode. Cluster failures by mechanism (tool name, missing
     argument, prompt-instruction omission, etc.).
  3. Form ONE hypothesis tied to a mechanism present in >=3 of those failures.
  4. Choose ONE patch op: add_node | replace_node | disable_node.
       * add_node: create exactly one file at
         {COMPONENTS_DIR.relative_to(ROOT)}/component_iter{iteration}_<slug>.py
         exporting `COMPONENT: Component`.
       * replace_node: BEFORE editing, run
           cp {COMPONENTS_DIR.relative_to(ROOT)}/<existing>.py \\
              {COMPONENTS_DIR.relative_to(ROOT)}/<existing>.py.bak_iter{iteration}
         then overwrite (keep COMPONENT.name unchanged).
       * disable_node: write no file; reference the existing component
         name in the workflow_patch block.
  5. Validate registration + trust:
       python -c "
       import sys; sys.path.insert(0, '.')
       from agent_toolathlon.component_runtime.registry import load_components_from_dir
       comps = load_components_from_dir('{COMPONENTS_DIR.relative_to(ROOT)}',
                                        only=['<COMPONENT.name>'])
       assert any(c.name == '<COMPONENT.name>' for cs in comps.values() for c in cs)
       print('ok')
       "
  6. Write {PENDING_EVAL.relative_to(ROOT)} with this JSON schema:
       {{
         "candidate": {{
           "name": "{candidate_slug_prefix}_iter{iteration}_<slug>",
           "hypothesis": "...",
           "changes": "...",
           "component": {{
             "name": "...", "cls": "...", "mount": "...",
             "file": "{COMPONENTS_DIR.relative_to(ROOT)}/...py",
             "trust": {{...}}
           }},
           "workflow_patch": {{
             "op": "add_node|replace_node|disable_node",
             "name": "<COMPONENT.name>",
             "file": "{COMPONENTS_DIR.relative_to(ROOT)}/...py",
             "edges_in": [], "edges_out": []
           }}
         }}
       }}
  7. Print one final line: CANDIDATE: <candidate_name>

CRITICAL:
  - Do NOT run benchmarks. Do NOT modify Toolathlon-src/, bench/toolathlon/,
    toolathlon_runner.py, agent_toolathlon/runtime/, agent_toolathlon/v0/,
    agent_toolathlon/cr/, agent_toolathlon/component_runtime/ (the runtime
    itself is locked; you only write components in
    {COMPONENTS_DIR.relative_to(ROOT)}/).
  - The target inference model (deepseek-v4-pro via Together AI) is
    LOCKED by toolathlon_runner.py. Do NOT attempt to override it.
  - v2 wraps MCP tools as SDK FunctionTools, so PRE_TOOL_USE REWRITE_TOOL_ARGS
    and true BLOCK work even in single_turn_mode; POST_TOOL_USE
    INJECT_CONTEXT is CONCATENATED into the tool result string and is
    visible to the LLM on its very next inference. See SKILL.md
    §"v2 mount semantics".
  - DEFER and POST_LLM_RESPONSE are STILL rejected at registration in v2
    (planned for v2.5 / v3 respectively). Do NOT design components that
    need them.
  - No task-specific hardcoding (no train task_ids, entity names, or
    fixed strings from any individual task in the component file).

NEW (Phase D event-runtime additions; see SKILL.md "Event runtime additions"):
  - Tier-1 events fire alongside the v1/v2 hooks path. The post-Runner
    subset is emitted in v1: task_received / pre_context_build /
    pre_agent_construct / post_llm_response_raw / on_length_truncation /
    on_empty_response / on_explicit_terminate / session_end. The 5
    SDK-internal events (pre_llm_request / pre_tool_arg_validation /
    post_tool_result_raw / on_tool_error / on_no_tool_call_emitted) are
    declared but NOT emitted in v1 (deferred to v3 ModelProvider wrap).
  - You MAY use `listens="on_explicit_terminate"` to install an artifact
    gate (verify workspace file exists before allowing the agent to
    terminate). A BLOCK decision from the subscriber refuses termination
    and the agent loop continues.
  - `ctx.chat(messages, max_tokens=..., temperature=...)` is available for
    sub-LLM verifier patterns. It routes through agent.llm.chat (the SAME
    locked SUT model name the SDK Runner uses; NOT the SDK ModelProvider).
    No `model=` kwarg. Declare `capabilities=(Capability.LLM_CALL,)` if used.
  - `ctx.emit("iter<N>_<slug>_<event>")` / `ctx.emit_upstream(...)` let two
    components coordinate within one task. Declare `emits=(...)` on the
    publisher for audit / discovery.
"""


def run_proposer(iteration: int, log_dir: Path, skill_name: str,
                 candidate_slug_prefix: str) -> dict:
    print(f"\n=== toolathlon iter {iteration}: proposer starting "
          f"(skill={skill_name}) ===", flush=True)
    max_attempts = 6
    for attempt in range(1, max_attempts + 1):
        result = claude_wrapper.run(
            prompt=_proposer_prompt(iteration, skill_name, candidate_slug_prefix),
            model=PROPOSER_MODEL,
            allowed_tools=PROPOSER_TOOLS,
            cwd=str(ROOT),
            log_dir=str(log_dir),
            name=f"toolathlon_iter{iteration}_proposer",
            skills=[skill_name],
            skill_dir=str(SKILLS_PARENT),
            disable_skills=True,
            disable_mcp=True,
            progress=True,
            docker_skill=skill_name,
            docker_container_name=f"robagent-proposer-toolathlon-iter{iteration}-"
                                  f"{int(time.time())}",
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


# ---------------------------------------------------------------------------
# Train eval.
# ---------------------------------------------------------------------------


def run_eval(workflow: Workflow, iteration: int, agent_name: str,
             hypothesis: str, changes: str, plugin_payload: dict) -> dict:
    active = list(workflow.active_nodes())
    print(f"\n=== toolathlon eval iter {iteration} active={active} on "
          f"train-{len(_read_train_ids())} ===", flush=True)

    summary_jsonl = LOGS / f"iter{iteration}_train_summary.jsonl"
    env = os.environ.copy()
    env["COMPONENT_WORKFLOW"] = str(WORKFLOW_YAML)
    env["COMPONENT_NAMES"] = ",".join(active)
    env["COMPONENT_DIR"] = str(COMPONENTS_DIR)
    env["COMPONENT_RUN_TAG"] = f"toolathlon_iter{iteration}"

    cmd = [
        sys.executable, "-u", str(ROOT / "toolathlon_runner.py"),
        "--candidate", "cr",
        "--task-ids-file", str(TRAIN_IDS_FILE),
        "--max-concurrency", str(TRAIN_PARALLEL),
        "--save-to", str(summary_jsonl),
        # Pin each iter's dumps under cr/iter{N}/finalpool/<tid>/ so the
        # next iter's proposer can still read this iter's trace via
        # evolution_summary[*].per_task[*].dump_dir.
        "--run-tag", f"iter{iteration}",
    ]
    started = time.time()
    ret = subprocess.run(cmd, cwd=ROOT, env=env)
    elapsed = time.time() - started
    print(f"  exit={ret.returncode} elapsed={elapsed:.0f}s", flush=True)
    if ret.returncode != 0:
        raise SystemExit(f"toolathlon_runner.py exited {ret.returncode}")

    n_correct = 0
    per_task: dict[str, dict] = {}
    for line in summary_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        passed = bool(rec.get("passed"))
        n_correct += int(passed)
        per_task[rec["task_id"]] = {
            "passed": passed,
            "tier": rec.get("tier"),
            "dump_dir": rec.get("dump_dir"),
            "score": rec.get("score"),
            "error": rec.get("error"),
        }
    n_total = len(per_task)
    print(f"  train_correct = {n_correct}/{n_total}", flush=True)

    row = {
        "iteration": iteration,
        "agent_name": agent_name,
        "hypothesis": hypothesis,
        "changes": changes,
        "plugin": plugin_payload,
        "train_score": {"n_correct": n_correct, "n_total": n_total},
        "per_task": per_task,
        "summary_jsonl": str(summary_jsonl),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    with EVOLUTION_SUMMARY.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")
    return row


# ---------------------------------------------------------------------------
# Champion gate + patch rollback.
# ---------------------------------------------------------------------------


def _previous_accepted_correct() -> int:
    """Champion-gate: n_correct of the most recent ACCEPTED row.

    iter 0 bootstrap row counts as accepted (the bootstrap writes
    accepted=True). Rejected candidates never inflate the gate.
    """
    if not EVOLUTION_SUMMARY.exists():
        raise SystemExit(
            "evolution_summary.jsonl missing. Run "
            "`--bootstrap-from <baseline.jsonl>` first."
        )
    for line in reversed(EVOLUTION_SUMMARY.read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("accepted") is True:
            return int(row.get("train_score", {}).get("n_correct", 0))
    raise SystemExit(
        "no accepted row in evolution_summary.jsonl; rerun bootstrap."
    )


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
                  f"{patch.file}.bak_iter<N>; file remains as-is", flush=True)


def _commit_patch(patch: Patch, backup: tuple[Path, Path] | None) -> None:
    if patch.op is PatchOp.REPLACE_NODE and backup is not None:
        _, bak = backup
        if bak.exists():
            bak.unlink()


def _score_and_update(iteration: int, candidate_row: dict,
                      prev_workflow: Workflow, next_workflow: Workflow,
                      patch: Patch,
                      backup: tuple[Path, Path] | None) -> bool:
    prev_correct = _previous_accepted_correct()
    cand_correct = int(candidate_row.get("train_score", {}).get("n_correct", 0))
    accepted = cand_correct >= prev_correct

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
        # Refresh frontier_val.json with the new champion's per_task map.
        per_task = candidate_row.get("per_task", {})
        FRONTIER_VAL.write_text(json.dumps({
            "iteration": iteration,
            "n_correct": cand_correct,
            "n_total": last["train_score"]["n_total"],
            "per_task": per_task,
            "summary_jsonl": candidate_row.get("summary_jsonl"),
        }, indent=2))
        print(f"  accepted (train {cand_correct} ≥ {prev_correct}); "
              f"frontier → {list(next_workflow.active_nodes())}", flush=True)
    else:
        print(f"  REJECTED (train {cand_correct} < {prev_correct}); "
              f"rolling back patch", flush=True)
        _rollback_patch(prev_workflow, patch, backup)
    return accepted


# ---------------------------------------------------------------------------
# Final test eval (held-out 49 task ids).
# ---------------------------------------------------------------------------


def final_test_eval() -> None:
    wf = _load_frontier_workflow()
    print(f"\n=== final test eval: workflow active={list(wf.active_nodes())} ===",
          flush=True)
    summary_jsonl = LOGS / f"final_test_summary.jsonl"
    env = os.environ.copy()
    env["COMPONENT_WORKFLOW"] = str(WORKFLOW_YAML)
    env["COMPONENT_NAMES"] = ",".join(wf.active_nodes())
    env["COMPONENT_DIR"] = str(COMPONENTS_DIR)
    env["COMPONENT_RUN_TAG"] = "toolathlon_final_test"
    cmd = [
        sys.executable, "-u", str(ROOT / "toolathlon_runner.py"),
        "--candidate", "cr",
        "--task-ids-file", str(TEST_IDS_FILE),
        "--max-concurrency", str(TRAIN_PARALLEL),
        "--save-to", str(summary_jsonl),
        "--run-tag", "final_test",
    ]
    started = time.time()
    ret = subprocess.run(cmd, cwd=ROOT, env=env)
    print(f"  exit={ret.returncode} elapsed={time.time()-started:.0f}s",
          flush=True)
    if ret.returncode != 0:
        print(f"  WARN: test eval exited {ret.returncode}", flush=True)
    # Aggregate.
    recs = [json.loads(l) for l in summary_jsonl.read_text().splitlines() if l.strip()]
    passed = sum(1 for r in recs if r.get("passed"))
    print(f"  test_correct = {passed}/{len(recs)}", flush=True)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=1,
                   help="number of evolution iterations to run")
    p.add_argument("--train-parallel", type=int, default=8)
    p.add_argument("--final-test", action="store_true",
                   help="after iterations finish, run accepted frontier on test set")
    p.add_argument("--proposer-only", action="store_true")
    p.add_argument("--skill", default=DEFAULT_SKILL,
                   help=f"skill under {SKILLS_PARENT.relative_to(ROOT)}/")
    p.add_argument("--candidate-slug-prefix", default="mh_toolathlon",
                   help="prefix for candidate names")
    p.add_argument("--bootstrap-from", type=Path, default=None,
                   help="initialise iter 0 from an existing baseline jsonl "
                        "and exit (no proposer / no eval). Required before the "
                        "first iter unless logs_components_toolathlon already "
                        "has evolution_summary.jsonl.")
    args = p.parse_args()

    global TRAIN_PARALLEL
    TRAIN_PARALLEL = args.train_parallel

    if args.bootstrap_from is not None:
        bootstrap_from_baseline(args.bootstrap_from.resolve())
        return

    LOGS.mkdir(parents=True, exist_ok=True)
    if not (SKILLS_PARENT / args.skill / "SKILL.md").exists():
        raise SystemExit(f"skill not found: {SKILLS_PARENT / args.skill}/SKILL.md")

    proposer_log_dir = LOGS / "proposer_sessions"
    proposer_log_dir.mkdir(parents=True, exist_ok=True)

    for _ in range(args.iterations):
        iteration = _current_iteration()
        prev_workflow = _load_frontier_workflow()

        pending = run_proposer(iteration, proposer_log_dir, args.skill,
                               args.candidate_slug_prefix)
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
