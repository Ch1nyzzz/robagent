"""Outer evolution loop for hook-harness-tau2.

Sibling of meta_harness_tau2.py. The differences:

  * The proposer runs the `hook-harness-tau2` skill, whose artefact is a
    single hook file under `agent_tau2/hooks/`, not a new agent directory.
  * The "candidate" evaluated by tau2_runner is always the fixed
    `hook_runtime` module; the active hook set is selected at runtime via
    the env var `HOOK_NAMES` (`agent_tau2/hook_runtime/agent.py::build_agent`
    reads it).
  * The frontier is a SET of hook names, not a single agent. Each iteration
    composes  (frontier ∪ {add}) / (frontier with name replaced for modify)
    / (frontier \\ {removed}) and scores that union.

State files live under a separate logs dir (default
`meta_harness/logs_tau2_hooks/`) so this loop never collides with the
robust-harness-tau2 / meta-harness-tau2 frontier.

Usage:
  python meta_harness/meta_harness_hooks.py --iterations 1 --proposer-only
  python meta_harness/meta_harness_hooks.py --iterations 5
"""
from __future__ import annotations

import argparse
import contextlib
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

import claude_wrapper  # noqa: E402

DEFAULT_LOGS = THIS_DIR / "logs_tau2_hooks"
TRAIN_IDS = THIS_DIR / "tau2_train_task_ids.txt"
TEST_IDS = THIS_DIR / "tau2_test_task_ids.txt"
SKILLS_PARENT = THIS_DIR / ".claude" / "skills"
DEFAULT_SKILL = "hook-harness-tau2"
DOMAIN = "banking_knowledge"
HOOK_RUNTIME_CANDIDATE = "hook_runtime"  # the fixed agent_tau2/hook_runtime/

LOGS: Path
PENDING_EVAL: Path
FRONTIER_VAL: Path
FRONTIER_HOOKS: Path           # JSON: {"names": [...]} — the active hook set
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

# Paths the proposer must not see. Patterns are evaluated at run time so a
# freshly-added iter directory is also hidden.
_ISOLATE_GLOBS = [
    ("agent_tau2", "mh_tau2_iter*"),       # every prior LLMAgent-subclass candidate
    ("meta_harness", "logs_tau2*"),         # robust / baseline frontiers; logs_tau2_hooks excluded below
]
_ISOLATE_KEEP_LOGS_DIR = "logs_tau2_hooks"

# Memory files whose entire content is "what we tried in iter <N>" rather than
# domain knowledge. The merely-touches-an-iter-number files are kept; the
# proposer prompt explicitly tells the proposer to ignore stray iter mentions.
_ISOLATE_MEMORY_FILES = [
    "tau2-frontier-fail-decomposition.md",  # 100% iter / frontier viewpoint
    "hook-harness-tau2-design.md",          # contains the prior iter1 hook write-up
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
    """Filter MEMORY.md so links to hidden files are gone. Return the original
    content for restoration. Returns None if MEMORY.md is absent."""
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
def _isolate_legacy(root: Path):
    """Hide prior-iter artefacts so the proposer cannot read them.

    Each target is renamed under a timestamped stash directory; MEMORY.md
    is rewritten to drop links to hidden memory files. On exit (success or
    exception) every move is reversed and the stash is removed. If the
    process is killed mid-section, the stash is left in place with a
    `moved.json` manifest so a human / next run can restore manually.
    """
    targets = _isolation_targets(root)
    ts = time.strftime("%Y%m%d_%H%M%S")
    stash = root / f".hook_proposer_hidden_{ts}"
    stash.mkdir(parents=True, exist_ok=False)
    manifest_path = stash / "moved.json"

    moved: list[tuple[Path, Path]] = []
    memory_md = _MEMORY_DIR / "MEMORY.md"
    memory_original: str | None = None

    try:
        # Move each target to stash with a unique stem so absolute-path
        # collisions are impossible (e.g. two `mh_tau2_iter5_*` dirs).
        manifest: list[dict] = []
        for src in targets:
            stem = f"{abs(hash(str(src))) & 0xffff_ffff:08x}__{src.name}"
            dst = stash / stem
            src.rename(dst)
            moved.append((src, dst))
            manifest.append({"src": str(src), "dst": str(dst)})
        manifest_path.write_text(json.dumps(manifest, indent=2))

        # Filter MEMORY.md after the file moves so the new link-targets
        # really don't exist.
        memory_original = _hide_memory_index_lines(
            memory_md, hidden_basenames=set(_ISOLATE_MEMORY_FILES)
        )

        print(f"  isolation: hid {len(moved)} paths into {stash.name}/", flush=True)
        yield
    finally:
        # Restore in reverse so partially-restored state never points at a
        # mid-rename. Errors are reported but never re-raised — restoration
        # must run to completion even if some targets are already gone.
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


def _frontier_hook_names() -> list[str]:
    if not FRONTIER_HOOKS.exists():
        return []
    return list(json.loads(FRONTIER_HOOKS.read_text()).get("names", []))


def _save_frontier_hook_names(names: list[str]) -> None:
    FRONTIER_HOOKS.write_text(json.dumps({"names": sorted(set(names))}, indent=2))


def _proposer_prompt(iteration: int, skill_name: str, slug_prefix: str) -> str:
    name_pattern = (
        f"hook_iter{iteration}_{slug_prefix}_<slug>" if slug_prefix
        else f"hook_iter{iteration}_<slug>"
    )
    return f"""You are a tau2-bench hook proposer. Follow the SKILL.md ({skill_name}).

Iteration: {iteration}
Working directory: {ROOT}

State files (relative to working directory):
  - Frontier per-task best:   {FRONTIER_VAL.relative_to(ROOT)}
  - Frontier hook set:        {FRONTIER_HOOKS.relative_to(ROOT)}
  - Prior iteration summary:  {EVOLUTION_SUMMARY.relative_to(ROOT)}
  - Output pending eval to:   {PENDING_EVAL.relative_to(ROOT)}
  - Train task ids:           meta_harness/tau2_train_task_ids.txt (30 banking_knowledge tasks)
  - Base agent (read-only):   agent_tau2/v0/agent.py  (stock tau2 LLMAgent)
  - Hook runtime (read-only): agent_tau2/hook_runtime/  (do not modify)
  - Existing hooks:           agent_tau2/hooks/  (read-only; modify only by reusing HOOK.name)
  - Reference example hook:   agent_tau2/hooks/close_account_strip_optional_reason.py
  - tau2 sim records dir:     tau2-bench-src/data/simulations/tau2-runs/meta/
      * hook-loop sims:        <dir>/{HOOK_RUNTIME_CANDIDATE}__{DOMAIN}.json/results.json
      * BOOTSTRAP (iter 1, no hook sims yet): read v0's sim at
        <dir>/v0__{DOMAIN}.json/results.json — that is the stock base
        agent on the same 30 tasks and tells you which tasks v0 fails.
        Do NOT look for or read any `mh_tau2_iter*` sim or directory —
        those are from the sibling robust-harness-tau2 skill and are
        physically hidden during your run; this is by design.
  - Hook fire trace:          .hook-state/iter<N-1>/fired.jsonl (absent on iter 1)

Steps:
  1. Read frontier_val.json, frontier_hooks.json, and the last few entries
     of evolution_summary.jsonl. If this is iter 1 (hook frontier empty),
     the relevant base sim is v0's at
     tau2-bench-src/data/simulations/tau2-runs/meta/v0__{DOMAIN}.json/results.json.
  2. Pick 4-6 train task_ids the current hook frontier (or v0 on iter 1)
     fails. Read their tau2 simulation (messages + reward_info) under
     tau2-bench-src/data/simulations/tau2-runs/meta/. Cross-reference
     .hook-state/iter<N-1>/fired.jsonl (when present) to see which
     frontier hooks fired on them.
  3. Form ONE hypothesis tied to a mechanism present in >=3 of those sims.
  4. Choose ONE operation: add | modify | remove. Create exactly one file
     at agent_tau2/hooks/{name_pattern}.py exporting `HOOK: Hook`. For
     `modify`, reuse the existing HOOK.name in the new file. For `remove`,
     do not write a hook file; just name the removed hook in the manifest.
  5. Validate registration:
       python -c "
       from agent_tau2.hook_runtime.registry import load_hooks_from_dir
       hooks = load_hooks_from_dir(only=['<HOOK.name>'])
       assert any(h.name == '<HOOK.name>' for hs in hooks.values() for h in hs)
       print('ok')
       "
  6. Write {PENDING_EVAL.relative_to(ROOT)} with the JSON schema from
     SKILL.md (operation, name, file, class, event, decision_kinds,
     activation_predicate, generalization_argument, fallback, dead_when).
  7. Print one final line: CANDIDATE: <candidate_name>

CRITICAL:
  - Do NOT run benchmarks. Do NOT modify tau2-bench-src/, tau2_runner.py,
    meta_harness/*, agent_tau2/v0/, agent_tau2/hook_runtime/, or any
    earlier hook file (modify-by-reuse-name is allowed; in-place edit
    of someone else's hook file is not).
  - Do NOT build a new agent directory under agent_tau2/. That's the
    robust-harness-tau2 skill; this skill only writes hook files.
  - **Isolation invariant**: agent_tau2/mh_tau2_iter*, meta_harness/logs_tau2_*
    (except logs_tau2_hooks), and a small set of iter-viewpoint memory
    files are physically moved out of the tree before you start. If you
    discover a path that names "mh_tau2_iter" or a robust/baseline frontier
    log, the isolation has a bug — do not read it, treat the contents as
    out-of-scope.
  - Stray references to "iter<N>" in the surviving memory files are
    historical chatter; treat the underlying *fact* as domain knowledge
    but do NOT use them as design indicators for what hook to build.
  - No task-specific hardcoding. No customer names / order ids in code.
"""


def run_proposer(iteration: int, log_dir: Path, skill_name: str, slug_prefix: str) -> dict:
    print(f"\n=== hook iter {iteration}: proposer starting (skill={skill_name}) ===", flush=True)
    max_attempts = 6
    for attempt in range(1, max_attempts + 1):
        result = claude_wrapper.run(
            prompt=_proposer_prompt(iteration, skill_name, slug_prefix),
            model=PROPOSER_MODEL,
            allowed_tools=PROPOSER_TOOLS,
            cwd=str(ROOT),
            log_dir=str(log_dir),
            name=f"hook_iter{iteration}_proposer_{skill_name}",
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
            print(f"  proposer attempt {attempt} failed; retry in {wait}s", flush=True)
            time.sleep(wait)
    raise SystemExit(f"proposer failed in iter {iteration} after {max_attempts} attempts")


def _hook_set_after(operation: str, name: str) -> list[str]:
    """Apply (add | modify | remove) to the frontier hook set."""
    names = set(_frontier_hook_names())
    if operation == "add" or operation == "modify":
        names.add(name)
    elif operation == "remove":
        names.discard(name)
    else:
        raise SystemExit(f"unknown hook operation: {operation!r}")
    return sorted(names)


def run_eval(hook_names: list[str], iteration: int, ids_file: Path,
             parallel: int, label: str) -> Path:
    print(f"\n=== hook eval iter {iteration} hooks={hook_names} on {label} ===", flush=True)
    env = os.environ.copy()
    env.setdefault("TAU2_DATA_ROOT", str(ROOT / "tau2-bench-src" / "data" / "tau2"))
    env["HOOK_NAMES"] = ",".join(hook_names)
    env["HOOK_RUN_TAG"] = f"iter{iteration}"
    cmd = [
        sys.executable, str(ROOT / "tau2_runner.py"),
        "--candidate", HOOK_RUNTIME_CANDIDATE,
        "--domain", DOMAIN,
        "--task-ids-file", str(ids_file),
        "--max-concurrency", str(parallel),
    ]
    started = time.time()
    ret = subprocess.run(cmd, cwd=ROOT, env=env)
    print(f"  exit={ret.returncode} elapsed={time.time()-started:.0f}s", flush=True)
    if ret.returncode != 0:
        raise SystemExit(f"tau2_runner exited {ret.returncode}")
    summary = ROOT / "traces" / f"tau2_{HOOK_RUNTIME_CANDIDATE}__summary.jsonl"
    if not summary.exists():
        raise SystemExit(f"expected summary missing: {summary}")
    return summary


def _previous_frontier_correct() -> int:
    """train_score.correct of the previous accepted hook set (== current
    frontier). Read BEFORE score_candidate.py runs, so it reflects the
    state we're comparing against, not the per-task best after merge."""
    if not FRONTIER_VAL.exists():
        return 0
    fv = json.loads(FRONTIER_VAL.read_text())
    return int(fv.get("frontier_score", {}).get("correct", 0))


def score_and_update(iteration: int, pending: dict, summary_path: Path,
                     prev_hook_set: list[str], candidate_hook_names: list[str]) -> None:
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
    hook = cand.get("hook")
    if hook:
        # score_candidate.py accepts --plugin-json verbatim; same shape as
        # hook manifests, so it lands in evolution_summary.jsonl unchanged.
        cmd += ["--plugin-json", json.dumps(hook)]
    if subprocess.run(cmd, cwd=ROOT).returncode != 0:
        raise SystemExit("score_candidate.py failed")

    # Hook-set acceptance is stricter than score_candidate's per-task best:
    # the union must not regress the overall train score. (A per-task best
    # update is still recorded in frontier_val for analysis.)
    last_line = EVOLUTION_SUMMARY.read_text().splitlines()[-1]
    last = json.loads(last_line)
    cand_correct = int(last.get("train_score", {}).get("correct", 0))
    accepted = cand_correct >= prev_correct

    last["accepted"] = accepted
    last["hook_set_after"] = candidate_hook_names if accepted else prev_hook_set
    # Rewrite the final line with the acceptance fields.
    lines = EVOLUTION_SUMMARY.read_text().splitlines()
    lines[-1] = json.dumps(last, default=str)
    EVOLUTION_SUMMARY.write_text("\n".join(lines) + "\n")

    op = hook["operation"] if hook else "add"
    file_rel = (hook or {}).get("file", "")
    if accepted:
        _save_frontier_hook_names(candidate_hook_names)
        print(f"  accepted (train {cand_correct} ≥ {prev_correct}); "
              f"frontier hook set → {candidate_hook_names}", flush=True)
    else:
        print(f"  REJECTED (train {cand_correct} < {prev_correct}); "
              f"frontier hook set unchanged", flush=True)
        # Clean up the candidate's hook file so a future `add` with the same
        # slug does not silently overwrite a frontier hook. `modify` cannot
        # be cleanly reverted in v1 — warn and leave the file (the next
        # iteration sees the modified file but FRONTIER_HOOKS already lists
        # the original name, which now resolves to the modified file). This
        # is a known v1 limitation; in practice, the next iteration's
        # proposer reads the rejection in evolution_summary.jsonl and
        # decides whether to re-`modify` back.
        if op == "add" and file_rel:
            f = ROOT / file_rel
            if f.exists():
                f.unlink()
                print(f"  cleaned up rejected hook file: {file_rel}", flush=True)
        elif op == "modify":
            print(f"  WARN: modify rejected but {file_rel} was overwritten; "
                  f"the next proposer must inspect and revert if needed", flush=True)
    PENDING_EVAL.unlink()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--train-parallel", type=int, default=8)
    p.add_argument("--proposer-only", action="store_true")
    p.add_argument("--skill", default=DEFAULT_SKILL,
                   help=f"skill under {SKILLS_PARENT.relative_to(ROOT)}/ (default: {DEFAULT_SKILL})")
    p.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS,
                   help="state dir for frontier_val / frontier_hooks / evolution_summary")
    p.add_argument("--candidate-slug-prefix", default="",
                   help="inject prefix into candidate name for parallel A/B runs")
    args = p.parse_args()

    if not (SKILLS_PARENT / args.skill / "SKILL.md").exists():
        raise SystemExit(f"skill not found: {SKILLS_PARENT / args.skill}/SKILL.md")

    global LOGS, PENDING_EVAL, FRONTIER_VAL, FRONTIER_HOOKS, EVOLUTION_SUMMARY
    LOGS = args.logs_dir.resolve()
    LOGS.mkdir(parents=True, exist_ok=True)
    PENDING_EVAL = LOGS / "pending_eval.json"
    FRONTIER_VAL = LOGS / "frontier_val.json"
    FRONTIER_HOOKS = LOGS / "frontier_hooks.json"
    EVOLUTION_SUMMARY = LOGS / "evolution_summary.jsonl"
    if not FRONTIER_HOOKS.exists():
        _save_frontier_hook_names([])

    proposer_log_dir = LOGS / "proposer_sessions"
    proposer_log_dir.mkdir(parents=True, exist_ok=True)

    for _ in range(args.iterations):
        iteration = _current_iteration()
        # Hide prior-iter artefacts ONLY around the proposer; the runner
        # below needs the full tree intact (it loads hook files, which the
        # proposer just wrote).
        with _isolate_legacy(ROOT):
            pending = run_proposer(iteration, proposer_log_dir,
                                   args.skill, args.candidate_slug_prefix)
        if args.proposer_only:
            print("--proposer-only; stopping"); return
        op = pending["candidate"]["hook"]["operation"]
        name = pending["candidate"]["hook"]["name"]
        prev_hook_set = _frontier_hook_names()
        next_hook_set = _hook_set_after(op, name)
        summary = run_eval(next_hook_set, iteration, TRAIN_IDS,
                           args.train_parallel, "train-30")
        score_and_update(iteration, pending, summary, prev_hook_set, next_hook_set)


if __name__ == "__main__":
    main()
