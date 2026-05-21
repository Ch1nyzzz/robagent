# Meta-Harness for GAIA

Stanford IRIS-Lab Meta-Harness (https://github.com/stanford-iris-lab/meta-harness) adapted to optimize the `robagent` baseline against GAIA validation.

## Idea

Each iteration:
1. A headless `claude -p` session loads `meta-harness-gaia` SKILL.md, reads `frontier_val.json` + `evolution_summary.jsonl` + sampled failed traces from train-30, proposes ONE new agent variant under `agent/mh_iter<N>_<slug>/`, and writes `logs/pending_eval.json`.
2. The outer loop reads `pending_eval.json` and runs `run_benchmark.py gaia --agent-version <candidate> --task-ids-file train_task_ids.txt`.
3. Score the candidate on train-30, update the per-task frontier, append `evolution_summary.jsonl`.

After N iterations, optionally re-score the frontier candidate on the held-out test-135 (`--final-test`).

## Layout

```
meta_harness/
  README.md
  meta_harness.py                 # outer loop
  claude_wrapper.py               # vendored from upstream
  .claude/skills/meta-harness-gaia/SKILL.md  # proposer prior
  scripts/
    select_split.py               # one-time: split 165 → 30 train + 135 test
    score_candidate.py            # update frontier + summary from a candidate's run
  train_task_ids.txt              # 30 task_ids (stratified by L1/L2/L3, seed=7)
  test_task_ids.txt               # 135 task_ids (held-out)
  logs/
    split_meta.json               # baseline scores on train/test from v0 summary
    frontier_val.json             # per-task best agent on train-30
    evolution_summary.jsonl       # one row per iteration
    pending_eval.json             # proposer → runner handoff (deleted after eval)
    proposer_sessions/            # claude -p artifacts per iteration
```

## Quick start

```bash
# 1. (already done) build train/test split + bootstrap v0 frontier
python meta_harness/scripts/select_split.py
python meta_harness/scripts/score_candidate.py \
  --agent-name v0 --summary-path traces/gaia__summary.jsonl --iteration 0 \
  --hypothesis "baseline" --changes "single LLM call max_tokens=2048"

# 2. Run one iteration end-to-end (proposer + eval on train-30)
python meta_harness/meta_harness.py --iterations 1 --train-parallel 4

# 3. Run several iterations then evaluate the best on test-135
python meta_harness/meta_harness.py --iterations 5 --train-parallel 4 --final-test

# 4. Smoke-test the proposer only (writes pending_eval.json + candidate code, no eval)
python meta_harness/meta_harness.py --proposer-only
```

## Baselines (from v0 summary projection)

| split | tasks | v0 correct | v0 acc |
|---|---|---|---|
| train | 30 | 8 | 0.2667 |
| test | 135 | 24 | 0.1778 |

## Candidate naming

Candidates land at `agent/mh_iter<N>_<slug>/`. `run_benchmark.py` resolves `agent_version` starting with `mh` directly to `agent.<name>.base`. The existing v0-v13 line is untouched.

## Differences from upstream `terminal_bench_2/`

| upstream | here |
|---|---|
| Harbor/Terminus2 agent class | `agent.<name>.base.run_task` function contract |
| `agents/*.py` single-file candidate | `agent/<name>/base.py` directory candidate |
| Opus 4.6, full 89 TB2 tasks, 2 trials | DeepSeek-V4-Pro (Together), 30 GAIA tasks, 1 trial |
| Anthropic Pro/API for proposer | `claude -p` subprocess (uses your Pro/API auth) |
| `~$500/iter, 4-6h` | depends on candidate complexity, ~minutes |
