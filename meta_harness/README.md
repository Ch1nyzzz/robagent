# Meta-Harness

The outer evolution loops that drive component-harness-tau2 (banking_knowledge)
and component-harness-gaia (GAIA) — both inspired by Stanford IRIS-Lab
Meta-Harness (https://github.com/stanford-iris-lab/meta-harness), adapted to
the typed workflow-graph component model.

## Idea

Each iteration:
1. A headless `claude -p` session loads the appropriate component-harness
   skill (`component-harness-tau2` or `component-harness-gaia`), reads the
   workflow graph + frontier + failed traces, proposes ONE workflow patch
   (add_node / replace_node / disable_node) plus a new component file
   (`agent/components/<name>.py` for GAIA or `agent_tau2/components/<name>.py`
   for tau2), and writes `pending_eval.json`.
2. The outer loop applies the patch to the frontier workflow YAML, sets
   `COMPONENT_NAMES` / `COMPONENT_WORKFLOW` / `COMPONENT_RUN_TAG` env vars,
   and runs the benchmark with `--candidate component_runtime` (the fixed
   graph runtime that resolves the active set from the YAML).
3. Score the candidate workflow on train-30, update per-task frontier,
   append `evolution_summary.jsonl`. Accept patch if not regressed; on
   reject, roll back the YAML and (for `add_node`) delete the new file or
   (for `replace_node`) restore from `.bak_iter<N>`.

After N iterations, optionally re-score the frontier on the held-out test set
(`--final-test`).

## Layout

```
meta_harness/
  README.md
  meta_harness_components.py        # tau2 outer loop
  meta_harness_components_gaia.py   # GAIA outer loop
  claude_wrapper.py                 # vendored claude -p subprocess driver
  .claude/skills/
    component-harness-tau2/         # tau2 proposer SKILL.md + templates + patterns
    component-harness-gaia/         # GAIA proposer SKILL.md + templates + patterns
  scripts/
    select_split.py                 # GAIA train/test split
    score_candidate.py              # update frontier + summary from a run
    durability_audit.py             # post-champion audit (--source component)
  workflows/
    tau2_main.yaml                  # tau2 frontier workflow graph
    gaia_main.yaml                  # GAIA frontier workflow graph
  train_task_ids.txt                # GAIA: 30 task_ids stratified by L1/L2/L3
  test_task_ids.txt                 # GAIA: 135 task_ids (held-out)
  tau2_train_task_ids.txt           # tau2: 30 task_ids of banking_knowledge
  tau2_test_task_ids.txt            # tau2: 67 task_ids (held-out)
  logs_components_gaia/             # GAIA: frontier + evolution_summary + proposer sessions
  logs_tau2_components/             # tau2: same shape, tau2 frontier
```

## Quick start

```bash
# GAIA: 20-iter graph evolution + final test on test-135
python meta_harness/meta_harness_components_gaia.py \
    --iterations 20 --train-parallel 8 --test-parallel 8 --final-test

# tau2 (Together AI endpoint): 20-iter graph evolution
TAU2_LLM_ENDPOINT=together MH_PROPOSER_MODEL=opus \
python meta_harness/meta_harness_components.py \
    --iterations 20 --train-parallel 8

# Smoke-test the GAIA proposer only (writes pending_eval.json, no eval)
python meta_harness/meta_harness_components_gaia.py --proposer-only

# Smoke-test the tau2 proposer only
TAU2_LLM_ENDPOINT=together \
python meta_harness/meta_harness_components.py --proposer-only
```

## Candidate naming

Component files land at `agent/components/<name>.py` (GAIA) or
`agent_tau2/components/<name>.py` (tau2). Each file exports
`COMPONENT: Component` with a stable `name` field. To MODIFY an existing
component, reuse the same `name` in a new file and use the `replace_node`
patch op (the outer loop requires a `.bak_iter<N>` copy first; SKILL.md
documents this).

## Endpoints

| benchmark | runner | default endpoint | switch |
|---|---|---|---|
| GAIA | `run_benchmark.py` via `agent/llm.py` | DeepSeek official | — |
| tau2 | `tau2_runner.py` | DeepSeek official (`deepseek-v4-pro`) | `TAU2_LLM_ENDPOINT=together` for `deepseek-ai/DeepSeek-V4-Pro` on Together AI |

## See also

- `RESULTS.md` (project root): pre-graph A/B numbers and the analysis that
  motivated the graph refactor (interpretation-layer compounding ε).
- `agent/component_runtime/` and `agent_tau2/component_runtime/`: the two
  graph runtimes (types / policy matrix / registry / workflow / runner).
