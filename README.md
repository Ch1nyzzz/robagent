# robagent

**Evolving LLM-centric agent workflows into robust, code-dominated deterministic harnesses — automatically.**

## The problem

Most agents today are "an LLM in a while-loop": the model is handed the whole
task and improvised end-to-end. This is fragile — the LLM is doing work that a
parser, a router, a lookup, or a state transition could do deterministically,
and every one of those steps is a place the system can silently fail or
hallucinate.

The robust form is the opposite: **a deterministic state machine that calls the
LLM only at the irreducible decision points**, with everything else — retrieval,
file reading, normalization, validation, control flow — handled by ordinary
Python. (See [12-factor-agents](https://github.com/humanlayer/12-factor-agents)
for the design lineage.)

## What this project investigates

Can a **coding agent** make that transition *on its own* — take a naive
LLM-centric agent and, iteration by iteration, evolve it into a typed
workflow-graph harness where the LLM carries minimal responsibility?

The core hypothesis: **the optimizer needs a design philosophy AND a typed
composition substrate, not just a "go improve it" instruction.** Each iteration
proposes ONE atomic graph patch (add_node / replace_node / disable_node) against
a frontier workflow. Components are typed by their **Mount** (where in the
lifecycle they fire), **Class** (mechanism_layer / reactive_guard / channel /
induced_rule), and **Trust** profile (evidence_anchor + out_of_evidence probe).
A load-time **class × mount × decision matrix** rejects unsafe combinations
(e.g. `induced_rule` can only inject advisory context — it can never
mechanically override the LLM).

## How it works

```
meta_harness/                            outer evolution loops
  ├─ meta_harness_components.py          tau2 graph evolution
  ├─ meta_harness_components_gaia.py     GAIA graph evolution
  ├─ workflows/                          frontier workflow YAML
  └─ .claude/skills/
       ├─ component-harness-tau2/        proposer SKILL.md for tau2
       └─ component-harness-gaia/        proposer SKILL.md for GAIA

agent/                                   GAIA agents
  ├─ base.py                             v0 baseline (single-shot LLM call)
  ├─ component_runtime/                  graph runtime (5 GAIA-specific mounts)
  ├─ components/                         GAIA component files (evolution products)
  └─ mh_iter*/, v1-v14/                  historical evolved agents

agent_tau2/                              tau2 agents
  ├─ v0/                                 v0 baseline (stock tau2 LLMAgent)
  ├─ component_runtime/                  graph runtime (8 tau2-specific mounts)
  ├─ components/                         tau2 component files (evolution products)
  └─ mh_tau2_iter*/                      historical evolved agents

bench/                                   GAIA + tau2bench official scorers (read-only)
tau2_runner.py                           tau2 candidate runner (Together / DeepSeek endpoint switch)
run_benchmark.py                         GAIA candidate runner
```

The benchmarks are **GAIA** (general-assistant tasks: web research, file QA,
multi-step reasoning; train-30 / test-135) and **tau2-bench banking_knowledge**
(banking customer-service simulator; train-30 / test-67).

## Headline result (pre-graph; the `robust` skill in `RESULTS.md`)

An A/B run before the graph-mode refactor — same orchestrator, same model
(`deepseek-v4-pro`), same train/test split, the **only** variable being the
skill's design philosophy:

| skill | champion train-30 | champion **test-135** |
|---|---|---|
| baseline workflow (no design direction) | 12/30 | 28/135 = 20.7% |
| robust (first-principles, deterministic-first) | 15/30 | **42/135 = 31.1%** |

**+14 tasks on held-out (+50% relative)** — with the robust skill stopped early
at 20 iterations vs the baseline's 30. On tau2-banking the same skill
overfit (train 19/30 → test 7/67); `RESULTS.md §4` traces it to
"interpretation-layer code" compounding ε≈5% false-positives across 10 stacked
layers. The graph-mode component runtime is the next attempt: load-time
rejection of `induced_rule` mechanical override + advisory-only injection,
explicit mount/class typing, and three atomic patch ops in place of monolithic
candidate dirs.

Why the deterministic-first direction generalizes: the wins are anchored in
**deterministic mechanisms** (file parsing, retrieval) that transfer to any
task of that shape, rather than in the LLM happening to know an answer. The
generalization gap (train→test decay) is correspondingly smaller — except when
the iter encoded policy interpretation as deterministic code, which the graph
runtime now rejects at load time.

## Status

Active research. Current focus: validate the graph-mode runtime can reproduce
the GAIA test gain AND prevent the tau2-banking test regression. See
`RESULTS.md` for the pre-graph A/B numbers and the analysis that motivated the
graph refactor.

## Setup

```bash
pip install -r requirements.txt
# .env (not committed) needs:
#   DEEPSEEK_API_KEY   for DeepSeek official API
#   TOGETHER_AI_API    for Together AI endpoint (optional)
#   HF_TOKEN           for GAIA dataset loading

# GAIA baseline (v0)
python run_benchmark.py gaia --agent-version v0

# GAIA 20-iter graph evolution + final test on test-135
python meta_harness/meta_harness_components_gaia.py \
    --iterations 20 --train-parallel 8 --test-parallel 8 --final-test

# tau2 baseline (v0) on banking_knowledge
python tau2_runner.py --candidate v0 --domain banking_knowledge \
    --task-ids-file meta_harness/tau2_train_task_ids.txt --max-concurrency 8

# tau2 20-iter graph evolution (Together AI endpoint, Opus builder)
TAU2_LLM_ENDPOINT=together MH_PROPOSER_MODEL=opus \
python meta_harness/meta_harness_components.py \
    --iterations 20 --train-parallel 8
```
