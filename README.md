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
LLM-centric agent and, iteration by iteration, evolve it into a robust harness
where the LLM carries minimal responsibility?

The core hypothesis: **the optimizer needs a design philosophy, not just a
"go improve it" instruction.** "Iteratively optimize this agent" is far too wide
a target — the optimizer doesn't know *what good looks like*. Encoding the
direction (deterministic-first, tools bridge channel gaps, honest blocks over
fabrication) as a reusable **skill** turns ad-hoc tinkering into a repeatable
engineering method.

## How it works

```
meta_harness/                 outer evolution loop
  ├─ spawns a headless coding agent each iteration
  ├─ the agent follows a SKILL.md, reads failure traces, proposes ONE candidate
  └─ scores the candidate on a held-out train/test split

.claude/skills/robust-harness-gaia/   the skill under study
  └─ first principles: LLM is the last resort; tools bridge channel gaps,
     not reasoning gaps; honest blocks beat silent fabrication

agent/        v0 baseline → evolved candidates (v1-v14, mh_iter*)
bench/        GAIA + tau2bench official scorers (eval contract, read-only)
harness/      derived audit gates, invariants, iteration logs
tools/        eval harness used by the orchestrator
```

The benchmark is **GAIA** (general-assistant tasks: web research, file QA,
multi-step reasoning), split into train-30 / test-135.

## Headline result

An A/B run — same orchestrator, same model (`deepseek-v4-pro`), same train/test
split, the **only** variable being the skill's design philosophy:

| skill | champion train-30 | champion **test-135** |
|---|---|---|
| `meta-harness-gaia` (workflow + failure menu, no design direction) | 12/30 | 28/135 = 20.7% |
| `robust-harness-gaia` (first-principles, deterministic-first) | 15/30 | **42/135 = 31.1%** |

**+14 tasks on held-out (+50% relative)** — with the robust skill stopped early
at 20 iterations vs the baseline's 30.

Why it generalizes better: the robust skill's wins are anchored in
**deterministic mechanisms** (file parsing, retrieval) that transfer to any
task of that shape, rather than in the LLM happening to know an answer. The
generalization gap (train→test decay) is correspondingly smaller.

## Status

Research in progress. Next: a 3-rung ladder (naive "just iterate" baseline →
structured skill → first-principles skill) to isolate *design direction* as the
treatment variable.

## Setup

```bash
pip install -r requirements.txt
# .env (not committed) needs: DEEPSEEK_API_KEY, HF_TOKEN
python run_benchmark.py gaia --agent-version v0
python meta_harness/meta_harness.py --iterations 30 --skill robust-harness-gaia
```
