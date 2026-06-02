# Ballast

**Deterministic stabilization scaffolding for LLM agents — evolved automatically by a coding agent.**

> Ballast is the weight a ship carries to stay upright. It adds no propulsion;
> it keeps the vessel from capsizing. This project applies the same idea to LLM
> agents: a layer of deterministic Python that does not extend what the agent
> *can* do — it keeps the agent from failing on what it can *already* do.

## The problem

Most agents today are "an LLM in a while-loop": the model is handed the whole
task and improvises end-to-end. This is fragile. The LLM ends up doing work a
parser, a router, a lookup, or a state transition could do deterministically,
and every one of those steps is a place the system can silently fail or
hallucinate.

The robust form is the opposite: **a deterministic harness that calls the LLM
only at the irreducible decision points**, with everything else — retrieval,
file reading, normalization, validation, retry, control flow — handled by
ordinary Python. (See [12-factor-agents](https://github.com/humanlayer/12-factor-agents)
for the design lineage.)

## What Ballast investigates

Can a **coding agent** make that transition *on its own* — take a naive
LLM-centric agent and, iteration by iteration, grow a layer of deterministic
stabilization around it, without ever touching the model weights or the agent's
core reasoning?

Each iteration, a coding-agent *proposer* reads prior failure traces, forms one
falsifiable hypothesis about a recurring failure mode, and writes **one atomic
stabilization change** — a lifecycle hook — against a frontier. An outer loop
scores it on a train split and keeps it **only if accuracy does not regress**.

## Our optimization vs. the default

This is the core claim, and the repository contains a controlled A/B for it.

The "default" way to point an optimizer at an agent is to say *"go make this
better."* A capable coding agent, given that instruction, does what coding
agents naturally do: it adds interpretation-layer code, special-cases the tasks
it sees, and makes the prompt cleverer. On a train split this works. On held-out
data it overfits.

Ballast changes **one variable**: the design philosophy handed to the same
optimizer (same outer loop, same model, same train/test split). Instead of
*"go make this better"*, the proposer is given a first-principles, deterministic-first
contract (the four principles below) **and** a typed substrate that rejects the
unsafe moves at load time.

| optimizer | design direction | GAIA train-30 | GAIA **test-135** |
|---|---|---|---|
| default | "go improve it" — no direction | 12/30 | 28/135 = 20.7% |
| **Ballast** | first-principles, deterministic-first | 15/30 | **42/135 = 31.1%** |

**+14 held-out tasks (+50% relative)** — and the Ballast run was stopped early
(20 iterations vs the default's 30). The wins generalize because they are
anchored in **deterministic mechanisms** (file parsing, output normalization,
length recovery) that transfer to any task of that shape, rather than in the LLM
happening to know an answer.

The same experiment also shows the *failure* mode it guards against: on
tau2-banking, the undirected optimizer overfit hard (train 19/30 → test 7/67).
`RESULTS.md §4` traces it to stacked "interpretation-layer" code — ~10 layers
each carrying ε≈5% false-positives, compounding to ~40% false-positive on
held-out data. The typed substrate now **rejects that class of change at load
time** (see "guardrails" below). The difference between our optimization and
the default is not a different algorithm — it is direction plus guardrails.

## Design philosophy

Four principles the proposer must not violate (validated in `RESULTS.md §4`):

1. **The main agent is the protagonist.** Hooks stabilize its environment,
   outputs, and retries; they never replace its reasoning. If a change would
   make it "a fundamentally different agent," the change is too heavy.
2. **Capability vs. stabilization — kept strictly separate.** If the agent
   fails because it *cannot reach* something (a new file format, API, or tool),
   the answer is to **register a tool** — a capability change, out of scope for
   a hook. Hooks only stabilize what the agent can already do.
3. **The LLM is the last resort within stabilization.** When you do write a
   hook, find one place the LLM is repeating deterministic work — output
   normalization, tool-arg validation, retry-on-observed-failure — and move it
   into Python.
4. **Code earns its place by capturing stable structure, not by fitting recent
   failures.** A hook must anchor on something *outside* its evidence traces — a
   system field, an LLM API field (`finish_reason`), a tool's JSON schema, a
   general algorithm. A hook induced purely from N failed traces is
   memorization, and memorization is what compounds into the test regression
   above.

## How it works

A hook is a single Python file declaring one `Component`: it `listens` to a
lifecycle event the runtime emits, and reacts with a `Decision`
(`allow` / `block` / `rewrite` / `inject_context`). The runtime fires events
across the agent's loop — per task, per turn, and per tool call (e.g.
`pre_tool_use`, `on_length_truncation`, `pre_answer_emit`).

**Guardrails (the typed substrate).** Each component declares a *class*
(`mechanism_layer` / `reactive_guard` / `induced_rule`) and a *Trust* block
(`evidence_anchor` + `out_of_evidence_probe`). A load-time
**class × event × decision matrix** rejects unsafe combinations: an
`induced_rule` (a reading of policy text) can only inject *advisory* context —
it can never mechanically override the LLM — and a `predictive_heuristic`
(a hook keyed on raw prompt/response surface text) is rejected outright. This is
the mechanical embodiment of principle 4.

**The outer loop** is benchmark-specific but always the same shape: clone /
snapshot the current frontier → run the proposer (one atomic change, no
benchmark execution) → score the candidate on the train split → accept iff train
accuracy does not regress, else roll back. Two frontier representations exist:

- **graph frontier** (GAIA / tau2 / SOP-Bench / Toolathlon): the active hook set
  is a node list in a workflow YAML; a change is an `add` / `replace` / `disable`
  patch.
- **directory-as-frontier** (EnterpriseOps): the whole agent directory is the
  frontier; the loop clones it, the proposer edits anything inside, and a
  `current_<domain>` symlink repoints to the latest accepted clone.

## Repository layout

```
ballast/                     the evolution engine (the project)
  ├─ evolve_gaia.py            outer loop per benchmark
  ├─ evolve_tau2.py            (formerly meta_harness_components_*.py)
  ├─ evolve_enterpriseops.py
  ├─ evolve_sopbench.py
  ├─ evolve_toolathlon.py
  ├─ component_runtime_core/   shared runtime: dispatcher, policy matrix, types
  ├─ .claude/skills/           the proposer contracts (SKILL.md per benchmark) —
  │                            this is where the design philosophy is encoded
  ├─ workflows/                graph-frontier YAML (active hook set per benchmark)
  └─ logs_components_*/         lightweight experiment record (frontier + summary)

agent/                       GAIA reference agent + its evolved hooks
  ├─ base.py                   tool-using FC loop (file_read / url_fetch / web_search / python_exec)
  ├─ component_runtime/        GAIA lifecycle runtime
  └─ components/               evolved hooks (the products of evolution)
agent_tau2/ agent_toolathlon/ agent/enterpriseops/   per-benchmark agents

bench/                       official scorers (read-only)
run_benchmark.py             GAIA candidate runner   ┐ thin eval adapters; the
tau2_runner.py               tau2 candidate runner   │ benchmarks are validation
toolathlon_runner.py         Toolathlon runner       ┘ surfaces, not the product
RESULTS.md                   the A/B numbers + the overfit analysis (§4)
```

**On benchmarks.** Ballast is the *method*; the benchmarks are where it was
validated. The datasets and simulators belong to their upstream projects (GAIA,
[tau2-bench](https://github.com/sierra-research/tau2-bench), Toolathlon,
SOP-Bench, EnterpriseOps-Gym) and are **never vendored here** — clone/install
them locally (see `.gitignore`). The runner scripts are thin adapters that let
the outer loop score a candidate; the lightweight experiment record
(`frontier_val.json` + `evolution_summary.jsonl`) is checked in as evidence.

## Reproduce

```bash
pip install -r requirements.txt
# .env (never committed) needs:
#   DEEPSEEK_API_KEY     DeepSeek official API (the locked SUT model)
#   TAVILY_API_KEY       web_search tool (optional; DuckDuckGo fallback)
#   TOGETHER_AI_API      Together AI endpoint (optional, tau2)
#   HF_TOKEN             GAIA dataset loading

# GAIA baseline (no hooks)
python run_benchmark.py gaia --agent-version v0

# GAIA: 20-iteration evolution + final test on test-135
python ballast/evolve_gaia.py --iterations 20 --train-parallel 8 \
    --test-parallel 8 --final-test

# tau2 baseline on banking_knowledge
python tau2_runner.py --candidate v0 --domain banking_knowledge \
    --task-ids-file ballast/tau2_train_task_ids.txt --max-concurrency 8

# tau2: 20-iteration evolution (Together endpoint, Opus proposer)
TAU2_LLM_ENDPOINT=together MH_PROPOSER_MODEL=opus \
python ballast/evolve_tau2.py --iterations 20 --train-parallel 8
```

## Status

Active research. The pre-graph A/B above (the `default` vs `robust` skill in
`RESULTS.md`) is the headline evidence. The typed component runtime is the
current substrate; it exists to reproduce the GAIA held-out gain while
mechanically preventing the tau2-banking-style test regression. Contributions
that add a benchmark adapter, a lifecycle event, or a well-anchored stabilization
pattern are welcome — see `CONTRIBUTING.md`.

## License

MIT — see `LICENSE`.
