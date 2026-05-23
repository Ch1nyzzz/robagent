---
name: robust-harness-tau2
description: Run ONE iteration of tau2-bench harness evolution. Propose ONE candidate that shifts responsibility from the LLM into deterministic code, classify it on the durability axis, justify why the code generalizes beyond your training evidence, and write a plugin manifest to pending_eval.json.
---

# robust-harness-tau2

Run ONE iteration of agent evolution against tau2-bench. **You do NOT run the simulator.** You understand the domain, analyze prior simulations, propose ONE candidate, classify it on the durability axis, implement it, write `pending_eval.json` (a plugin manifest), and exit. The outer loop (`meta_harness/meta_harness_tau2.py`) evaluates on the train-30 subset.

## First principles

1. **The LLM is the last resort.** Each iteration, find one place the LLM is doing work deterministic code could do — parsing, arithmetic, lookup, comparison, ranking, sequencing, state tracking — and move it into Python.

2. **Decompose; don't defer.** A failure that looks "reasoning-bound" is rarely atomic. Split it: what is the *minimal* judgment the LLM must make, and what computation / lookup / validation / sequencing around it is fully deterministic? Build that deterministic part.

3. **Code earns its place by capturing stable structure, not by fitting recent failures.** Anything you encode must point at a fact that lives OUTSIDE your training evidence — a system field, a tool's declared schema, a protocol invariant, a general algorithm. If your code is a rule INDUCED from the N failed simulations you read, you have written a memorized map of those N simulations, not a piece of harness. The system is code-centric / micro-LLM only when the code generalizes; otherwise you have just moved overfit from the LLM's in-context reasoning into a Python file.

4. **Honest over fabricated.** When the agent cannot satisfy a request within policy, it says so and follows the escalation path — it does not invent an action or a confirmation.

(Inspired by 12-factor-agents — apply the spirit; ignore the letter.)

## The durability axis: classify your candidate

Your candidate IS one plugin: ONE new or modified component wrapping tau2's `LLMAgent`. Its **class** is *derived* from what its activation predicate tests — you do not assert it, the predicate decides.

| class | the component's `if` tests... | when the model gets stronger | risk |
|---|---|---|---|
| `channel` | task structure — a request needing content the agent cannot otherwise reach | stays needed: the model still cannot *reach* it | low |
| `reactive_guard` | an observed failure event — `tool.failed`, a malformed tool call, an empty/looping turn | self-disables: a model that stops failing never triggers it → zero drag | low |
| `deterministic_glue` | nothing — always-on **mechanical** transform/validation on data already in hand. The rule's behavior is fully determined by a system field, a tool-declared schema, a protocol invariant, or a general algorithm — NOT by your reading of a policy document or your N evidence simulations. | durable; harmless even when redundant | low |
| `induced_rule` | always-on rule INDUCED from finite training evidence — your reading of a human-language policy document compiled into branches. The rule's correctness depends on your N failed simulations matching the true policy's structure. | the rule is an MLE on small N; out-of-evidence policy edges expose false positives that can override correct LLM judgments — can turn **negative** | HIGH |
| `predictive_heuristic` | the prompt / conversation text — a regex/keyword guess about what the agent *should* do next | can override the model on a prior and turn **negative** | HIGH |

`channel`, `reactive_guard`, and `deterministic_glue` are all low-risk because their behavior is anchored to facts independent of your training evidence. `induced_rule` and `predictive_heuristic` are HIGH-risk because their behavior rests on an interpretation (of policy text, or of conversation text) that may not survive out-of-evidence cases.

**Preference**: reach for the lowest-risk class that fits the failure. If you find yourself reaching for `induced_rule`, first ask: can a mechanism — better retrieval, structured state tracking, a protocol state machine, advisory context injection that surfaces the relevant policy section — let the LLM make the right judgment on its own?

### The HIGH-risk gate (applies to both `induced_rule` and `predictive_heuristic`)

A HIGH-risk candidate is admissible only when ALL THREE hold — otherwise downgrade or redesign:

1. **≥3 evidence simulations** share the failure mechanism you target.
2. **No low-risk fix is available**: explicitly argue, in the manifest, why no `channel` / `reactive_guard` / `deterministic_glue` move can fix this failure.
3. **Non-destructive deployment**: the rule may NOT silently override the LLM's output. Either pre-filter the case before the LLM sees it, or inject the rule's verdict as advisory context into the LLM's prompt and let the LLM make the final call. A rule induced from finite evidence that hard-overrides the LLM will turn negative on out-of-evidence policy edges.

The `generalization_argument` field of the manifest (below) is where you discharge this gate.

## Durability auditing

tau2's own simulation record (`messages`, `tool_calls`, `reward_info`) is rich enough to see whether a plugin's branch engaged on a given task — no separate event log is needed. Keep each plugin's branch point clean and identifiable so a later audit can tell, after a model upgrade, whether a plugin still fires (durable) or has gone inert on every task (dead weight).

## Hard rules

- Exactly ONE new candidate per invocation. Do not loop. Do not abort early.
- **You do NOT run the simulator** — no `tau2_runner.py`, no `tau2 run`. The outer loop scores your candidate.
- **No task-specific code.** No customer names, account/document ids, or per-task branching — and no encoding of gold answers or gold action sets.
- General documented policy may enter the system as **advisory context** injected into the LLM's prompt (preferred to having the LLM rediscover doc text every turn). Compiling policy into a branch that **overrides** the LLM's output is `induced_rule` and must pass the §HIGH-risk gate.
- READ-ONLY: `tau2-bench-src/`, `tau2_runner.py`, `meta_harness/*`, `agent_tau2/v0/`, and every earlier `agent_tau2/mh_tau2_iter*/`.

## Candidate interface

A candidate is `agent_tau2/<name>/` exposing, from `agent.py`:

```python
def build_agent(tools, domain_policy, **kwargs):
    # returns a tau2 HalfDuplexAgent
    ...
```

`<name>` matches `^mh_tau2_iter\d+_[a-z0-9_]+$`. Add an empty `__init__.py`. The directory MAY contain helper modules; keep `agent.py` the thin wiring layer.

Start from `agent_tau2/v0/agent.py` (the stock `LLMAgent`). An evolved candidate returns a **subclass of `LLMAgent`** (or a thin wrapper) — see `tau2-bench-src/src/tau2/agent/llm_agent.py` for the overridable methods (`generate_next_message`, `get_init_state`). Keep `kwargs["llm"]` / `kwargs["llm_args"]` passed through unchanged.

## Workflow

### 1. Understand the workflow (before any failure analysis)

Before you read a single failed simulation, understand the domain end-to-end. Read the domain's policy document(s), the full tool catalog, and any protocol invariants (unlock/call/give orders for discoverable tools, prerequisite chains, KB structure, session-state requirements). Write a workflow summary in your builder log covering:

- The end-to-end happy path: what does a successful task look like, from authentication to final action.
- The protocol invariants: orderings that are mandatory, calls that cannot be skipped, dependencies between tools.
- The infrastructure the workflow depends on: KB retrieval, session state, prerequisite reads, schema validation, conversation budget.

This summary is your reference frame for what stable structure looks like in this domain. You will form hypotheses against THIS, not just against whatever the visible failure surface highlights — the latter biases you toward visible policy errors and away from infrastructure gaps.

### 2. Read state and failed simulations

```
meta_harness/logs_tau2/frontier_val.json        per-task best across candidates
meta_harness/logs_tau2/evolution_summary.jsonl  every prior candidate + score + plugin manifest
meta_harness/tau2_train_task_ids.txt            30 tasks — your pool
```

(Exact paths are in your runtime prompt — use those.)

Pick 4-6 train tasks the frontier still fails (score 0). For each, open the newest tau2 simulation at `tau2-runs/meta/<candidate>__<domain>.json/results.json` and read `simulations[].messages` (user / assistant / tool turns, `tool_calls`) and `simulations[].reward_info`:

- `db_check` — did the database reach the target state?
- `action_checks` — were the expected tool calls made, with matching arguments?
- `nl_assertions`, `communicate_checks` — did the agent say the required things?
- `termination_reason` — normal stop vs error vs max-steps.

### 3. Form ONE hypothesis

> **HYPOTHESIS**: <falsifiable claim about train-30 reward>
> **MECHANISM**: <failure mode> seen in N≥3 task simulations [tid1, tid2, ...]
> **STABLE STRUCTURE**: <the system field / tool schema / protocol invariant / general algorithm your code will capture — must point OUTSIDE your N evidence simulations>
> **FIX**: <what code you build, expressed in terms of that stable structure>
> **CLASS**: <one of the 5 classes — derived from the activation predicate>
> **PREDICTION**: train-30 reward <current> → <expected>

If you cannot name a stable structure outside your evidence, your candidate is `induced_rule` (HIGH risk) and must pass the §HIGH-risk gate.

### 4. Classify

Determine the class from the activation predicate (see the axis table). If your activation depends on the builder's reading of policy text + evidence, it is `induced_rule`. If on a system field, tool schema, or protocol invariant, it is `deterministic_glue` / `channel`. If on an observed failure event, it is `reactive_guard`. If on a guess from conversation text, it is `predictive_heuristic`.

### 5. Implement

Create `agent_tau2/<name>/{__init__.py, agent.py, +helper modules}`. Validate the import:

```bash
python -c "from agent_tau2.<name>.agent import build_agent; print('ok')"
```

This is the ONLY shell command you run.

### 6. Write `pending_eval.json` (plugin manifest)

```json
{
  "iteration": <N>,
  "candidate": {
    "name": "<name>",
    "hypothesis": "<falsifiable claim>",
    "mechanism": "<failure this targets>",
    "evidence_task_ids": ["<tid1>", "<tid2>", "<tid3>"],
    "changes": "<plain-English diff vs prior frontier>",
    "expected_delta": "<expected reward change on train-30>",
    "plugin": {
      "name": "<stable component id — REUSE the existing id if you modify an existing component>",
      "class": "channel | reactive_guard | deterministic_glue | induced_rule | predictive_heuristic",
      "activation_predicate": "<the exact condition the component branches on>",
      "generalization_argument": "<REQUIRED. Answer BOTH:
        (a) Which stable structure does this code capture? Point at something that
            lives OUTSIDE your N evidence simulations — a system field's semantics,
            a tool's declared schema, a protocol invariant, a general algorithm. If
            you cannot answer (a) concretely, your class is induced_rule.
        (b) If induced_rule or predictive_heuristic: name ONE plausible out-of-
            evidence case where this rule could fire incorrectly, and describe what
            your code does there. If you can think of one easily, redesign. If you
            cannot, list the policy edge cases you checked and explain why coverage
            is complete. AND describe how the rule is deployed non-destructively
            (pre-filter, or advisory injection — see §HIGH-risk gate #3).>",
      "fallback": "<what happens when the component does not engage or mis-fires>",
      "dead_when": "<observable condition under which this component is provably dead weight>"
    }
  }
}
```

The outer loop reads `name` / `hypothesis` / `changes` as before; the `plugin` block is recorded in `evolution_summary.jsonl` for the durability axis. `generalization_argument` is the load-bearing field — if you cannot fill it in concretely, you have a candidate that does not belong in the harness.

### 7. Session log + exit

Write a concise log to `meta_harness/logs_tau2/builder_sessions/iter<N>/log.md` — workflow summary, mechanism, hypothesis, the stable structure your code captures, plugin class + why, files written. Final line of your reply:

```
CANDIDATE: <name>
```

## What this skill does NOT do

- Run the tau2 simulator (the outer loop does).
- Modify `tau2-bench-src/`, the eval, the model, `tau2_runner.py`, `meta_harness/`, or prior candidates.
- Task-specific hardcoding — encoding gold answers or branches keyed to specific task ids.
- Loop or propose multiple candidates in one invocation.
- Override the LLM's output with a rule induced from finite evidence, unless that rule passes the §HIGH-risk gate (and even then, only via pre-filter or advisory injection, never as a silent override).
