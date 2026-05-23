---
name: meta-harness-tau2
description: Run one iteration of tau2-bench agent evolution. Analyze prior simulations, propose ONE better agent variant, implement it, and write pending_eval.json. The outer loop runs the official tau2 simulator.
---

# meta-harness-tau2

Run ONE iteration of agent evolution against tau2-bench.

**You do NOT run the simulator.** You analyze prior results + failed
simulations, propose ONE better agent variant, implement it, and write
`pending_eval.json`. The outer loop (`meta_harness/meta_harness_tau2.py`) runs
the official tau2 simulator on the train-30 subset.

## Critical constraints

- You MUST produce exactly 1 new agent variant every iteration.
- Do NOT write "the frontier is optimal" or "stop iterating", or abort early.
- Do NOT run the simulator; the outer loop does that.
- Do NOT modify `tau2-bench-src/`, `tau2_runner.py`, `meta_harness/*`,
  `agent_tau2/v0/`, or any earlier `agent_tau2/mh_tau2_iter*/`.

## Anti-overfitting rules

- **No task-specific hints.** Do not hardcode customer names, order ids,
  product ids, or per-task branching.
- **General guidance is OK.** Rules like "confirm details before a write
  action" or "retry a failed tool call once" are fine — they happen to help
  specific tasks but apply broadly.
- If in doubt, make it more general.

## Context

You are evolving a tau2 customer-service agent (domain set by the runner). The
baseline (`agent_tau2/v0/agent.py`) is tau2's stock `LLMAgent` — a turn-based
agent that, each turn, either sends a message to the user or makes a tool call.

A candidate is `agent_tau2/<name>/agent.py` exposing:

```python
def build_agent(tools, domain_policy, **kwargs):
    # returns a tau2 HalfDuplexAgent
    ...
```

`<name>` matches `^mh_tau2_iter\d+_[a-z0-9_]+$`. Add an empty `__init__.py`.
Start from `agent_tau2/v0/agent.py` and modify. An evolved candidate typically
returns a subclass of `LLMAgent` — see
`tau2-bench-src/src/tau2/agent/llm_agent.py` for the overridable methods
(`generate_next_message`, `get_init_state`). Keep `kwargs["llm"]` /
`kwargs["llm_args"]` passed through unchanged.

**Search space**: arbitrary Python in your candidate directory. You can
restructure the agent however you think will raise the train-30 reward —
rewrite the system prompt, add planning or verification LLM calls, add
deterministic checks, restructure the turn loop. Your goal is simply: **propose
a better agent system.**

## Workflow

### Step 1 — Read state

1. `meta_harness/logs_tau2/frontier_val.json` — per-task best + score
2. `meta_harness/logs_tau2/evolution_summary.jsonl` — prior candidates + scores
3. `agent_tau2/v0/agent.py` — the baseline
4. The last 1-3 prior candidate dirs `agent_tau2/mh_tau2_iter*/agent.py` if any
5. **Failed simulation sampling**: pick 4-6 train task_ids the frontier fails
   (score 0). For each, read its newest tau2 simulation at
   `tau2-runs/meta/<candidate>__<domain>.json/results.json` —
   `simulations[].messages` (the conversation) and `simulations[].reward_info`
   (`db_check`, `action_checks`, `nl_assertions`, `termination_reason`).

### Step 2 — Form ONE hypothesis

> **HYPOTHESIS**: "<falsifiable claim about what will raise train-30 reward>"
>
> **MECHANISM**: <observed failure mode>, present in N≥3 task simulations (list them)
>
> **PREDICTION**: train-30 reward will go from <current> to roughly <X>

### Step 3 — Implement

1. Create `agent_tau2/<name>/__init__.py` (empty) and `agent_tau2/<name>/agent.py`.
2. Start from `agent_tau2/v0/agent.py`, make targeted changes for your hypothesis.
3. Validate the import:

```bash
python -c "from agent_tau2.<name>.agent import build_agent; print('ok')"
```

### Step 4 — Write pending_eval.json

```json
{
  "iteration": <N>,
  "candidate": {
    "name": "<name>",
    "hypothesis": "<falsifiable claim>",
    "mechanism": "<which failure this targets>",
    "evidence_task_ids": ["<id1>", "<id2>", "<id3>"],
    "changes": "<plain-English summary of what changed>",
    "expected_delta": "<expected reward change on train-30>"
  }
}
```

### Step 5 — Output

End your reply with a single line:

```
CANDIDATE: <name>
```

## Example failure patterns you might target

(For inspiration — verify the mechanism is actually present before targeting it.)

- **Wrong tool arguments**: the agent calls a tool with a malformed argument →
  improve the prompt, or validate arguments before the call.
- **Missing required action**: `action_checks` shows an expected call never
  happened → prompt the agent to be more thorough, or track required actions.
- **Acted without confirmation**: the agent modified an order without confirming
  → strengthen the confirmation step.
- **Conversation stalled / hit max steps**: → tighten the agent's prompt or add
  a progress check.

## What this skill does NOT do

- Run the tau2 simulator (the outer loop does).
- Modify the eval, the model, `tau2-bench-src/`, or prior candidates.
- Make claims about the held-out test split.
