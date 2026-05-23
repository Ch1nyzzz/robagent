---
name: hook-harness-tau2
description: Run ONE iteration of tau2-bench harness evolution by proposing ONE hook — a single Python file that attaches to a defined lifecycle event of the agent (session_start / pre_tool_use / post_tool_use). The hook is classified on the durability axis and the runtime enforces what each class is allowed to do. Sibling of robust-harness-tau2 (which produces full LLMAgent subclasses); this skill produces composable, event-scoped hooks instead.
---

# hook-harness-tau2

Run ONE iteration of agent evolution against tau2-bench by proposing ONE
**hook** — a single Python file in `agent_tau2/hooks/<name>.py` exporting
`HOOK: Hook`. **You do NOT run the simulator.** You understand the domain,
analyse prior simulations, propose ONE hook, classify it on the durability
axis, implement it as a hook, write `pending_eval.json` (a hook manifest),
and exit. The outer loop (`meta_harness/meta_harness_hooks.py`) composes
the frontier hook set with your candidate and scores it on the train-30
subset.

## Why hooks (vs. the robust-harness-tau2 candidate model)

`robust-harness-tau2` builds candidates as **whole `LLMAgent` subclasses**
under `agent_tau2/mh_tau2_iter*_*/agent.py`. Each candidate carries one
deterministic intervention, but architecturally consumes the full agent —
**so you cannot stack two interventions in one run**. The frontier is a
single agent, not a set of mechanisms.

`hook-harness-tau2` builds candidates as **single hooks** attached to a
defined lifecycle event. The base agent is always `LLMAgent` (verbatim),
and the frontier is a **set of hooks** that all run together. Adding,
modifying, and removing a hook are first-class operations of one iteration.

Both skills share the same first principles. The hook system also
narrows the durability axis to the three low-risk classes — the two
HIGH-risk classes (`induced_rule`, `predictive_heuristic`) are deleted,
because the structural enforcement of the hook runtime would not save a
mis-induced rule from over-firing on out-of-evidence policy edges.

## First principles (identical to robust-harness-tau2)

1. **The LLM is the last resort.** Each iteration, find one place the LLM is
   doing work deterministic code could do — parsing, arithmetic, lookup,
   comparison, ranking, sequencing, state tracking — and move it into Python.
2. **Decompose; don't defer.** A failure that looks "reasoning-bound" is
   rarely atomic. Split it: what is the *minimal* judgment the LLM must
   make, and what computation / lookup / validation / sequencing around it
   is fully deterministic? Build that deterministic part as a hook.
3. **Code earns its place by capturing stable structure, not by fitting
   recent failures.** Anything you encode must point at a fact that lives
   OUTSIDE your training evidence — a system field, a tool's declared
   schema, a protocol invariant, a general algorithm. A hook INDUCED from
   the N failed simulations you read is a memorised map of those N
   simulations, not a piece of harness.
4. **Honest over fabricated.** When the agent cannot satisfy a request
   within policy, it says so and follows the escalation path — it does not
   invent an action or a confirmation.

## The hook lifecycle

A hook attaches to exactly one event:

| event                 | when it fires                                             | typical decisions |
|-----------------------|-----------------------------------------------------------|-------------------|
| `session_start`       | once, before any LLM call (extends `system_prompt`)       | `inject_context` |
| `user_prompt_submit`  | incoming `UserMessage`, before the LLM sees it            | `inject_context` |
| `pre_tool_use`        | per `ToolCall` the LLM emitted, before it leaves the agent | `rewrite_tool_args`, `block`, `defer`, `allow` |
| `post_tool_use`       | each incoming `ToolMessage` from the env                  | `inject_context` |
| `stop`                | the agent attempts to end the turn                        | `block` (gate) |
| `session_end`         | once, after the final message                             | bookkeeping only |

(v1 of the runtime dispatches `session_start`, `pre_tool_use`, and
`post_tool_use`. `user_prompt_submit`, `stop`, `session_end` are declared
and validated but not yet dispatched — your hook will load and the runtime
will reject any decision its class is not permitted to make.)

## The durability axis — and how the runtime enforces it

Your hook IS one component. Its **class** is *derived* from what its
matcher tests — you do not assert it, the predicate decides. There are
three admissible classes (the two HIGH-risk classes from
robust-harness-tau2, `induced_rule` and `predictive_heuristic`, are
**deleted** from the hook system — see "Why only three classes" below).
The runtime permits each class only the event / decision combinations
that match its risk profile (see
`agent_tau2/hook_runtime/policy.py::ALLOWED`):

| class                  | the matcher tests...                                                                                                   | permitted events                            | permitted decisions                           |
|------------------------|------------------------------------------------------------------------------------------------------------------------|---------------------------------------------|-----------------------------------------------|
| `channel`              | task structure — a request needing content the agent cannot otherwise reach                                            | session_start, user_prompt_submit           | inject_context                                |
| `reactive_guard`       | an observed failure event — `tool.failed`, malformed call, empty / looping turn                                         | pre_tool_use, post_tool_use, user_prompt_submit, stop | block, rewrite_tool_args, inject_context  |
| `deterministic_glue`   | nothing — always-on **mechanical** transform / validation on data in hand; behaviour fully determined by a system field, tool-declared schema, protocol invariant, or general algorithm | pre_tool_use, post_tool_use, session_start | rewrite_tool_args, defer, inject_context |

**The runtime enforces this matrix.** A hook that registers on a class /
event combo not in the table raises `HookPolicyError` at load time; a
hook whose handler returns a decision its class is not allowed to make
raises `HookPolicyError` at fire time. There is no convention to violate
— misclassification cannot reach the simulator.

### Why only three classes (no `induced_rule`, no `predictive_heuristic`)

`induced_rule` and `predictive_heuristic` describe hooks whose behaviour
is INDUCED from finite training evidence — your reading of a policy
document compiled into branches, or a regex / keyword guess about what
the agent *should* do next. Both are MLEs on small N: when a real task
hits a policy edge your N simulations did not cover, the hook fires
incorrectly and overrides what would have been a correct LLM judgement.

The hook system simply does not accept them. If you find yourself
wanting to write such a hook, the runtime rejects it; the signal is to
**redesign so the matcher points at structure that lives outside your
evidence**:

- Can the policy text be retrieved on demand instead of being hard-coded
  into a branch? That is `channel` — inject the document.
- Can the failure event be observed and reacted to? That is
  `reactive_guard` — wait for the tool error or malformed call.
- Is the behaviour actually determined by a tool's declared schema, a
  system field, or a protocol invariant? That is `deterministic_glue` —
  encode the structural fact, not the policy interpretation.

If none of the three rewrites fit, **do not write the hook**. The LLM
is where un-anchored judgement lives; pushing such judgement into the
harness pretends to determinism it does not have.

## Composition model

Each iteration proposes ONE of:

- **`add`** a new hook (new file, new `HOOK.name`).
- **`modify`** an existing hook (new file overwriting the existing one,
  keeping the same `HOOK.name`).
- **`remove`** a hook (delete the file; the manifest names the removed
  hook).

The frontier is the set of hooks that, when all run together, produced the
best train-30 score. The outer loop:

1. Composes `frontier ∪ {your candidate}` (or `frontier \ {removed}` /
   `frontier with name replaced`).
2. Sets `HOOK_NAMES=<comma-separated names>` and `HOOK_RUN_TAG=iter<N>`.
3. Runs `tau2_runner.py --candidate hook_runtime` (the runtime resolves
   the hook set from env vars).
4. Compares to the frontier score. If your candidate is ≥ frontier,
   `pending_eval.json` is admitted into the frontier hook set; otherwise
   the hook file is discarded.

Hook fires land in `.hook-state/iter<N>/fired.jsonl` for durability
auditing (`meta_harness/scripts/durability_audit.py`).

## Hard rules

- Exactly ONE hook per invocation. `add`, `modify`, or `remove` — pick one.
- **You do NOT run the simulator.** No `tau2_runner.py`, no `tau2 run`.
  The outer loop scores.
- **No task-specific code.** No customer names, account / document ids,
  per-task branching, and no encoding of gold answers or gold action sets.
- General documented policy may enter as **advisory context** via a
  CHANNEL hook on `session_start` / `user_prompt_submit` with decision
  `inject_context` — surface the retrieval, do not encode the
  interpretation. Compiling policy into a branch is not admissible
  (the runtime has no class that accepts it).
- READ-ONLY: `tau2-bench-src/`, `tau2_runner.py`, `meta_harness/*`,
  `agent_tau2/v0/`, `agent_tau2/hook_runtime/`, and every earlier
  `agent_tau2/hooks/*.py`. **You may not read anything under
  `agent_tau2/mh_tau2_iter*/` or under `meta_harness/logs_tau2_*/`
  except `logs_tau2_hooks/`** — the proposer harness physically hides
  those paths during your run; if you find a way around the isolation,
  treat it as a bug and ignore the contents.
- Do **not** create a new agent directory. Your only file write is one
  hook file under `agent_tau2/hooks/`.

## Hook interface

```python
# agent_tau2/hooks/<name>.py
from agent_tau2.hook_runtime.types import (
    Decision, Hook, HookClass, HookContext, HookEvent,
)

def _matches(ctx: HookContext) -> bool:
    # Cheap predicate. Reads ctx.tool_name / ctx.tool_args /
    # ctx.incoming_message / ctx.history. NEVER reads task ids.
    ...

def _handler(ctx: HookContext) -> Decision:
    return Decision.rewrite_tool_args({...})    # or Decision.inject_context(...)
                                                # or Decision.allow() / Decision.block(...)

HOOK = Hook(
    name="<stable_hook_id>",                    # reuse to MODIFY
    cls=HookClass.DETERMINISTIC_GLUE,           # see table above
    event=HookEvent.PRE_TOOL_USE,
    matcher=_matches,
    handler=_handler,
    generalization_argument="<see manifest>",
    fallback="<what happens when the hook does not fire>",
    dead_when="<observable condition under which the hook is provably dead weight>",
)
```

Reference example: `agent_tau2/hooks/close_account_strip_optional_reason.py`.

## Workflow

### 1. Understand the workflow (before any failure analysis)

Read the domain's policy document(s), the full tool catalog, and protocol
invariants. Write a workflow summary in your builder log (the end-to-end
happy path, mandatory orderings, prerequisite reads, KB structure,
session-state requirements). Form hypotheses against this reference frame.

### 2. Read state and failed simulations

```
meta_harness/logs_tau2_hooks/frontier_val.json        per-task best across hook sets
meta_harness/logs_tau2_hooks/frontier_hooks.json      the active hook NAMES in the frontier
meta_harness/logs_tau2_hooks/evolution_summary.jsonl  every prior candidate + score + hook manifest
meta_harness/tau2_train_task_ids.txt                  30 tasks — your pool
agent_tau2/hooks/                                     the hook files on disk
```

(Exact paths are in your runtime prompt — use those.)

Pick 4-6 train tasks the frontier still fails (score 0). For each, open
the newest tau2 simulation at
`tau2-runs/meta/hook_runtime__<domain>.json/results.json` and read
`simulations[].messages` / `simulations[].reward_info`. Cross-reference
`.hook-state/iter<N-1>/fired.jsonl` to see which existing hooks fired on
those tasks.

### 3. Form ONE hypothesis

> **HYPOTHESIS**: <falsifiable claim about train-30 reward>
> **MECHANISM**: <failure mode> seen in N≥3 task simulations [tid1, tid2, ...]
> **STABLE STRUCTURE**: <the system field / tool schema / protocol invariant / general algorithm your hook captures — must point OUTSIDE your N evidence simulations>
> **HOOK**: <new | modify <name> | remove <name>>, event=<...>, class=<...>
> **PREDICTION**: train-30 reward <current> → <expected>

If you cannot name a stable structure outside your evidence, the hook
system does not admit your hook (no class fits). Do not write it — see
"Why only three classes" above for the three rewrites to try instead.

### 4. Classify

Determine the class from the matcher's predicate (see the axis table).
The runtime will reject mis-registrations; choose the class whose
permitted events and decisions match what you actually need.

### 5. Implement

Create exactly one file:

```
agent_tau2/hooks/<name>.py
```

following the interface above. Validate the import and registration:

```bash
python -c "
from agent_tau2.hook_runtime.registry import load_hooks_from_dir
hooks = load_hooks_from_dir(only=['<name>'])
ok = any(h.name == '<name>' for hs in hooks.values() for h in hs)
print('hook loads + passes policy:', ok)
"
```

This is the ONLY shell command you run.

### 6. Write `pending_eval.json` (hook manifest)

```json
{
  "iteration": <N>,
  "candidate": {
    "name": "hook_iter<N>_<slug>",
    "hypothesis": "<falsifiable claim>",
    "mechanism": "<failure this targets>",
    "evidence_task_ids": ["<tid1>", "<tid2>", "<tid3>"],
    "changes": "<plain-English diff vs prior frontier>",
    "expected_delta": "<expected reward change on train-30>",
    "hook": {
      "operation": "add | modify | remove",
      "name": "<stable_hook_id — REUSE for modify, names the removed hook for remove>",
      "file": "agent_tau2/hooks/<name>.py",
      "class": "channel | reactive_guard | deterministic_glue",
      "event": "session_start | user_prompt_submit | pre_tool_use | post_tool_use | stop | session_end",
      "activation_predicate": "<the exact condition the matcher branches on>",
      "decision_kinds": ["<which Decision.* the handler can emit>"],
      "generalization_argument": "<REQUIRED. Which stable structure does this hook
        capture? Point at something that lives OUTSIDE your N evidence simulations —
        a system field's semantics, a tool's declared schema, a protocol invariant,
        a general algorithm. If you cannot answer concretely, no hook class fits and
        the runtime will reject the registration — that means do not write this hook;
        see SKILL.md §'Why only three classes' for the three redesigns to try.>",
      "fallback": "<what happens when the matcher returns False or the handler returns allow>",
      "dead_when": "<observable condition under which this hook is provably dead weight>"
    }
  }
}
```

The outer loop reads `name` / `hypothesis` / `changes` as before; the
`hook` block is recorded in `evolution_summary.jsonl` for the durability
audit. `generalization_argument` is load-bearing — if you cannot fill it
in concretely, you have a hook that does not belong in the harness.

### 7. Session log + exit

Write a concise log to
`meta_harness/logs_tau2_hooks/builder_sessions/iter<N>/log.md` — workflow
summary, mechanism, hypothesis, the stable structure your hook captures,
class + permitted decisions + why, files written. Final line of your
reply:

```
CANDIDATE: hook_iter<N>_<slug>
```

## What this skill does NOT do

- Run the tau2 simulator (the outer loop does).
- Modify `tau2-bench-src/`, the eval, the model, `tau2_runner.py`,
  `meta_harness/`, `agent_tau2/v0/`, `agent_tau2/hook_runtime/`, prior
  hook files, or any earlier `agent_tau2/mh_tau2_iter*/`.
- Build a full `LLMAgent` subclass (that's `robust-harness-tau2`).
- Task-specific hardcoding — encoding gold answers or branches keyed to
  specific task ids.
- Loop or propose multiple hooks in one invocation.
- Encode a policy interpretation or a regex / keyword guess as a hook
  branch — those classes are not in the runtime; the registration will
  fail. If you reach for one, redesign per §"Why only three classes".
