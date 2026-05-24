---
name: component-harness-gaia
description: Run ONE iteration of GAIA harness evolution by proposing ONE workflow graph patch — a single Python file declaring `COMPONENT: Component` plus a patch op (add_node / replace_node / disable_node) against the frontier workflow at `meta_harness/workflows/gaia_main.yaml`. The component is typed by mount + class + state_scope, gated by a class×mount×decision permission matrix, and verified by a Trust block (evidence_anchor + out_of_evidence_probe). GAIA sibling of component-harness-tau2; uses GAIA-specific mounts (no tool-use lifecycle).
---

# component-harness-gaia

Run ONE iteration of agent evolution against GAIA by proposing ONE **workflow graph patch**. A patch is one of `add_node` / `replace_node` / `disable_node`, applied to the frontier workflow at `meta_harness/workflows/gaia_main.yaml`. The node added or replaced is one **component** — a single Python file in `agent/components/<name>.py` exporting `COMPONENT: Component`. **You do NOT run benchmarks.** You analyse prior traces, write one component, choose one patch op, write `pending_eval.json`, and exit. The outer loop applies the patch, scores it on train-30, and either admits or rejects.

## Why workflow graph (vs. the prior robust skill)

`robust-harness-gaia` produced one full `agent/mh_iter*_<slug>/base.py` per iteration — a monolithic `run_task` carrying every prior intervention. Two interventions couldn't stack in one iter; the frontier was a single agent, not a set of mechanisms. `RESULTS.md §6.4` argued the right shape for interpretation-bound benchmarks is "give the LLM a sturdier shell rather than compile policy into code" — i.e. composable mechanisms, not monolithic agents.

`component-harness-gaia` builds candidates as **components** attached to defined lifecycle points of the GAIA `run_task` function. The base runner is `agent/component_runtime/base.py` (verbatim, runtime); the frontier is the **set of components** whose names appear in `meta_harness/workflows/gaia_main.yaml`. Adding, modifying, and disabling a component are first-class operations of one iteration.

## First principles

Unchanged across robust/component skills and across GAIA/tau2 — `RESULTS.md §4` validated them with the GAIA +50% relative test gain.

1. **The LLM is the last resort.** Each iteration, find one place the LLM is doing work deterministic code could do — file reading, URL fetching, regex extraction, format normalisation, retry on observed failure — and move it into Python.
2. **Decompose; don't defer.** A failure that looks "reasoning-bound" is rarely atomic. Split: what is the minimal judgment the LLM must make, and what computation / lookup / validation around it is fully deterministic? Build the deterministic part.
3. **Code earns its place by capturing stable structure, not by fitting recent failures.** Anything you encode must point at a fact OUTSIDE your training evidence — a system field (`extras.file_name`), an LLM API field (`finish_reason`), a file format, a general algorithm. A component INDUCED from N failed traces is a memorised map. `RESULTS.md §4.5`: 10 layers at ε=5% compound to ~40% test false-positive.
4. **Honest over fabricated.** When the agent cannot satisfy the task within its reach, it says so (emit `agent.blocked`) and follows the escalation path — never invents an answer.

## The component model

```python
@dataclass(frozen=True)
class Component:
    name: str
    cls: ComponentClass
    mount: Mount
    matcher: Optional[Callable[[Ctx], bool]]
    handler: Callable[[Ctx], Decision]
    trust: Trust
    state_scope: StateScope = StateScope.NONE
    capabilities: tuple[Capability, ...] = (Capability.NONE,)
    priority: int = 100
```

### Mount enum (GAIA-specific)

| mount               | when it fires                                                 | typical use                                              |
|---------------------|---------------------------------------------------------------|----------------------------------------------------------|
| `session_start`     | once before any LLM call (extends `system_prompt`)            | static framework-invariant injection                     |
| `pre_prompt_build`  | per task, before messages list is built                       | file/URL channel (read content, inject into prompt)      |
| `post_llm_response` | after raw LLM content received, before answer extraction      | rewrite (e.g., strip `<thinking>`), recovery, block       |
| `pre_answer_emit`   | after default extraction, before return                       | normalise (number format, strip prefixes)                |
| `session_end`       | bookkeeping only (v1)                                         | —                                                        |

There are NO tool-use mounts. GAIA tasks are single-shot prompt→answer; tool-like behaviour (file read, URL fetch, sub-LLM call) is performed INSIDE a component's handler via `capabilities`.

### ComponentClass enum

| class                 | the matcher tests…                                                                          | risk                          |
|-----------------------|---------------------------------------------------------------------------------------------|-------------------------------|
| `mechanism_layer`     | a system field, an LLM API field, a file format, a general algorithm                        | LOW                           |
| `reactive_guard`      | an observed failure event (`finish_reason=length`, empty content, "I cannot answer")        | LOW                           |
| `channel`             | task structure — `extras.file_name`, a URL named in the prompt, a needed external source    | LOW                           |
| `induced_rule`        | a reading of policy/instruction text — IF/THEN compiled from N evidence traces              | HIGH (admitted ADVISORY-ONLY) |
| `predictive_heuristic`| raw prompt text — regex/keyword guess about what the model is about to do                   | REJECTED at load time         |

### Decision

| decision         | semantics (mount-dependent payload)                                                         |
|------------------|---------------------------------------------------------------------------------------------|
| `allow`          | no-op                                                                                       |
| `block`          | mark task as blocked → returned answer is None with a reason                                |
| `rewrite`        | replace live payload: `prompt` at PRE_PROMPT_BUILD, `raw_response` at POST_LLM_RESPONSE, `answer` at PRE_ANSWER_EMIT (None marks blocked) |
| `inject_context` | SESSION_START / PRE_PROMPT_BUILD → append to system_prompt; POST_LLM_RESPONSE → recovery context (consumed by reactive_guards) |

### StateScope

`none` (default) / `session` (per-task scratchpad at `ctx.state[component_name]`) / `cross_session` (reserved).

### Trust

```python
@dataclass
class Trust:
    evidence_anchor: str        # REQUIRED
    blast_radius: str           # REQUIRED: local | workflow | global
    rollback_when: str          # REQUIRED
    out_of_evidence_probe: str  # REQUIRED for INDUCED_RULE
    fallback: str               # OPTIONAL
```

### Capability

`none` / `read_file` / `http_get` / `llm_call` (sub-LLM recovery) / `mutate_shared`.

## The class × mount × decision matrix

This is the load-time gate at `agent/component_runtime/policy.py::ALLOWED`.

| class \ mount         | session_start | pre_prompt_build              | post_llm_response          | pre_answer_emit  |
|-----------------------|---------------|-------------------------------|----------------------------|------------------|
| `mechanism_layer`     | inject_context| inject_context, rewrite, block| rewrite, block, inject_context | rewrite, block |
| `reactive_guard`      | —             | —                             | rewrite, block, inject_context | rewrite, block |
| `channel`             | inject_context| inject_context                | —                          | —                |
| `induced_rule`        | —             | **inject_context (advisory)** | —                          | —                |
| `predictive_heuristic`| rejected      | rejected                      | rejected                   | rejected         |

A component whose (class, mount) is absent raises `ComponentPolicyError` at load. A handler that emits a non-admitted decision raises at fire time.

## Why these 4 admissible classes

Same rationale as `component-harness-tau2`. `induced_rule` is REINSTATED but advisory-only (PRE_PROMPT_BUILD + INJECT_CONTEXT only). `predictive_heuristic` stays rejected because its matcher tests prompt-surface text — structurally unsafe even as advisory.

GAIA-specific: `reactive_guard` has only POST_LLM_RESPONSE and PRE_ANSWER_EMIT mount points, because GAIA has no tool-use loop to observe failures on. The canonical pattern is `finish_reason=length` recovery: a REACTIVE_GUARD at POST_LLM_RESPONSE matches on `ctx.shared["finish_reason"] == "length"` and either inject_contexts a recovery hint or REWRITEs raw_response after a sub-LLM call.

## The workflow graph

The frontier is at `meta_harness/workflows/gaia_main.yaml`:

```yaml
nodes:
  - file_reader_channel
  - answer_format_extractor
edges: []
disabled: []
```

Plus a JSON snapshot `meta_harness/logs_components_gaia/frontier_workflow.json` written on accept.

### Patch ops

| op             | meaning                                                                                                   |
|----------------|-----------------------------------------------------------------------------------------------------------|
| `add_node`     | append a new node; `agent/components/<id>.py` must be newly written                                       |
| `replace_node` | keep the existing node id; overwrite the file with new behavior (same `COMPONENT.name`)                    |
| `disable_node` | add the id to `disabled:`; file remains for the durability audit                                          |

## Hard rules

- Exactly ONE patch per invocation.
- **You do NOT run benchmarks.** No `run_benchmark.py`, no `tools/eval.py`. The outer loop scores.
- **No task-specific code.** No entity names ("Finding Nemo", "USGS"), no per-task branching, no encoded gold answers.
- **The target inference model is LOCKED.** It is the System Under Test. Do NOT pass a `model=` kwarg to `chat()` and do NOT call any other model API. `agent.llm.chat()` enforces this at call time — passing any model other than `DEFAULT_MODEL` raises `RuntimeError`. (No flash/lite/cheaper-variant fallbacks. No second-opinion calls to a different model. The endpoint and model name are fixed per run via `MODEL_NAME` env; the proposer does not see and does not control them.)
- General documented policy may enter as **advisory context** via `channel` or `induced_rule` (advisory only). Compiling policy text into a mechanical override is not admissible.
- For `replace_node`, the **first shell action** MUST be:
  ```bash
  cp agent/components/<existing>.py agent/components/<existing>.py.bak_iter<N>
  ```
  The outer loop relies on the `.bak` to roll back on reject.
- READ-ONLY: `bench/`, `evals.lock`, `agent/base.py`, all `agent/v*/`, all `agent/mh_iter*/`, `agent/component_runtime/`, all earlier `agent/components/*.py`, `run_benchmark.py`, `agent/llm.py`, `agent/events.py`, all of `meta_harness/` (the outer loop reads your output; you do not modify the loop).
- You may NOT create a new agent directory. Your only file writes are: ONE component file under `agent/components/<name>.py` (+ its `.bak_iter<N>` for replace_node) and the `pending_eval.json` manifest.

## Component file interface

```python
# agent/components/<name>.py
from __future__ import annotations

from agent.component_runtime.types import (
    Capability, Component, ComponentClass, ComponentContext,
    Decision, Mount, StateScope, Trust,
)


def _matches(ctx: ComponentContext) -> bool:
    # Reads ctx.extras / ctx.prompt / ctx.raw_response / ctx.answer /
    # ctx.shared["finish_reason"]. NEVER reads ctx.task_id (treats it as opaque).
    ...


def _handler(ctx: ComponentContext) -> Decision:
    return Decision.inject_context(...)   # or rewrite / block / allow


COMPONENT = Component(
    name="<stable_component_id>",
    cls=ComponentClass.CHANNEL,
    mount=Mount.PRE_PROMPT_BUILD,
    matcher=_matches,
    handler=_handler,
    state_scope=StateScope.NONE,
    capabilities=(Capability.READ_FILE,),
    priority=100,
    trust=Trust(
        evidence_anchor="<system field / LLM API field / file format / general algorithm>",
        blast_radius="local",
        rollback_when="<observable rollback signal>",
        out_of_evidence_probe="<concrete OOE case + handler return>",
        fallback="<matcher-false / handler-allow semantics>",
    ),
)
```

## Workflow

### 1. Read state

```
meta_harness/workflows/gaia_main.yaml                       active graph
meta_harness/logs_components_gaia/frontier_workflow.json    frontier snapshot
meta_harness/logs_components_gaia/frontier_val.json         per-task best
meta_harness/logs_components_gaia/evolution_summary.jsonl   one row per iter (incl. rejected)
meta_harness/train_task_ids.txt                              30 tasks — your pool
agent/components/                                           component files on disk
.component-state/iter<K>/fired.jsonl                        which components fired in iter K (preserved per iter)
```

Per-iter trace + summary locations (preserved across iters; nothing is overwritten):

```
traces/runs/iter<K>/gaia__<task_id>__<run_id>.jsonl                 per-task event log from iter K's eval
traces/iter<K>__gaia_<agent_version>__summary.jsonl                 iter K's summary jsonl (score + answer + trace_path per task)
traces/runs/gaia__<task_id>__<run_id>.jsonl                         legacy v0 baseline / ad-hoc runs (no iter bucket)
traces/gaia__summary.jsonl                                          legacy v0 baseline summary (no iter bucket)
```

Pick 4-6 train tasks the frontier still fails (`score == 0`). Inspect their trace at `traces/runs/iter<K>/`. Cross-reference `.component-state/iter<K>/fired.jsonl` to see which existing components fired in that iter. If `evolution_summary.jsonl` has any row with iter ≥ 1, also read those rows — each names `candidate.hypothesis` + `train_score` + `accepted`, so you can avoid re-proposing a mechanism a prior candidate already covered, and trace regressions back to the iter that introduced them.

### 2. Form ONE hypothesis

```
HYPOTHESIS:           <falsifiable claim about train-30 accuracy>
MECHANISM:            <failure mode> seen in N≥3 task traces [tid1, tid2, ...]
STABLE STRUCTURE:     <evidence_anchor — system field / LLM API field / file format /
                       general algorithm; must live OUTSIDE the N evidence traces>
OUT_OF_EVIDENCE PROBE: <one concrete case NOT in the evidence traces where the matcher
                       would fire, and exactly what the handler returns on it>
PATCH_OP:             <add_node | replace_node | disable_node>
COMPONENT:            mount=<...>, cls=<...>, state_scope=<...>, capabilities=<...>
EXPECTED_DELTA:       train-30 acc <current> → <expected>
```

If no STABLE STRUCTURE outside evidence exists AND OUT_OF_EVIDENCE PROBE can't be answered → no class admits your component (or only INDUCED_RULE with advisory inject_context).

### 3. Classify class + choose mount

| matcher tests…                                              | class             |
|-------------------------------------------------------------|-------------------|
| `extras.file_name`, URL pattern in prompt, source name      | `channel`         |
| `finish_reason=length`, empty content, error string in raw  | `reactive_guard`  |
| file format extension, LLM API field, general algorithm     | `mechanism_layer` |
| policy/instruction reading                                   | `induced_rule` (only at PRE_PROMPT_BUILD + inject_context) |
| raw prompt text via regex                                   | none — redesign or do not write |

### 4. Implement

Create exactly one file at `agent/components/<name>.py`. Validate:

```bash
python -c "
from agent.component_runtime.registry import load_components_from_dir
comps = load_components_from_dir(only=['<name>'])
ok = any(c.name == '<name>' for cs in comps.values() for c in cs)
print('component loads + passes policy:', ok)
"
```

For `replace_node`, FIRST run `cp ... .bak_iter<N>`.

This is the ONLY shell command you run.

### 5. Write `pending_eval.json`

```json
{
  "iteration": <N>,
  "candidate": {
    "name": "component_iter<N>_<slug>",
    "agent_version_arg": "component_runtime",
    "hypothesis": "<falsifiable claim>",
    "mechanism": "<failure mode this targets>",
    "evidence_task_ids": ["<tid1>", "<tid2>", "<tid3>"],
    "changes": "<plain-English diff vs prior frontier>",
    "expected_delta": "<expected acc change on train-30>",
    "component": {
      "id": "<stable_component_id>",
      "cls": "mechanism_layer | reactive_guard | channel | induced_rule",
      "mount": "session_start | pre_prompt_build | post_llm_response | pre_answer_emit | session_end",
      "state_scope": "none | session | cross_session",
      "capabilities": ["none | read_file | http_get | llm_call | mutate_shared"],
      "file": "agent/components/<name>.py",
      "trust": {
        "evidence_anchor": "<stable structure outside evidence traces>",
        "blast_radius": "local | workflow | global",
        "rollback_when": "<observable rollback signal>",
        "out_of_evidence_probe": "<concrete OOE case + handler return>",
        "fallback": "<matcher-false / handler-allow semantics>"
      }
    },
    "workflow_patch": {
      "op": "add_node | replace_node | disable_node",
      "name": "<component name; existing id for disable_node>",
      "file": "agent/components/<name>.py",
      "edges_in": [],
      "edges_out": []
    }
  }
}
```

`agent_version_arg` is always `"component_runtime"` — the fixed graph runtime that resolves the active set from the workflow YAML.

### 6. Session log + exit

Write a concise log to `meta_harness/logs_components_gaia/builder_sessions/iter<N>/log.md`. Final line:

```
CANDIDATE: component_iter<N>_<slug>
```

## Common patterns

| pattern                                          | class             | mount             | decision                  |
|--------------------------------------------------|-------------------|-------------------|---------------------------|
| read `extras.file_name` content into prompt      | channel           | pre_prompt_build  | inject_context            |
| fetch URL named in prompt                        | channel           | pre_prompt_build  | inject_context            |
| `BLOCKED{reason=model_capability_gap}` for vision| channel           | pre_prompt_build  | block                     |
| switch to `FINAL ANSWER:` format prompt          | mechanism_layer   | session_start     | inject_context            |
| recover from `finish_reason=length`              | reactive_guard    | post_llm_response | rewrite (sub-LLM call) / inject_context |
| strip `<thinking>` blocks from raw response      | mechanism_layer   | post_llm_response | rewrite                   |
| extract `FINAL ANSWER:` last line                | mechanism_layer   | pre_answer_emit   | rewrite                   |
| normalise number (strip thousand-separators)     | mechanism_layer   | pre_answer_emit   | rewrite                   |
| detect "I cannot answer" → BLOCKED               | reactive_guard    | pre_answer_emit   | rewrite (to None) / block |
| advisory: "remember to cite the source URL"      | induced_rule      | pre_prompt_build  | inject_context            |

## What this skill does NOT do

- Run benchmarks (the outer loop runs `run_benchmark.py` on train-30).
- Modify the eval rubric, `agent/component_runtime/`, prior `agent/components/*.py` (except via `replace_node`), or any `agent/v*/` / `agent/mh_iter*/`.
- Build a new agent directory (that pattern was retired in favour of components).
- Loop or propose multiple patches in one invocation.
- Encode a policy interpretation as a mechanism_layer override — route to induced_rule + inject_context only.
- Encode a prompt-shape regex as predictive_heuristic — rejected at load.
