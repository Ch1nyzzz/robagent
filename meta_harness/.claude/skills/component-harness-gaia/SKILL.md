---
name: component-harness-gaia
description: Run ONE iteration of GAIA harness evolution by proposing ONE workflow graph patch — a single Python file declaring `COMPONENT: Component` plus a patch op (add_node / replace_node / disable_node) against the frontier workflow at `meta_harness/workflows/gaia_main.yaml`. Components subscribe to a named event via `listens=` and react via a Decision; admission is gated by a class×event×decision permission matrix and a Trust block (evidence_anchor + out_of_evidence_probe). GAIA single-shot lifecycle — no tool-use events.
---

# component-harness-gaia

Run ONE iteration of agent evolution against GAIA by proposing ONE **workflow graph patch** — `add_node` / `replace_node` / `disable_node` applied to the frontier workflow at `meta_harness/workflows/gaia_main.yaml`. The node added or replaced is one **component**: a single Python file under `agent/components/<name>.py` exporting `COMPONENT: Component`. **You do NOT run benchmarks.** You analyse prior traces, write one component, choose one patch op, write `pending_eval.json`, and exit. The outer loop applies the patch, scores it on train-30, and either admits or rejects.

## First principles

`RESULTS.md §4` validated these with the GAIA +50% relative test gain. Do not violate.

1. **The LLM is the last resort.** Each iteration, find one place the LLM is doing work deterministic code could do — file reading, URL fetching, regex extraction, format normalisation, retry on observed failure — and move it into Python.
2. **Decompose; don't defer.** A failure that looks "reasoning-bound" is rarely atomic. Split: what is the minimal judgment the LLM must make, and what computation / lookup / validation around it is fully deterministic? Build the deterministic part.
3. **Code earns its place by capturing stable structure, not by fitting recent failures.** Anything you encode must point at a fact OUTSIDE your training evidence — a system field (`extras.file_name`), an LLM API field (`finish_reason`), a file format, a general algorithm. A component INDUCED from N failed traces is a memorised map. `RESULTS.md §4.5`: 10 layers at ε=5% compound to ~40% test false-positive.
4. **Honest over fabricated.** When the agent cannot satisfy the task within its reach, it says so (emit `agent.blocked`) and follows the escalation path — never invents an answer.

## The component model

```python
@dataclass(frozen=True, kw_only=True)
class Component:
    name: str                              # stable id; reuse for replace_node
    cls: ComponentClass                    # mechanism_layer | reactive_guard | channel | induced_rule
    listens: str                           # event name the dispatcher routes on
    matcher: Optional[Callable[[Ctx], bool]]
    handler: Callable[[Ctx], Decision]
    trust: Trust                           # required verification block
    priority: int = 100                    # smaller fires first within an event bucket
    emits: tuple[str, ...] = ()            # self-doc of custom Tier-2/3 events this raises
```

### Events the GAIA runtime emits

The runtime calls `dispatcher.emit("<event_name>", ctx)` at each lifecycle anchor. Subscribe by setting `listens="<event_name>"`. Components fire in `(priority, insertion)` order within each event.

| event                   | when it fires                                                          | typical use                                              |
|-------------------------|------------------------------------------------------------------------|----------------------------------------------------------|
| `task_received`         | right after `ComponentContext` is built                                | static framework-invariant injection                     |
| `session_start`         | once per task, paired with task_received                               | static framework-invariant injection                     |
| `pre_prompt_build`      | per task, before messages list is built                                | file/URL channel; rewrite prompt                         |
| `pre_context_build`     | alias of `pre_prompt_build` (cross-sibling vocabulary)                 | same as above                                            |
| `pre_agent_construct`   | last hook before messages list is sealed                               | inject inference hint                                    |
| `pre_llm_request`       | just before the SUT `chat()` call                                      | last-pass injection                                      |
| `post_llm_response`     | after raw LLM content received, before answer extraction               | rewrite (strip `<thinking>`), recovery, block            |
| `post_llm_response_raw` | alias of `post_llm_response` (cross-sibling vocabulary)                | same as above                                            |
| `on_length_truncation`  | **synthesised** when `finish_reason == "length"`                       | sub-LLM recovery with bigger budget                      |
| `on_empty_response`     | **synthesised** when raw_response is empty                             | reactive retry                                           |
| `pre_answer_emit`       | after default extraction, before return                                | normalise (number format, strip prefixes)                |
| `session_end`           | bookkeeping at end of task                                             | —                                                        |

GAIA has NO tool-use events — tasks are single-shot prompt→answer. Tool-like behaviour (file read, URL fetch, sub-LLM call) lives inside a component's handler via the helper methods on `ctx`.

### Component classes

| class                  | what the matcher tests                                                                       | risk                          |
|------------------------|----------------------------------------------------------------------------------------------|-------------------------------|
| `mechanism_layer`      | a system field, an LLM API field, a file format, a general algorithm                         | LOW                           |
| `reactive_guard`       | an observed failure event (`finish_reason=length`, empty content, "I cannot answer")          | LOW                           |
| `channel`              | task structure — `extras.file_name`, a URL named in the prompt, a needed external source     | LOW                           |
| `induced_rule`         | a reading of policy/instruction text — IF/THEN compiled from N evidence traces               | HIGH (admitted ADVISORY-ONLY) |
| `predictive_heuristic` | raw prompt text — regex / keyword guess about what the model is about to do                  | REJECTED at load time         |

### Decision

| decision         | semantics                                                                                 |
|------------------|-------------------------------------------------------------------------------------------|
| `allow`          | no-op                                                                                     |
| `block`          | mark task as blocked → returned answer is None with a reason                              |
| `rewrite`        | replace the live payload at this event (see Event → payload below)                        |
| `inject_context` | append text to system_prompt (pre-LLM events) or recovery queue (post-LLM events)         |

Event → REWRITE payload:

| event                              | payload type      | replaces                                    |
|------------------------------------|-------------------|---------------------------------------------|
| `pre_prompt_build` / `pre_context_build` | `str`        | `ctx.prompt` (task prompt)                  |
| `post_llm_response` / `post_llm_response_raw` / `on_length_truncation` / `on_empty_response` | `str` | `ctx.raw_response`                          |
| `pre_answer_emit`                  | `str` or `None`   | `ctx.answer` (None marks blocked)           |

### Trust

```python
@dataclass(frozen=True)
class Trust:
    evidence_anchor: str        # REQUIRED — name the stable structure outside evidence
    blast_radius: str           # REQUIRED — local | workflow | global
    rollback_when: str          # REQUIRED — observable rollback signal
    out_of_evidence_probe: str  # REQUIRED for INDUCED_RULE
    fallback: str               # OPTIONAL — matcher-false / handler-allow semantics
```

`evidence_anchor` + `out_of_evidence_probe` are **load-bearing** for the durability audit. They are not free-text rationale:

- `evidence_anchor`: name a system field, LLM API field, file format, or general algorithm. If your answer is "trace_017 says…" the structure you're anchored to is inside your evidence — pick a different class or do not write the component.
- `out_of_evidence_probe`: name one concrete case NOT in your evidence traces where your matcher fires, and what your handler returns on it. If you cannot construct one, the component overfits by construction.

### Handler helpers on `ctx`

| helper            | what it does                                                             |
|-------------------|--------------------------------------------------------------------------|
| `ctx.chat(...)`   | sub-LLM call via the locked SUT model (mutable inference params)         |
| `ctx.read_file`   | read a workspace file                                                    |
| `ctx.fetch`       | outbound HTTP GET (NOT WIRED in GAIA v1)                                 |
| `ctx.shared`      | per-task dict, free to read/write                                        |
| `ctx.emit(...)`   | fire a custom Tier-2/3 event (re-enters dispatcher; depth cap = 10)      |
| `ctx.emit_upstream(key, value)` | write to `ctx.upstream` for downstream subscribers         |

`ctx.chat(messages, max_tokens=..., temperature=..., system_override=..., tools=...)` routes through `agent.llm.chat` and rejects any model override at call time. Use for sub-LLM recovery (length truncation, empty response).

## The class × event × decision matrix

Load-time gate at `agent/component_runtime/policy.py::ALLOWED`. A component whose (cls, listens) is absent raises `ComponentPolicyError` at load. A handler that emits a non-admitted decision raises at fire time.

| class \ event           | session_start / task_received | pre_prompt_build / pre_context_build | pre_agent_construct / pre_llm_request | post_llm_response[_raw] | on_length_truncation / on_empty_response | pre_answer_emit | session_end |
|-------------------------|-------------------------------|--------------------------------------|---------------------------------------|-------------------------|-----------------------------------------|-----------------|-------------|
| `mechanism_layer`       | inject_context                | inject_context, rewrite, block       | inject_context                        | rewrite, block, inject_context | rewrite, block, inject_context | rewrite, block | allow |
| `reactive_guard`        | —                             | —                                    | —                                     | rewrite, block, inject_context | rewrite, block, inject_context | rewrite, block | —     |
| `channel`               | inject_context                | inject_context                       | inject_context                        | —                       | —                                       | —               | —     |
| `induced_rule`          | —                             | **inject_context (advisory)**        | —                                     | —                       | —                                       | —               | —     |
| `predictive_heuristic`  | rejected                      | rejected                             | rejected                              | rejected                | rejected                                | rejected        | rejected |

## Why these 4 admissible classes

`induced_rule` is REINSTATED from the hook system but **advisory-only**: restricted to pre-context events with `inject_context` only. An induced reading of a policy document, presented as an advisory note, is strictly safer than the LLM rediscovering the passage cold — worst case is a redundant prompt. The LLM keeps final authority.

`predictive_heuristic` stays rejected because its matcher tests **prompt-shape** (regex on user-message text). Even advisory injection on that activation conditions interpretation on raw surface text — structurally unsafe.

## The workflow graph

The frontier YAML lives at `meta_harness/workflows/gaia_main.yaml`:

```yaml
nodes:
  - length_recovery_guard
  - gaia_file_channel
  - cuneiform_numeric_decoder
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
- **The target inference model is LOCKED.** `agent.llm.chat()` rejects any `model=` override (raises `RuntimeError`). `ctx.chat()` does not even accept a `model` kwarg — only `messages`, `max_tokens`, `temperature`, `system_override`, `tools`.
- General documented policy may enter as **advisory context** via `channel` or `induced_rule` (advisory only). Compiling policy text into a mechanical override is not admissible.
- For `replace_node`, the **first shell action** MUST be:
  ```bash
  cp agent/components/<existing>.py agent/components/<existing>.py.bak_iter<N>
  ```
  The outer loop relies on the `.bak` to roll back on reject.
- READ-ONLY: `bench/`, `evals.lock`, `agent/base.py`, all `agent/v*/`, `agent/component_runtime/`, all earlier `agent/components/*.py`, `run_benchmark.py`, `agent/llm.py`, `agent/events.py`, all of `meta_harness/`.
- You may NOT create a new agent directory. Your only file writes are: ONE component file under `agent/components/<name>.py` (+ its `.bak_iter<N>` for replace_node) and the `pending_eval.json` manifest.

## Component file template

```python
# agent/components/component_iter<N>_<slug>.py
from __future__ import annotations

from agent.component_runtime.types import (
    Component, ComponentClass, ComponentContext,
    Decision, Trust,
)


def _matches(ctx: ComponentContext) -> bool:
    # Read ctx.extras / ctx.prompt / ctx.raw_response / ctx.answer /
    # ctx.shared.get("finish_reason"). NEVER read ctx.task_id.
    ...


def _handler(ctx: ComponentContext) -> Decision:
    return Decision.rewrite(...)        # or inject_context / block / allow


COMPONENT = Component(
    name="<stable_component_id>",
    cls=ComponentClass.REACTIVE_GUARD,
    listens="on_length_truncation",
    matcher=_matches,
    handler=_handler,
    priority=100,
    emits=(),                                # declare custom events you raise
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
.component-state/iter<K>/fired.jsonl                        which components fired in iter K
```

Per-iter trace + summary locations (preserved across iters; nothing is overwritten):

```
traces/runs/iter<K>/gaia__<task_id>__<run_id>.jsonl         per-task event log from iter K
traces/iter<K>__gaia_<agent_version>__summary.jsonl         iter K's summary
```

Pick 4-6 train tasks the frontier still fails (`score == 0`). Inspect their trace under `traces/runs/iter<K>/`. Cross-reference `.component-state/iter<K>/fired.jsonl` to see which existing components fired. If `evolution_summary.jsonl` has any row with iter ≥ 1, also read those rows — each names `candidate.hypothesis` + `train_score` + `accepted`.

### 2. Form ONE hypothesis

```
HYPOTHESIS:           <falsifiable claim about train-30 accuracy>
MECHANISM:            <failure mode> seen in N≥3 task traces [tid1, tid2, ...]
STABLE STRUCTURE:     <evidence_anchor — system field / LLM API field / file format /
                       general algorithm; must live OUTSIDE the N evidence traces>
OUT_OF_EVIDENCE PROBE: <one concrete case NOT in the evidence traces where the matcher
                       fires, and exactly what the handler returns on it>
PATCH_OP:             <add_node | replace_node | disable_node>
COMPONENT:            listens=<...>, cls=<...>
EXPECTED_DELTA:       train-30 acc <current> → <expected>
```

If no STABLE STRUCTURE outside evidence exists AND OUT_OF_EVIDENCE PROBE can't be answered → no class admits your component (or only `induced_rule` with advisory `inject_context`).

### 3. Classify class + choose event

| matcher tests…                                              | class            | suggested events                                    |
|-------------------------------------------------------------|------------------|----------------------------------------------------|
| `extras.file_name`, URL pattern in prompt, source name      | `channel`        | `pre_prompt_build` / `pre_context_build` (inject)  |
| `finish_reason=length`                                      | `reactive_guard` | `on_length_truncation` (rewrite / inject)           |
| empty content                                               | `reactive_guard` | `on_empty_response` (rewrite / inject)              |
| error string in raw response                                | `reactive_guard` | `post_llm_response` / `post_llm_response_raw`       |
| file format extension, LLM API field, general algorithm     | `mechanism_layer`| any event the (cls, event) cell admits              |
| policy/instruction reading                                  | `induced_rule`   | `pre_prompt_build` / `pre_context_build` (inject only) |
| raw prompt text via regex                                   | none — redesign or do not write |                                   |

### 4. Implement + validate

Create exactly one file at `agent/components/<name>.py`. Validate:

```bash
python -c "
from agent.component_runtime.registry import load_components_from_dir
comps = load_components_from_dir(only=['<COMPONENT.name>'])
assert any(c.name == '<COMPONENT.name>' for c in comps)
print('component loads + passes policy:', True)
"
```

For `replace_node`, FIRST run the `cp ... .bak_iter<N>` command (see Hard rules) before overwriting. This is the ONLY shell command you run.

### 5. Write `pending_eval.json`

```json
{
  "iteration": <N>,
  "candidate": {
    "name": "candidate_iter<N>_<slug>",
    "agent_version_arg": "component_runtime",
    "hypothesis": "<falsifiable claim>",
    "mechanism": "<failure mode this targets>",
    "evidence_task_ids": ["<tid1>", "<tid2>", "<tid3>"],
    "changes": "<plain-English diff vs prior frontier>",
    "expected_delta": "<expected acc change on train-30>",
    "component": {
      "id": "<stable_component_id>",
      "cls": "mechanism_layer | reactive_guard | channel | induced_rule",
      "listens": "<event_name>",
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
      "file": "agent/components/<name>.py"
    }
  }
}
```

`agent_version_arg` is always `"component_runtime"`. Final line of your reply:

```
CANDIDATE: candidate_iter<N>_<slug>
```

## Common patterns

| pattern                                              | class             | listens                       | decision                                    |
|------------------------------------------------------|-------------------|-------------------------------|---------------------------------------------|
| read `extras.file_name` content into prompt          | channel           | `pre_prompt_build`            | inject_context                              |
| fetch URL named in prompt                            | channel           | `pre_prompt_build`            | inject_context                              |
| `BLOCKED{reason=model_capability_gap}` for vision    | channel           | `pre_prompt_build`            | block                                       |
| switch to `FINAL ANSWER:` format prompt              | mechanism_layer   | `session_start`               | inject_context                              |
| recover from `finish_reason=length`                  | reactive_guard    | `on_length_truncation`        | rewrite (via `ctx.chat(max_tokens=32768)`)  |
| recover from empty response                          | reactive_guard    | `on_empty_response`           | rewrite or inject_context                   |
| strip `<thinking>` blocks from raw response          | mechanism_layer   | `post_llm_response`           | rewrite                                     |
| extract `FINAL ANSWER:` last line                    | mechanism_layer   | `pre_answer_emit`             | rewrite                                     |
| normalise number (strip thousand-separators)         | mechanism_layer   | `pre_answer_emit`             | rewrite                                     |
| detect "I cannot answer" → BLOCKED                   | reactive_guard    | `pre_answer_emit`             | rewrite (to None) / block                   |
| advisory: "remember to cite the source URL"          | induced_rule      | `pre_prompt_build`            | inject_context                              |
| publish / subscribe between two components           | mechanism_layer   | A emits `iter<N>_<slug>_X` → B `listens="iter<N>_<slug>_X"` (reads `ctx.upstream`) | allow / inject_context |

## Custom events (Tier 2/3)

A component may emit its own event name to signal downstream work in the same task. Naming convention:

```
iter<N>_<slug>_<event>        e.g. iter12_length_recovery_recovered
on_<thing>                    cross-iter failure-mode name
```

Always declare `emits=("iter<N>_<slug>_<event>", ...)` on the publisher. Subscribers declare `listens="iter<N>_<slug>_<event>"`. To admit non-`allow`/`inject_context` decisions on a custom event, add a string-key entry to `ALLOWED[ComponentClass.X]` in `agent/component_runtime/policy.py`. To discover taken names, grep `agent/components/*.py` for `emits=`.

## What this skill does NOT do

- Run benchmarks (the outer loop runs `run_benchmark.py` on train-30).
- Modify `agent/component_runtime/`, prior `agent/components/*.py` (except via `replace_node`), or any `agent/v*/`.
- Build a new agent directory (that pattern was retired in favour of components).
- Loop or propose multiple patches in one invocation.
- Encode a policy interpretation as a `mechanism_layer` override (`rewrite` / `block`). Route to `induced_rule` + `inject_context` instead.
- Encode a prompt-shape regex as `predictive_heuristic` — rejected at load.
