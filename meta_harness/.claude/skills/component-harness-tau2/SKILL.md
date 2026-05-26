---
name: component-harness-tau2
description: Run ONE iteration of tau2-bench harness evolution by proposing ONE workflow graph patch — a single Python file declaring `COMPONENT: Component` plus a patch op (add_node / replace_node / disable_node) against the frontier workflow at `meta_harness/workflows/tau2_main.yaml`. Components subscribe to a named event via `listens=` and react via a Decision; admission is gated by a class×event×decision permission matrix and a Trust block (evidence_anchor + out_of_evidence_probe). tau2 multi-turn tool-use lifecycle with `REWRITE_TOOL_ARGS` / `DEFER` decisions on top of `ALLOW` / `BLOCK` / `INJECT_CONTEXT`.
---

# component-harness-tau2

Run ONE iteration of agent evolution against tau2-bench by proposing ONE **workflow graph patch** — `add_node` / `replace_node` / `disable_node` applied to `meta_harness/workflows/tau2_main.yaml`. The node added or replaced is one **component**: a single Python file under `agent_tau2/components/<name>.py` exporting `COMPONENT: Component`. **You do NOT run the simulator.** You read the domain, analyse failed simulations, write one component, choose one patch op, write `pending_eval.json`, and exit.

## First principles

`RESULTS.md §4` validated these. The GAIA +50% relative test gain came from components honoring all four; the tau2-banking regression (test 7/67 vs train 19/30) came from stacks that violated principle (3).

1. **The LLM is the last resort.** Each iteration, find one place the LLM is doing work deterministic code could do — parsing, arithmetic, lookup, comparison, ranking, sequencing, state tracking — and move it into Python.
2. **Decompose; don't defer.** A failure that looks "reasoning-bound" is rarely atomic. Split it: what is the *minimal* judgment the LLM must make, and what computation / lookup / validation / sequencing around it is fully deterministic? Build that deterministic part.
3. **Code earns its place by capturing stable structure, not by fitting recent failures.** Anything you encode must point at a fact that lives OUTSIDE your training evidence — a system field, a tool's declared schema, a protocol invariant, a general algorithm. A component INDUCED from the N failed simulations you read is a memorised map. `RESULTS.md §4.5`: 10 layers at ε=5% compound to ~40% test-time false-positive.
4. **Honest over fabricated.** When the agent cannot satisfy a request within policy, it says so and follows the escalation path — it does not invent an action or a confirmation.

## The component model

```python
@dataclass(frozen=True, kw_only=True)
class Component:
    name: str                              # stable id; reuse for replace_node
    cls: ComponentClass
    listens: str                           # event name the dispatcher routes on
    matcher: Optional[Callable[[Ctx], bool]]
    handler: Callable[[Ctx], Decision]
    trust: Trust                           # required verification block
    state_scope: StateScope = StateScope.NONE
    capabilities: tuple[Capability, ...] = (Capability.NONE,)
    priority: int = 100                    # smaller fires first within an event bucket
    emits: tuple[str, ...] = ()            # self-doc of custom Tier-2/3 events
```

### Events the tau2 runtime emits

Per-task setup (fires once at agent construction):

| event                   | when it fires                                                          | typical use                                              |
|-------------------------|------------------------------------------------------------------------|----------------------------------------------------------|
| `task_received`         | start of `_collect_prompt_injection`                                   | lifecycle anchor                                         |
| `pre_context_build`     | paired with the PRE_CONTEXT_BUILD phase                                | dynamic channel retrieval; static framework injection    |
| `session_start`         | paired with SESSION_START phase                                        | static framework-invariant injection                     |
| `pre_agent_construct`   | last hook before instructions sealed                                   | inference-hint injection                                 |
| `user_prompt_submit`    | incoming `UserMessage`, before the LLM sees it                         | reactive notes / advisory injection                      |

Per LLM turn (`generate_next_message`):

| event                       | when it fires                                                          | typical use                                                       |
|-----------------------------|------------------------------------------------------------------------|-------------------------------------------------------------------|
| `pre_llm_request`           | just before the SUT call                                               | sub-LLM verifier prep                                             |
| `post_llm_response`         | after the assistant turn (sub-LLM verifier slot)                       | rewrite tool_calls / block / inject_context                       |
| `post_llm_response_raw`     | alias of `post_llm_response`                                           | same                                                              |
| `on_length_truncation`      | **synthesised** when `finish_reason == "length"`                       | sub-LLM recovery                                                  |
| `on_empty_response`         | **synthesised** when content + tool_calls are both empty               | reactive retry                                                    |
| `on_no_tool_call_emitted`   | **synthesised** when no tool_calls in assistant turn                   | retry-with-tool reminder                                          |

Per ToolCall (each call in `assistant_message.tool_calls`):

| event                       | when it fires                                                          | typical use                                                       |
|-----------------------------|------------------------------------------------------------------------|-------------------------------------------------------------------|
| `pre_tool_arg_validation`   | narrow schema-check phase before `pre_tool_use`                         | rewrite args via REWRITE_TOOL_ARGS                                |
| `pre_tool_use`              | per ToolCall, before it leaves the agent                                | wrap-tool / arg sanitization / blocking                           |

Per ToolMessage (each tool result):

| event                       | when it fires                                                          | typical use                                                       |
|-----------------------------|------------------------------------------------------------------------|-------------------------------------------------------------------|
| `post_tool_result_raw`      | per ToolMessage in `_fire_post_tool_use`                                | observe                                                           |
| `on_tool_error`             | **synthesised** when `tm.error` is set                                  | retry hint via `inject_context`                                   |
| `post_tool_use`             | per ToolMessage, main post-tool decision                                | reactive guard on observed `tool.failed`                          |

Exit phase: `on_explicit_terminate`, `stop`, `session_end` are declared in policy but not yet dispatched (reserved). A component declaring `listens="stop"` loads but never fires today.

To scope to one tool name use `matcher_for_tool("close_bank_account")` from `agent_tau2.component_runtime.types` instead of writing a free-form matcher.

### Component classes

| class                  | what the matcher tests                                                                              | risk                          |
|------------------------|-----------------------------------------------------------------------------------------------------|-------------------------------|
| `mechanism_layer`      | a system field, a tool-declared schema, a protocol invariant, a general algorithm                   | LOW                           |
| `reactive_guard`       | an observed failure event — `tool.failed`, malformed call, empty/looping turn                        | LOW                           |
| `channel`              | task structure — a request needing content the agent cannot otherwise reach (file, URL, KB doc)      | LOW                           |
| `induced_rule`         | a reading of policy text — IF/THEN compiled from N=3-5 train sim observations                        | HIGH (admitted ADVISORY-ONLY) |
| `predictive_heuristic` | raw prompt text — regex / keyword guess about what the model is about to do                          | REJECTED at load time         |

### Decision

| decision              | semantics                                                                                  |
|-----------------------|--------------------------------------------------------------------------------------------|
| `allow`               | no-op                                                                                      |
| `block`               | drop the tool_call (pre_tool_use / pre_tool_arg_validation) or clear tool_calls (post_llm_response) or terminate (anything else) |
| `rewrite_tool_args`   | replace the tool_call's arguments with the payload dict                                    |
| `defer`               | postpone the call until predicate fires (v1: falls back to `allow` and traced)             |
| `inject_context`      | append text to system_prompt (pre-LLM events) or next-turn SystemMessage (post events)     |

Event → effect of decisions on tau2 objects (handled internally by `_apply_tau2_decision`):

| event family                          | INJECT_CONTEXT lands in                  | REWRITE_TOOL_ARGS effect                  | BLOCK effect                          |
|---------------------------------------|------------------------------------------|-------------------------------------------|---------------------------------------|
| setup (task_received / pre_context_build / session_start / pre_agent_construct / user_prompt_submit / pre_llm_request) | `ctx.proposed_system_prompt` (visible to next subscriber) + `tier1_prompt_inject` for outer code | — | terminate task |
| `pre_tool_use` / `pre_tool_arg_validation` | (rare; goes to next-turn note)        | rewrite the in-flight ToolCall args        | drop this ToolCall                    |
| `post_llm_response` / `post_llm_response_raw` | next-turn SystemMessage              | rewrite first ToolCall args                | clear all tool_calls                  |
| `post_tool_use` / `post_tool_result_raw` / `on_tool_error` | next-turn SystemMessage    | —                                          | terminate task                        |

### StateScope

`none` (default) / `session` (per-task scratchpad at `ctx.state[component_name]`) / `cross_session` (reserved).

### Trust

```python
@dataclass(frozen=True)
class Trust:
    evidence_anchor: str        # REQUIRED: stable structure outside N evidence sims
    blast_radius: str           # REQUIRED: local | workflow | global
    rollback_when: str          # REQUIRED: observable disabling condition
    out_of_evidence_probe: str  # REQUIRED for INDUCED_RULE; optional otherwise
    fallback: str               # OPTIONAL
```

`evidence_anchor` and `out_of_evidence_probe` are **load-bearing** for the durability audit:

- `evidence_anchor`: name a system field, a tool schema field, a protocol invariant, or a general algorithm. If your answer is "doc_015 says…" the structure you're anchored to is inside your evidence — pick a different class or do not write the component.
- `out_of_evidence_probe`: name one concrete case NOT in your evidence sims where your matcher would fire, and state what your handler returns on it. If you cannot construct one, the component overfits by construction.

### Capability

| capability        | what it permits                                                       |
|-------------------|-----------------------------------------------------------------------|
| `none`            | pure function                                                         |
| `read_file`       | `open(path, "r")` on workspace paths                                  |
| `http_get`        | outbound HTTP GET                                                     |
| `llm_call`        | invoke `ctx.chat(...)` (locked SUT model name via `agent.llm.chat`)   |
| `tool_call`       | issue a sub-tool-call within the handler (retriever pattern)          |
| `mutate_shared`   | write to `ctx.shared`                                                 |

`ctx.chat(messages, max_tokens=..., temperature=..., system_override=..., tools=...)` — sub-LLM helper bound to the locked SUT model. `ctx.emit(custom_event_name, **fields)` re-enters the dispatcher (depth cap = 10). `ctx.emit_upstream(key, value)` writes to `ctx.upstream`.

## The class × event × decision matrix

Load-time gate at `agent_tau2/component_runtime/policy.py::ALLOWED`.

| class \ event family    | setup (pre-LLM)               | `pre_tool_use` / `pre_tool_arg_validation` | `post_llm_response[_raw]`               | `post_tool_use` / `on_tool_error` / `post_tool_result_raw` | `on_*` (length/empty/no_tool_call) | `stop` / `on_explicit_terminate` |
|-------------------------|-------------------------------|--------------------------------------------|-----------------------------------------|------------------------------------------------------------|------------------------------------|----------------------------------|
| `mechanism_layer`       | inject_context                | rewrite_tool_args, defer, block            | rewrite_tool_args, block, inject_context | inject_context                                              | block, inject_context              | block                            |
| `reactive_guard`        | inject_context (user_prompt_submit only) | block, rewrite_tool_args        | rewrite_tool_args, block, inject_context | inject_context                                              | block, inject_context              | block                            |
| `channel`               | inject_context                | —                                          | —                                       | —                                                          | —                                  | —                                |
| `induced_rule`          | **inject_context (advisory)** on pre_context_build / user_prompt_submit | — | —                          | —                                                          | —                                  | —                                |
| `predictive_heuristic`  | rejected                      | rejected                                   | rejected                                | rejected                                                   | rejected                           | rejected                         |

`induced_rule` is REINSTATED from the hook system but **advisory-only** — restricted to pre-context events with `inject_context` only. The LLM keeps final authority; the rule cannot mechanically override. `predictive_heuristic` stays rejected because its matcher tests **prompt-shape** (regex on user-message content) — even advisory injection on that activation conditions interpretation on raw text surface.

## The workflow graph

The frontier YAML is at `meta_harness/workflows/tau2_main.yaml`:

```yaml
nodes:
  - close_account_strip_optional_reason
  - discoverable_audit_channel
  - cc_account_workflow_doc_index
edges: []                 # v1: declarative only
disabled: []
```

Plus a JSON snapshot at `meta_harness/logs_tau2_components/frontier_workflow.json` (written ONLY at acceptance; the scoring loop's trust source for the "previous frontier" reference).

### Patch ops

| op             | meaning                                                                                                   |
|----------------|-----------------------------------------------------------------------------------------------------------|
| `add_node`     | append a new node; the file `agent_tau2/components/<id>.py` must be newly written                         |
| `replace_node` | keep the existing node id; overwrite the file with a new implementation (same `COMPONENT.name`)           |
| `disable_node` | add the existing node's id to `disabled:`; the file remains on disk for the durability audit              |

Sugar derivable from these (NOT exposed as separate ops):

- `wrap_tool(tool_name)` = `add_node` with `listens="pre_tool_use"` and `matcher_for_tool(tool_name)`.
- `insert_before(node_id)` = `add_node` with edges_in pointing at the position. (v1 dispatch ignores edges; use `priority` to bias ordering.)

## Composition model

Each iteration proposes exactly one patch. The outer loop:

1. Reads `tau2_main.yaml`.
2. Applies your patch to produce a candidate graph.
3. Writes the new YAML, sets `COMPONENT_NAMES=<active>`, `COMPONENT_RUN_TAG=iter<N>`, runs `tau2_runner.py --candidate component_runtime` on train-30.
4. Compares the candidate's train-30 reward to the frontier reward.
5. If ≥ frontier: patch admitted; `frontier_workflow.json` snapshot written; component file kept; `.bak_iter<N>` deleted for `replace_node`. Otherwise: YAML reverted, new file unlinked for `add_node`, original restored from `.bak_iter<N>` for `replace_node`.

Component fires land in `.component-state/iter<N>/fired.jsonl` for the durability audit.

## Hard rules

- Exactly ONE patch per invocation. One of `add_node`, `replace_node`, `disable_node`.
- **You do NOT run the simulator.** No `tau2_runner.py`, no `tau2 run`. The outer loop scores.
- **No task-specific code.** No customer names, account / document ids, per-task branching, no encoding of gold answers or gold action sets.
- General documented policy may enter as **advisory context** via a `channel` (setup events) or an `induced_rule` (`pre_context_build` / `user_prompt_submit`, `inject_context` only).
- **The target inference model is LOCKED via the tau2 LLM config.** Components may NOT spin up a different model. `ctx.chat()` IS permitted for sub-LLM verifier patterns — it routes through `agent.llm.chat` (locked SUT model name) with mutable inference params. Declare `Capability.LLM_CALL`.
- For `replace_node`, the **first shell action** you take MUST be:
  ```bash
  cp agent_tau2/components/<existing>.py agent_tau2/components/<existing>.py.bak_iter<N>
  ```
  before overwriting the file. The outer loop relies on this `.bak` to roll back on reject.
- READ-ONLY paths: `tau2-bench-src/`, `tau2_runner.py`, `meta_harness/*`, `agent_tau2/v0/`, `agent_tau2/component_runtime/`, all earlier `agent_tau2/components/*.py` (you may only modify by `replace_node` which goes through the .bak protocol), all `meta_harness/logs_tau2_*/` directories EXCEPT `logs_tau2_components/`.
- You may NOT create a new agent directory. Your only file writes are: ONE component file under `agent_tau2/components/<name>.py` (+ its `.bak_iter<N>` for replace_node) and the `pending_eval.json` manifest.

## Component file template

```python
# agent_tau2/components/<name>.py
from __future__ import annotations

from agent_tau2.component_runtime.types import (
    Capability, Component, ComponentClass, ComponentContext,
    Decision, StateScope, Trust,
)


def _matches(ctx: ComponentContext) -> bool:
    # Cheap predicate. Reads ctx.tool_name / ctx.tool_args / ctx.incoming_message
    # / ctx.history / ctx.domain_policy. ctx.event names the firing event.
    # NEVER reads ctx.task_id (the runtime hides it).
    ...


def _handler(ctx: ComponentContext) -> Decision:
    return Decision.rewrite_tool_args({...})  # or inject_context / block / allow / defer


COMPONENT = Component(
    name="<stable_component_id>",
    cls=ComponentClass.MECHANISM_LAYER,
    listens="pre_tool_use",
    matcher=_matches,
    handler=_handler,
    state_scope=StateScope.NONE,
    capabilities=(Capability.NONE,),
    priority=100,
    emits=(),
    trust=Trust(
        evidence_anchor="<system field / tool schema / protocol invariant / general algorithm>",
        blast_radius="local",
        rollback_when="<observable rollback condition>",
        out_of_evidence_probe="<concrete OOE case + handler return>",
        fallback="<matcher-false / handler-allow semantics>",
    ),
)
```

Reference examples (already in the frontier):
- `agent_tau2/components/close_account_strip_optional_reason.py` — `mechanism_layer` / `pre_tool_use` / `rewrite_tool_args`
- `agent_tau2/components/discoverable_audit_channel.py` — `mechanism_layer` / `session_start` / `inject_context`

## Workflow

### 1. Understand the workflow (before any failure analysis)

Read the domain's policy document(s), the full tool catalog, and protocol invariants. Write a workflow summary in your builder log: the end-to-end happy path, mandatory orderings, prerequisite reads, KB structure, session-state requirements. Form hypotheses against this reference frame.

### 2. Read state and failed simulations

```
meta_harness/workflows/tau2_main.yaml                                the active graph
meta_harness/logs_tau2_components/frontier_workflow.json             frontier snapshot
meta_harness/logs_tau2_components/frontier_val.json                  per-task best
meta_harness/logs_tau2_components/evolution_summary.jsonl            one row per iter (incl. rejected)
meta_harness/tau2_train_task_ids.txt                                 30 tasks — your pool
agent_tau2/components/                                                component files on disk
.component-state/iter<K>/fired.jsonl                                  which components fired in iter K
```

Per-iter simulation dumps + summaries (preserved across iters):

```
tau2-bench-src/data/simulations/tau2-runs/meta/v0__<domain>.json/results.json       v0 baseline / iter 0
tau2-bench-src/data/simulations/tau2-runs/meta/iter<K>/component_runtime__<domain>.json/results.json   iter K's candidate run
traces/iter<K>__tau2_component_runtime__summary.jsonl                               iter K's summary jsonl
```

Each `results.json` carries `simulations[].messages` / `simulations[].reward_info` for every task. Pick 4-6 train tasks the frontier still fails (reward 0) and open the appropriate iter's `results.json`. Cross-reference `.component-state/iter<K>/fired.jsonl`.

### 3. Form ONE hypothesis

```
HYPOTHESIS:           <falsifiable claim about train-30 reward>
MECHANISM:            <failure mode> seen in N≥3 task simulations [tid1, tid2, ...]
STABLE STRUCTURE:     <evidence_anchor — system field / tool schema / protocol invariant /
                       general algorithm; must live OUTSIDE the N evidence sims>
OUT_OF_EVIDENCE PROBE: <one concrete case NOT in the evidence sims where the matcher fires,
                       and exactly what the handler returns on it>
PATCH_OP:             <add_node | replace_node | disable_node>
COMPONENT:            listens=<...>, cls=<...>, state_scope=<...>, capabilities=<...>
EXPECTED_DELTA:       train-30 reward <current> → <expected>
```

If you cannot name a STABLE STRUCTURE outside your evidence AND cannot answer OUT_OF_EVIDENCE PROBE concretely, no class admits your component (or `induced_rule` is the only fit — then your decision MUST be `inject_context` on a pre-context event, no override).

### 4. Classify class + choose event

| matcher tests…                                                | class             | suggested events                                                |
|---------------------------------------------------------------|-------------------|----------------------------------------------------------------|
| system field / tool schema field / protocol invariant         | `mechanism_layer` | `pre_tool_use` / `pre_tool_arg_validation`                      |
| observed failure event in `ctx.incoming_message`              | `reactive_guard`  | `post_tool_use` / `on_tool_error`                               |
| task structure (file present, URL mentioned, KB doc needed)   | `channel`         | `pre_context_build` / `session_start` / `user_prompt_submit`    |
| "the policy text says X applies to this case"                 | `induced_rule`    | `pre_context_build` / `user_prompt_submit` (`inject_context`)    |
| raw prompt text via regex/keywords                            | none — redesign or do not write |                                              |

### 5. Implement + validate

Create exactly one file at `agent_tau2/components/<name>.py`. Validate the import:

```bash
python -c "
from agent_tau2.component_runtime.registry import load_components_from_dir
comps = load_components_from_dir(only=['<COMPONENT.name>'])
assert any(c.name == '<COMPONENT.name>' for c in comps)
print('component loads + passes policy:', True)
"
```

For `replace_node`, FIRST run the `cp ... .bak_iter<N>` command (see Hard rules). This is the ONLY shell command you run.

### 6. Choose patch op

| situation                                                                 | op             |
|---------------------------------------------------------------------------|----------------|
| no existing node addresses this mechanism                                 | `add_node`     |
| an existing node addresses this mechanism but has a known bug             | `replace_node` (reuse `COMPONENT.name`; .bak first) |
| an existing node is provably dead weight or actively harmful              | `disable_node` (no new file; reference existing id) |

### 7. Write `pending_eval.json`

```json
{
  "iteration": <N>,
  "candidate": {
    "name": "candidate_iter<N>_<slug>",
    "hypothesis": "<falsifiable claim>",
    "mechanism": "<failure mode this targets>",
    "evidence_task_ids": ["<tid1>", "<tid2>", "<tid3>"],
    "changes": "<plain-English diff vs prior frontier>",
    "expected_delta": "<expected reward change on train-30>",
    "component": {
      "id": "<stable_component_id>",
      "cls": "mechanism_layer | reactive_guard | channel | induced_rule",
      "listens": "<event_name>",
      "state_scope": "none | session | cross_session",
      "capabilities": ["none | read_file | http_get | tool_call | llm_call | mutate_shared"],
      "file": "agent_tau2/components/<name>.py",
      "trust": {
        "evidence_anchor": "<stable structure outside evidence sims>",
        "blast_radius": "local | workflow | global",
        "rollback_when": "<observable rollback condition>",
        "out_of_evidence_probe": "<concrete OOE case + handler return>",
        "fallback": "<matcher-false / handler-allow semantics>"
      }
    },
    "workflow_patch": {
      "op": "add_node | replace_node | disable_node",
      "name": "<component name; existing id for disable_node>",
      "file": "agent_tau2/components/<name>.py",
      "edges_in": [],
      "edges_out": []
    }
  }
}
```

For `disable_node`, the `component` block can be omitted; only `workflow_patch.name` matters.

### 8. Session log + exit

Write a concise log to `meta_harness/logs_tau2_components/builder_sessions/iter<N>/log.md`. Final line of your reply:

```
CANDIDATE: candidate_iter<N>_<slug>
```

## Common patterns

| pattern                                | class             | listens                | decision                  | example                                                |
|----------------------------------------|-------------------|------------------------|---------------------------|--------------------------------------------------------|
| strip an optional arg                  | mechanism_layer   | `pre_tool_use`         | rewrite_tool_args         | `close_account_strip_optional_reason`                  |
| inject a framework constant            | mechanism_layer   | `session_start`        | inject_context            | `discoverable_audit_channel`                           |
| retrieve a KB doc per session          | channel           | `pre_context_build`    | inject_context            | retriever with `capabilities=(TOOL_CALL,)`             |
| wrap one tool's args                   | mechanism_layer   | `pre_tool_use`         | rewrite_tool_args         | matcher uses `matcher_for_tool("close_bank_account")`  |
| react to `tool.failed`                 | reactive_guard    | `on_tool_error`        | inject_context            | append a "retry with X" note for next turn             |
| block a malformed tool call            | reactive_guard    | `pre_tool_use`         | block                     | observed mid-turn shape error                          |
| sub-LLM verifier on assistant turn     | mechanism_layer   | `post_llm_response`    | rewrite_tool_args / block | self-consistency check before dispatch                 |
| schema-validate args narrowly          | mechanism_layer   | `pre_tool_arg_validation` | rewrite_tool_args      | dedicated phase before main pre_tool_use               |
| length-recovery via sub-LLM            | reactive_guard    | `on_length_truncation` | inject_context (or rewrite via ctx.chat) |                                |
| advisory policy note                   | induced_rule      | `pre_context_build`    | inject_context            | "the closure flow has prerequisites; consult doc_021"  |

## Custom events (Tier 2/3)

A component can emit its own event name to coordinate with sibling components in the same task. Convention:

```
iter<N>_<slug>_<event>        e.g. iter12_audit_visibility_resolved
on_<thing>                    cross-iter failure-mode name
```

Always declare `emits=("iter<N>_<slug>_<event>", ...)` on the publisher; subscribers declare `listens="iter<N>_<slug>_<event>"`. To admit non-`allow` / `inject_context` decisions on a custom event, add a string-key entry to `ALLOWED[ComponentClass.X]` in `agent_tau2/component_runtime/policy.py`. Grep `agent_tau2/components/*.py` for `emits=` to find taken names.

## What this skill does NOT do

- Run the tau2 simulator (the outer loop does).
- Modify `tau2-bench-src/`, the eval, the model, `tau2_runner.py`, `meta_harness/`, `agent_tau2/v0/`, `agent_tau2/component_runtime/`, prior component files (except via `replace_node` + `.bak` protocol).
- Build a full `LLMAgent` subclass (that pattern was retired in favour of components).
- Task-specific hardcoding — encoding gold answers or branches keyed to specific task ids.
- Loop or propose multiple patches in one invocation.
- Encode a policy interpretation as a `mechanism_layer` override (`rewrite_tool_args` / `block`). Route to `induced_rule` + `inject_context` only.
- Encode a prompt-shape regex as `predictive_heuristic` — rejected at load.
