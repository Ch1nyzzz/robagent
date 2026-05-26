---
name: component-harness-toolathlon
description: Run ONE iteration of Toolathlon (hkust-nlp/Toolathlon) harness evolution by proposing ONE workflow graph patch — a single Python file declaring `COMPONENT: Component` plus a patch op (add_node / replace_node / disable_node) against the frontier workflow at `meta_harness/workflows/toolathlon_main.yaml`. Components subscribe to a named event via `listens=` and react via a Decision; admission is gated by a class×event×decision permission matrix and a Trust block (evidence_anchor + out_of_evidence_probe). Targets the OpenAI Agents SDK + MCP gateway loop with mandatory FunctionTool wrapping; some lifecycle events live inside the SDK Runner and are not yet observable.
---

# component-harness-toolathlon

Run ONE iteration of agent evolution against Toolathlon's task corpus by proposing ONE **workflow graph patch** — `add_node` / `replace_node` / `disable_node` applied to `meta_harness/workflows/toolathlon_main.yaml`. The node added or replaced is one **component**: a single Python file under `agent_toolathlon/components/<name>.py` exporting `COMPONENT: Component`. **You do NOT run benchmarks.** You analyse prior baselines + traces, write one component, choose one patch op, write `pending_eval.json`, and exit. The outer loop applies the patch, scores it on the train subset (28 task ids), and admits or rejects.

## What Toolathlon tasks look like

A task = one directory under `Toolathlon-src/tasks/finalpool/<task_id>/`. The agent receives:
  * a natural-language user instruction (the `task_str`)
  * a set of MCP servers exposing tools (filesystem, GitHub, Notion, GCP, arxiv-local, scholarly, browser, etc.)
  * a Python-based per-task verifier that runs after the agent finishes

The agent runs an OpenAI Agents SDK loop with MCP tools and (for multi-turn tasks) a user simulator. **The verifier is binary** (pass / fail). Failure mechanisms vary widely:

  * **wrong final emission** — passable answer in wrong format (Markdown when plain text asked).
  * **missing required tool call** — answered from prior knowledge without invoking the prescribed lookup.
  * **tool-arg hallucination** — `paper_id="2505.20286v1"` (with version) when the corpus uses versionless ids, or invented filesystem paths outside `/workspace/dumps/workspace`.
  * **prompt-instruction omission** — ignored part of a multi-part request.
  * **scoping mistake** — touched the wrong Notion page / GitHub repo because the verifier's allowlist wasn't read.
  * **runaway tool loop** — retried the same failing tool dozens of times.

## First principles

1. **The LLM is the last resort.** Move system-prompt scaffolding, output format constraints, tool-arg shape validation, retry-on-loop heuristics into Python.
2. **Decompose; don't defer.** Split a "reasoning-bound" failure into the minimal LLM judgment + the deterministic computation around it.
3. **Code earns its place by capturing stable structure, not by fitting recent failures.** Anchor on a tool's declared JSON Schema, an SDK API field, an MCP server's documented behavior, the user-instruction grammar ("return X, Y, Z in this exact format"). IF/THEN induced from N failed train rows is a memorised map; it overfits.
4. **Prefer events that work in single-turn mode.** ~90% of Toolathlon tasks run with `single_turn_mode=True`; the user simulator never re-prompts. Setup events (`pre_context_build` / `session_start` / `user_prompt_submit`) and per-tool events (`pre_tool_use` / `post_tool_use`) reach the LLM in every task. Post-Runner events (`post_llm_response_raw`, `on_explicit_terminate`, `session_end`) fire after the SDK Runner returns.

## SDK-internal event gap (read this carefully)

Toolathlon mandates v2 tool wrapping: every MCP tool is wrapped as an SDK FunctionTool BEFORE the Agent sees it, so the dispatcher can intercept BEFORE and AFTER the real tool invocation with the actual arguments and the actual result string.

Some events live INSIDE the SDK Runner's `Runner.run(...)` call and are not surfaced to the dispatcher. These are **declared in policy.py** (so a forward-compatible YAML can mention them) but **NOT emitted in v1**:

  * `pre_llm_request` — mid-turn pre-inference hook
  * `pre_tool_arg_validation` — mid-turn arg-validation phase
  * `post_tool_result_raw` — mid-turn raw-result anchor
  * `on_tool_error` — mid-turn tool-error event
  * `on_no_tool_call_emitted` — mid-turn no-tool-call event

Subscribers to these events load but never fire. Use the post-Runner subset for now (`post_llm_response_raw` + `on_length_truncation` + `on_empty_response`); these fire AFTER `Runner.run` returns and see `result.raw_responses`.

## The component model

```python
@dataclass(frozen=True, kw_only=True)
class Component:
    name: str
    cls: ComponentClass
    listens: str                           # event name the dispatcher routes on
    matcher: Optional[Callable[[Ctx], bool]]
    handler: Callable[[Ctx], Decision]
    trust: Trust
    priority: int = 100
    emits: tuple[str, ...] = ()
```

### Events the toolathlon runtime emits

Per-task setup (in `setup_agent`, before SDK Agent is built):

| event                   | when it fires                                                          | typical use                                              |
|-------------------------|------------------------------------------------------------------------|----------------------------------------------------------|
| `task_received`         | top of `run_interaction_loop`                                          | lifecycle anchor                                         |
| `pre_context_build`     | in `setup_agent`, paired with the instructions-build phase             | inject system-prompt scaffolding                         |
| `session_start`         | in `setup_agent`, after pre_context_build                              | inject framework constants                               |
| `pre_agent_construct`   | last hook in `setup_agent` before `Agent(...)` is built                | inference-hint injection                                 |
| `user_prompt_submit`    | each outer user turn, after user_query obtained                         | append text to the user message before it lands in logs  |

Per MCP tool call (via FunctionTool wrapper; sees REAL args + result):

| event                       | when it fires                                                          | typical use                                                       |
|-----------------------------|------------------------------------------------------------------------|-------------------------------------------------------------------|
| `pre_tool_use`              | inside FunctionTool wrapper, before MCP `call_tool`                     | ALLOW / REWRITE_TOOL_ARGS / true BLOCK (composes across components) |
| `post_tool_use`             | inside FunctionTool wrapper, after MCP `call_tool`                      | INJECT_CONTEXT is **concatenated INTO the tool result string** so the LLM sees it on its next inference |

Post-Runner (after `Runner.run` returns):

| event                       | when it fires                                                          | typical use                                                       |
|-----------------------------|------------------------------------------------------------------------|-------------------------------------------------------------------|
| `post_llm_response_raw`     | post-`Runner.run`, parses `result.raw_responses`                       | observe / inject for next turn                                    |
| `on_length_truncation`      | **synthesised** when `result.raw_responses[-1].finish_reason == "length"` | flag the limit                                                    |
| `on_empty_response`         | **synthesised** when `result.final_output.strip() == ""`               | retry hint                                                        |
| `on_explicit_terminate`     | after `termination_checker(...)` returns True                          | **artifact gate**: BLOCK refuses termination and the loop continues |
| `session_end`               | top of `save_results`                                                  | bookkeeping                                                       |

### Component classes

| class                  | what the matcher tests                                                                                                              | risk                          |
|------------------------|-------------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| `mechanism_layer`      | a system field, an MCP tool-declared JSON Schema, an SDK lifecycle property, a general algorithm (URL parse, version stripping)      | LOW                           |
| `reactive_guard`       | an observed failure event (tool error string contains "denied", missing required substring in final response, repeated identical call) | LOW                           |
| `channel`              | task structure — instruction text grammar ("return X, Y, Z"), allowed-MCP-list, presence/absence of a server in `task_config`         | LOW                           |
| `induced_rule`         | a reading of policy/instruction text — IF/THEN compiled from N evidence rows                                                          | HIGH (admitted ADVISORY-ONLY) |
| `predictive_heuristic` | raw prompt / response text guessing what the model is about to do                                                                     | REJECTED at load time         |

### Decision

| decision           | semantics                                                                                                                                                                                                |
|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `allow`            | no-op                                                                                                                                                                                                    |
| `block`            | `pre_tool_use`: the MCP `call_tool` is SKIPPED; the tool result handed to the LLM is a `<component_block …>` string. `on_explicit_terminate`: refuses termination (loop continues). Other events: terminate task. |
| `inject_context`   | setup events → spliced into `Agent.instructions`. `user_prompt_submit` → appended to user_query before it lands in logs. `post_tool_use` → CONCATENATED into the tool result string (LLM sees next inference). |
| `rewrite_tool_args`| `pre_tool_use`: the dict you return REPLACES the LLM-emitted args before the MCP call. Multiple components compose left→right.                                                                            |
| `defer`            | REJECTED at load time (replay queue is v2.5).                                                                                                                                                            |

`ctx.tool_call` at `pre_tool_use` is a **dict** `{"name": str, "arguments": dict}` (not a tau2-style ToolCall object). Components that REWRITE_TOOL_ARGS return `Decision.rewrite_tool_args(new_args_dict)`; the apply layer rebuilds the dict.

### Trust

```python
@dataclass(frozen=True)
class Trust:
    evidence_anchor: str        # REQUIRED — name the stable structure outside evidence
    blast_radius: str           # REQUIRED: local | workflow | global
    rollback_when: str          # REQUIRED — concrete disable signal
    out_of_evidence_probe: str  # REQUIRED for INDUCED_RULE
    fallback: str               # OPTIONAL
```

### Handler helpers on `ctx`

| helper            | what it does                                                          |
|-------------------|-----------------------------------------------------------------------|
| `ctx.chat(...)`   | sub-LLM call via the locked SUT model (mutable inference params)      |
| `ctx.read_file`   | read a workspace file                                                 |
| `ctx.fetch`       | outbound HTTP GET                                                     |
| `ctx.shared`      | per-task dict, free to read/write                                     |
| `ctx.emit(...)`   | fire a custom Tier-2/3 event (re-enters dispatcher; depth cap = 10)   |
| `ctx.emit_upstream(key, value)` | write to `ctx.upstream` for downstream subscribers      |

`ctx.chat(messages, max_tokens=..., temperature=..., system_override=..., tools=...)` routes through `agent.llm.chat` (the SAME locked SUT model name the SDK Runner uses, NOT the OpenAI Agents SDK ModelProvider). No `model=` kwarg.

## The class × event × decision matrix

Load-time gate at `agent_toolathlon/component_runtime/policy.py::ALLOWED`.

Setup events (`task_received`, `pre_context_build`, `session_start`, `pre_agent_construct`, `user_prompt_submit`):

| class                  | admissible decisions                            |
|------------------------|-------------------------------------------------|
| `mechanism_layer`      | inject_context                                  |
| `channel`              | inject_context                                  |
| `induced_rule`         | inject_context (`pre_context_build` / `user_prompt_submit` only — ADVISORY) |
| `reactive_guard`       | inject_context (`user_prompt_submit` only)      |

Per-tool events (`pre_tool_use`, `post_tool_use`):

| class                  | admissible decisions                                                                |
|------------------------|-------------------------------------------------------------------------------------|
| `mechanism_layer`      | `pre_tool_use`: allow, block, rewrite_tool_args. `post_tool_use`: allow, inject_context. |
| `reactive_guard`       | same                                                                                |

Post-Runner events (`post_llm_response_raw`, `on_length_truncation`, `on_empty_response`):

| class                  | admissible decisions                            |
|------------------------|-------------------------------------------------|
| `mechanism_layer`      | inject_context (+ block on synthesised failure events) |
| `reactive_guard`       | same                                            |

Termination events (`on_explicit_terminate`, `stop`):

| class                  | admissible decisions                            |
|------------------------|-------------------------------------------------|
| `mechanism_layer`      | block                                            |
| `reactive_guard`       | block                                            |

`predictive_heuristic` is rejected at every event.

## The workflow graph

Single frontier YAML: `meta_harness/workflows/toolathlon_main.yaml`.

```yaml
nodes:
  - some_component_name
disabled: []
```

Plus a JSON snapshot `meta_harness/logs_components_toolathlon/frontier_workflow.json` written on accept.

### Patch ops

| op             | meaning                                                                                                   |
|----------------|-----------------------------------------------------------------------------------------------------------|
| `add_node`     | append a new node; `agent_toolathlon/components/<id>.py` must be newly written                            |
| `replace_node` | keep the existing node id; overwrite the file (same `COMPONENT.name`)                                     |
| `disable_node` | add the id to `disabled:`; file remains for the durability audit                                          |

## Hard rules

- Exactly ONE patch per invocation.
- **You do NOT run benchmarks.** No `toolathlon_runner.py`. The outer loop scores.
- **No task-specific code.** No train task_ids in matchers. No hardcoded entity strings (paper IDs, GitHub repo names, Notion page slugs). No encoded gold answers.
- **The target inference model (deepseek-v4-pro via Together AI) is LOCKED.** Components do NOT call any other LLM API. `ctx.chat()` IS permitted (locked SUT model name via `agent.llm.chat`, NOT the SDK ModelProvider).
- For `replace_node`, the **first shell action** MUST be:
  ```bash
  cp agent_toolathlon/components/<existing>.py agent_toolathlon/components/<existing>.py.bak_iter<N>
  ```
- READ-ONLY: `Toolathlon-src/`, `bench/toolathlon/`, `toolathlon_runner.py`, `agent_toolathlon/v0/`, `agent_toolathlon/component_runtime/`. Also: `meta_harness/meta_harness_components_toolathlon.py`, `meta_harness/toolathlon_*.txt`.
- Component file naming: `agent_toolathlon/components/component_iter<N>_<slug>.py`. The `COMPONENT.name` should follow the same pattern.
- Components that need DEFER or the 5 SDK-internal events (`pre_llm_request` / `pre_tool_arg_validation` / `post_tool_result_raw` / `on_tool_error` / `on_no_tool_call_emitted`) load but never fire — pick a different design.

## Component file template

```python
# agent_toolathlon/components/component_iter<N>_<slug>.py
from __future__ import annotations

from agent_toolathlon.component_runtime.types import (
    Component, ComponentClass, ComponentContext,
    Decision, Trust,
)


def _matches(ctx: ComponentContext) -> bool:
    # Read ctx.tool_name (pre/post_tool_use), ctx.tool_args (REAL args via wrapper),
    # ctx.incoming_message (post_tool_use: dict with tool_name + args + output),
    # ctx.history (snapshot of self.logs), ctx.tool_names (full tool list),
    # ctx.shared, ctx.proposed_system_prompt (during setup events).
    # ctx.event names the firing event for branchable handlers.
    # NEVER read or compare against task_id-specific data — that is memorisation.
    ...


def _handler(ctx: ComponentContext) -> Decision:
    return Decision.inject_context("...")   # or allow / block / rewrite_tool_args


COMPONENT = Component(
    name="component_iter<N>_<slug>",
    cls=ComponentClass.MECHANISM_LAYER,
    listens="pre_context_build",
    matcher=_matches,                       # or None for always-on
    handler=_handler,
    priority=100,
    emits=(),
    trust=Trust(
        evidence_anchor="...",
        blast_radius="local",
        rollback_when="...",
        out_of_evidence_probe="",   # required if cls=induced_rule
    ),
)
```

## `pending_eval.json` schema

```json
{
  "candidate": {
    "name": "candidate_toolathlon_iter<N>_<slug>",
    "hypothesis": "one-sentence claim of the failure mode",
    "changes": "one-sentence description of what your component does",
    "component": {
      "name": "<COMPONENT.name>",
      "cls": "mechanism_layer",
      "listens": "pre_context_build",
      "file": "agent_toolathlon/components/component_iter<N>_<slug>.py",
      "trust": {
        "evidence_anchor": "...",
        "blast_radius": "local",
        "rollback_when": "...",
        "out_of_evidence_probe": "",
        "fallback": ""
      }
    },
    "workflow_patch": {
      "op": "add_node",
      "name": "<COMPONENT.name>",
      "file": "agent_toolathlon/components/component_iter<N>_<slug>.py"
    }
  }
}
```

For `disable_node`, omit `file` and the only `name` is the existing component's name. For `replace_node`, `name` reuses the existing `COMPONENT.name`.

## How to investigate

State files and trace locations:

```
meta_harness/workflows/toolathlon_main.yaml                            active workflow graph
meta_harness/logs_components_toolathlon/frontier_workflow.json         frontier snapshot
meta_harness/logs_components_toolathlon/frontier_val.json              per-task best
meta_harness/logs_components_toolathlon/evolution_summary.jsonl        one row per iter (incl. rejected)
.component-state-toolathlon/toolathlon_iter<K>/fired.jsonl             which components fired in iter K
```

Per-task trace directories (preserved across iters):

```
Toolathlon-runs/v0/finalpool/<tid>/                          v0 baseline / iter 0
Toolathlon-runs/cr/iter<K>/finalpool/<tid>/                  iter K's candidate run
Toolathlon-runs/cr/final_test/finalpool/<tid>/               held-out test (only after evolution finishes)
```

Each `<tid>/` directory holds `traj_log.json`, `eval_res.json`, `host_loop.log`. The `evolution_summary.jsonl` row for iter K already has `per_task[*].dump_dir` filled with the correct path.

1. **Read `frontier_val.json`'s `per_task` map.** Each entry has `passed`, `tier`, `dump_dir`, `error`. This is the CURRENT frontier — start here.
2. **If `evolution_summary.jsonl` has any row with iter ≥ 1, also read those rows.** Each names a `candidate.hypothesis` + `train_score` + `accepted` + `per_task`. Compare a prior row's per_task against the v0 / frontier per_task to see which tasks that candidate broke (regression = was passing before, failing in iter K). Avoid re-proposing a mechanism a prior `hypothesis` already covered.
3. **For each failing task_id (`passed=false`)**, read the three trace files at `<dump_dir>/`:
   * `traj_log.json` — full message log, `config.single_turn_mode`, `key_stats`, `status` (success / failed / max_turns_reached / interrupted).
   * `eval_res.json` — `{pass: bool, details: str}` from the per-task verifier (often the most informative single file).
   * `host_loop.log` — pretty-printed TOOL_CALL / TOOL_OUT / SUMMARY events in chronological order.
4. **Look for ≥3 failures sharing the same mechanism.** Examples:
   - 3 failures end with the assistant returning a Markdown answer when the instruction asked for "plain text without markdown" → `pre_context_build` INJECT a strict-format reminder.
   - 3 failures call `gw-arxiv_local-read_paper` with `paper_id="2505.20286v1"` (with version) when the corpus only has versionless ids → `pre_tool_use` REWRITE_TOOL_ARGS that strips the version suffix (anchor: arxiv id grammar).
   - 3 failures retry the same failing tool >5 times → `session_start` INJECT a "if a tool fails the same way 3 times, switch strategy" instruction.
5. **Form ONE hypothesis** tied to a stable structure (an MCP tool's schema; the user instruction's grammar; an SDK status field; an OpenAI API field). State the structure in `trust.evidence_anchor`.
6. **Write ONE component** with the smallest possible matcher/handler. Resist embedding task-specific entity names.
7. **Validate**:
   ```bash
   python -c "
   import sys; sys.path.insert(0, '.')
   from agent_toolathlon.component_runtime.registry import load_components_from_dir
   comps = load_components_from_dir('agent_toolathlon/components',
                                     only=['<COMPONENT.name>'])
   assert any(c.name == '<COMPONENT.name>' for c in comps)
   print('ok')
   "
   ```
8. **Write `pending_eval.json`** and print `CANDIDATE: <name>`.

## Common patterns

| pattern                                              | class             | listens                       | decision                                                  |
|------------------------------------------------------|-------------------|-------------------------------|-----------------------------------------------------------|
| strict-format reminder in instructions               | mechanism_layer   | `pre_context_build`           | inject_context                                            |
| inject "always strip arxiv version suffix"           | mechanism_layer   | `session_start`               | inject_context                                            |
| advisory: policy paragraph                           | induced_rule      | `pre_context_build`           | inject_context                                            |
| reactive note appended to user query                 | reactive_guard    | `user_prompt_submit`          | inject_context                                            |
| strip arxiv version suffix from tool args            | mechanism_layer   | `pre_tool_use`                | rewrite_tool_args                                         |
| block tool call outside workspace dir                | reactive_guard    | `pre_tool_use`                | block                                                     |
| reformat tool output before LLM sees it              | mechanism_layer   | `post_tool_use`               | inject_context (concatenated into tool result string)     |
| artifact gate before termination                     | reactive_guard    | `on_explicit_terminate`       | block (refuses termination; loop continues)               |
| length-recovery hint                                 | reactive_guard    | `on_length_truncation`        | inject_context                                            |
| publish task progress for downstream observer        | mechanism_layer   | A `listens="post_llm_response_raw"`, emits `iter<N>_<slug>_progress` → B `listens="iter<N>_<slug>_progress"` | allow / inject_context |

## Custom events (Tier 2/3)

```
iter<N>_<slug>_<event>        e.g. iter9_workspace_artifact_present
on_<thing>                    failure-mode style
```

Always declare `emits=(...)` on the publisher. Subscribers declare `listens="iter<N>_<slug>_<event>"`. To admit non-`allow` decisions on a custom event, add a string-key entry to `ALLOWED[ComponentClass.X]` in `agent_toolathlon/component_runtime/policy.py`. Grep `agent_toolathlon/components/*.py` for `emits=` to find taken names.

## What NOT to write

- Components that memorise train-set answers (encode "if task instruction contains 'Alita', return paper_id=2505.20286").
- Components that try to compute the final answer deterministically without calling the LLM (defeats the SUT measurement).
- Components that touch `toolathlon_runner.py`, `Toolathlon-src/`, or `agent_toolathlon/component_runtime/`.
- Components that need DEFER, or the 5 SDK-internal events (`pre_llm_request` / `pre_tool_arg_validation` / `post_tool_result_raw` / `on_tool_error` / `on_no_tool_call_emitted`) — declared but not emitted in v1.
- Components that fire on EVERY task (`matcher=None`, priority=0) — that's effectively a prompt rewrite, not a component.
