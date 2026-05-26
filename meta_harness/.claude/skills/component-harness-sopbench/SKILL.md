---
name: component-harness-sopbench
description: Run ONE iteration of Amazon SOP-Bench harness evolution by proposing ONE workflow graph patch — a single Python file declaring `COMPONENT: Component` plus a patch op (add_node / replace_node / disable_node) against the frontier workflow at `meta_harness/workflows/sopbench_<domain>.yaml`. Components subscribe to a named event via `listens=` and react via a Decision; admission is gated by a class×event×decision permission matrix and a Trust block (evidence_anchor + out_of_evidence_probe). SOP-Bench FC-loop lifecycle — per-turn + per-tool-call events.
---

# component-harness-sopbench

Run ONE iteration of agent evolution against one Amazon SOP-Bench domain by proposing ONE **workflow graph patch** — `add_node` / `replace_node` / `disable_node` applied to the frontier workflow at `meta_harness/workflows/sopbench_<domain>.yaml`. The node added or replaced is one **component**: a single Python file under `agent/components_sopbench_<domain>/<name>.py` exporting `COMPONENT: Component`. **You do NOT run benchmarks.** You analyse prior baselines + traces, write one component, choose one patch op, write `pending_eval.json`, and exit. The outer loop applies the patch, scores it on the domain's train subset (30 rows), and admits or rejects.

## What SOP-Bench tasks look like

A task = one row from `third_party/SOP-Bench/src/amazon_sop_bench/benchmarks/data/<domain>/test_set_with_outputs.csv`. The agent receives:
  * `sop` (str): the natural-language SOP document
  * `task` (dict): structured inputs (per `metadata.json::input_columns`)
  * `tools` (ToolManager): function-callable utilities the SOP says to call

The agent runs a multi-turn function-calling loop, then emits XML like `<hazard_class>Hazard Class B</hazard_class>`. Common failure modes:
  * **format / case mismatch** — model writes `hazard class b`; grader expects `Hazard Class B`.
  * **incorrect reasoning** — wrong branch of the SOP decision tree.
  * **tool-arg hallucination** — passes a parameter the tool spec doesn't accept.
  * **missing tool call** — answers without invoking the SOP-prescribed validation step.
  * **format guard** — emits XML report with the right value buried in extra tags.

## First principles

1. **The LLM is the last resort.** Each iteration, find one place the LLM is doing work deterministic code could do — output normalisation, tool-arg validation, format extraction, retry on observed failure — and move it into Python.
2. **Decompose; don't defer.** Split a "reasoning-bound" failure into the minimal LLM judgment + the deterministic computation around it. Build the deterministic part.
3. **Code earns its place by capturing stable structure, not by fitting recent failures.** Anything you encode must point at a fact OUTSIDE evidence — a tool's declared schema, a JSON Schema invariant, the SOP's quoted output format, an LLM API field. IF/THEN induced from N failed train rows is a memorised map; it overfits.
4. **Per-domain specialization.** Components live in per-domain dirs and are namespaced by domain. Your work is for ONE domain only; do not optimize for cross-domain transfer.

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

### Events the SOP-Bench runtime emits

The runtime calls `dispatcher.emit("<event_name>", ctx)` at each lifecycle anchor.

Per-task setup:

| event                   | when it fires                                                          | typical use                                              |
|-------------------------|------------------------------------------------------------------------|----------------------------------------------------------|
| `task_received`         | right after `ComponentContext` is built                                | static framework injection                               |
| `session_start`         | once per task                                                          | static system-prompt injection                           |
| `pre_prompt_build`      | per task; default system+user prompts built                            | rewrite user prompt; inject SOP step checklist           |
| `pre_context_build`     | alias of `pre_prompt_build` (cross-sibling vocabulary)                 | same as above                                            |
| `pre_agent_construct`   | last hook before messages list is sealed                               | inference-hint injection                                 |

Per LLM turn (fires multiple times per task):

| event                       | when it fires                                                        | typical use                                                       |
|-----------------------------|----------------------------------------------------------------------|-------------------------------------------------------------------|
| `pre_llm_turn`              | per turn, before `chat()` call                                       | rewrite messages list                                             |
| `pre_llm_request`           | per turn, just before SUT call (Tier-1 alias for `pre_llm_turn`)     | sub-LLM verifier prep                                             |
| `post_llm_response`         | per turn, after `chat()` response                                    | rewrite assistant content; trigger retry via `inject_context`     |
| `post_llm_response_raw`     | alias of `post_llm_response`                                         | same                                                              |
| `on_length_truncation`      | **synthesised** when `finish_reason == "length"`                     | sub-LLM recovery with bigger budget                               |
| `on_empty_response`         | **synthesised** when raw_response is empty                           | reactive retry                                                    |
| `on_no_tool_call_emitted`   | **synthesised** when no tool_calls emitted                           | retry-with-tool reminder                                          |

Per tool call (fires once per ToolCall within a turn):

| event                       | when it fires                                                        | typical use                                                       |
|-----------------------------|----------------------------------------------------------------------|-------------------------------------------------------------------|
| `pre_tool_arg_validation`   | per tool call, narrow schema-check phase before `pre_tool_use`        | rewrite args (drop unknown kwargs) / block                        |
| `pre_tool_use`              | per tool call, main pre-tool decision                                 | validate / rewrite args; block invalid calls                      |
| `post_tool_use`             | per tool call, after invocation                                       | reformat result; insert validation flag                           |
| `post_tool_result_raw`      | per tool call, raw result anchor                                      | observe / inject                                                  |
| `on_tool_error`             | **synthesised** when `current_tool_success == False`                  | retry-hint via `inject_context`                                   |

Per-task exit:

| event                       | when it fires                                                        | typical use                                                       |
|-----------------------------|----------------------------------------------------------------------|-------------------------------------------------------------------|
| `on_explicit_terminate`     | **synthesised** when max_iterations exhaust without final answer      | exit-gate; may BLOCK to refuse termination                        |
| `pre_final_emit`            | after the loop, before returning final output                         | normalise XML casing; ensure required tag present                  |
| `session_end`               | bookkeeping                                                          | —                                                                 |

### Component classes

| class                  | what the matcher tests                                                                              | risk                          |
|------------------------|-----------------------------------------------------------------------------------------------------|-------------------------------|
| `mechanism_layer`      | a system field, a tool-declared JSON Schema, an LLM API field, a general algorithm                  | LOW                           |
| `reactive_guard`       | an observed failure event (tool error, malformed args, empty content, missing XML tag)               | LOW                           |
| `channel`              | task structure — `task_input` keys, SOP excerpt content, an external lookup                          | LOW                           |
| `induced_rule`         | a reading of policy/SOP text — IF/THEN compiled from N evidence rows                                 | HIGH (admitted ADVISORY-ONLY) |
| `predictive_heuristic` | raw prompt / response text — regex/keyword guess about what the model is about to do                 | REJECTED at load time         |

### Decision

| decision         | semantics                                                                                                       |
|------------------|-----------------------------------------------------------------------------------------------------------------|
| `allow`          | no-op                                                                                                           |
| `block`          | terminate task as blocked (final answer empty) — EXCEPT at `pre_tool_use` / `pre_tool_arg_validation` where it skips just this tool call |
| `rewrite`        | replace the live payload at this event (see Event → payload below)                                              |
| `inject_context` | `task_received`/`session_start`/`pre_prompt_build`/etc. → append to `ctx.system_prompt`; post-LLM / post-tool events → queue for next turn |

Event → REWRITE payload:

| event                                          | payload type                | replaces                              |
|------------------------------------------------|-----------------------------|---------------------------------------|
| `pre_prompt_build` / `pre_context_build`       | `str`                       | `ctx.user_prompt`                     |
| `pre_llm_turn`                                 | `list[dict]`                | `ctx.messages`                        |
| `post_llm_response` / `post_llm_response_raw` / `on_length_truncation` / `on_empty_response` / `on_no_tool_call_emitted` | `str` | `ctx.raw_response` |
| `pre_tool_use` / `pre_tool_arg_validation`     | `dict`                      | `ctx.current_tool_args`               |
| `post_tool_use` / `post_tool_result_raw` / `on_tool_error` | `str`           | `ctx.current_tool_result_str`         |
| `pre_final_emit`                               | `str` or `None`             | `ctx.final_output` (None marks blocked) |

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

`ctx.chat(messages, max_tokens=..., temperature=..., system_override=..., tools=...)` routes through `agent.llm.chat` (locked SUT model name).

## The class × event × decision matrix

Load-time gate at `agent/component_runtime_sopbench/policy.py::ALLOWED`. A component whose (cls, listens) is absent raises `ComponentPolicyError` at load.

Setup-phase events (`task_received`, `session_start`, `pre_prompt_build`, `pre_context_build`, `pre_agent_construct`):

| class                  | admissible decisions                            |
|------------------------|-------------------------------------------------|
| `mechanism_layer`      | inject_context, rewrite (pre_prompt_build only), block (pre_prompt_build only) |
| `channel`              | inject_context                                  |
| `induced_rule`         | inject_context (pre_prompt_build / pre_context_build only — ADVISORY) |
| `reactive_guard`       | — (does not fire at setup)                      |

Per-turn / per-tool events (`pre_llm_turn`, `post_llm_response[_raw]`, `on_*`, `pre_tool_use`, `pre_tool_arg_validation`, `post_tool_use`, `post_tool_result_raw`, `on_tool_error`):

| class                  | admissible decisions                                                                |
|------------------------|-------------------------------------------------------------------------------------|
| `mechanism_layer`      | rewrite, block, inject_context (where the cell admits each — see policy.py)         |
| `reactive_guard`       | rewrite, block, inject_context (where the cell admits each)                          |
| `channel` / `induced_rule` | — (do not fire on tool / response events)                                       |

Exit-phase events (`on_explicit_terminate`, `pre_final_emit`, `session_end`):

| class                  | admissible decisions                            |
|------------------------|-------------------------------------------------|
| `mechanism_layer`      | rewrite + block (pre_final_emit); block (on_explicit_terminate); allow (session_end) |
| `reactive_guard`       | same                                            |

`predictive_heuristic` is rejected at every event.

## The workflow graph

Per-domain frontier YAML: `meta_harness/workflows/sopbench_<domain>.yaml`.

```yaml
nodes:
  - sopbench_dangerous_goods_final_xml_recovery
disabled: []
```

Plus a JSON snapshot `meta_harness/logs_components_sopbench_<domain>/frontier_workflow.json`.

### Patch ops

| op             | meaning                                                                                                   |
|----------------|-----------------------------------------------------------------------------------------------------------|
| `add_node`     | append a new node; `agent/components_sopbench_<domain>/<id>.py` must be newly written                     |
| `replace_node` | keep the existing node id; overwrite the file (same `COMPONENT.name`)                                     |
| `disable_node` | add the id to `disabled:`; file remains for the durability audit                                          |

## Hard rules

- Exactly ONE patch per invocation.
- **You do NOT run benchmarks.** No `run_sopbench_baseline.py`, no `evaluate()`. The outer loop scores.
- **No task-specific code.** No domain-specific entity names in matchers (no chemical IDs, partner IDs, PO numbers). No encoded gold answers.
- **The target inference model is LOCKED.** `agent.llm.chat()` rejects any `model=` override. `ctx.chat()` does not accept a `model` kwarg.
- For `replace_node`, the **first shell action** MUST be:
  ```bash
  cp agent/components_sopbench_<domain>/<existing>.py agent/components_sopbench_<domain>/<existing>.py.bak_iter<N>
  ```
- READ-ONLY: `third_party/SOP-Bench/`, `amazon_sop_bench/`, `bench/`, `agent/sopbench_agent.py`, `agent/component_runtime_sopbench/`, `agent/llm.py`, `agent/events.py`, `agent/components/` (GAIA's), `meta_harness/scripts/run_sopbench_baseline.py`, `meta_harness/meta_harness_components_sopbench.py`.
- Component file naming: `agent/components_sopbench_<domain>/component_sopbench_<domain>_iter<N>_<slug>.py`. The `COMPONENT.name` should also include the domain slug. Per-domain dirs ensure other domains' components are not loaded by your runtime.
- Domain context: you are evolving for exactly ONE domain. The outer loop knows the domain; you don't pick it.

## Component file template

```python
# agent/components_sopbench_<domain>/component_sopbench_<domain>_iter<N>_<slug>.py
from __future__ import annotations

from agent.component_runtime_sopbench import (
    Component, ComponentClass, ComponentContext,
    Decision, Trust,
)


def _matches(ctx: ComponentContext) -> bool:
    # Read ctx.raw_response / ctx.final_output / ctx.current_tool_args /
    # ctx.current_tool_result_str / ctx.finish_reason / ctx.shared.
    # ctx.event names the firing event for branchable handlers.
    # NEVER read ctx.task_id (treat it as opaque).
    # NEVER read ctx.task_input fields you are about to compare against —
    # that's just memorisation of the test set.
    ...


def _handler(ctx: ComponentContext) -> Decision:
    return Decision.rewrite(...)        # or inject_context / block / allow


COMPONENT = Component(
    name="sopbench_<domain>_<slug>",
    cls=ComponentClass.REACTIVE_GUARD,
    listens="pre_final_emit",
    matcher=_matches,
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
    "name": "candidate_iter<N>_<slug>",
    "hypothesis": "one-sentence claim of the failure mode",
    "changes": "one-sentence description of what your component does",
    "component": {
      "name": "<COMPONENT.name>",
      "cls": "reactive_guard",
      "listens": "pre_final_emit",
      "file": "agent/components_sopbench_<domain>/component_sopbench_<domain>_iter<N>_<slug>.py",
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
      "file": "agent/components_sopbench_<domain>/component_sopbench_<domain>_iter<N>_<slug>.py"
    }
  }
}
```

For `disable_node`, omit `file` and the only `name` is the existing component's name. For `replace_node`, `name` reuses the existing `COMPONENT.name`; the `file` is the same path you overwrote (the runtime registry uses last-loaded-wins by name).

## How to investigate

1. **Read frontier_val.json's `per_task`** to find tasks the frontier scores 0 on. Each entry has `agent`, `score`, `predicted`.
2. **Read v0 baseline results.json** at `meta_harness/logs_components_sopbench_<domain>/v0__train__results.json`. Each task result has `task_id`, `success` (= correct), `predicted_output`, `expected_output`, `tool_calls`, `error`, `reasoning_trace`.
3. **Look for ≥3 failures sharing the same mechanism**. Examples:
   - All 3 outputs are correctly-named XML tags but lowercase → `pre_final_emit` REWRITE that title-cases the inner text.
   - All 3 failures call `calculate_X_score` with an extra `assessmentFormId` param the tool rejects → `pre_tool_arg_validation` REWRITE that strips unknown kwargs.
   - All 3 failures emit prose without any XML tag → `post_llm_response` REACTIVE_GUARD that on regex-miss `inject_context` a retry hint.
4. **Form ONE hypothesis** tied to a stable structure (the JSON Schema of the tool, the SOP's quoted Output section, an OpenAI-API field). State the structure in `trust.evidence_anchor`.
5. **Write ONE component** with the smallest possible matcher/handler. Resist embedding domain-specific entity names.
6. **Validate**:
   ```bash
   python -c "
   from agent.component_runtime_sopbench import load_components_from_dir
   comps = load_components_from_dir(only=['<COMPONENT.name>'])
   assert any(c.name == '<COMPONENT.name>' for c in comps)
   print('ok')
   "
   ```
7. **Write `pending_eval.json`** and print `CANDIDATE: <name>`.

## Common patterns

| pattern                                          | class             | listens                       | decision                                    |
|--------------------------------------------------|-------------------|-------------------------------|---------------------------------------------|
| inject SOP output-format hint                    | mechanism_layer   | `session_start`               | inject_context                              |
| inject SOP step checklist                        | channel           | `pre_prompt_build`            | inject_context                              |
| advisory: "valid params include X, Y, Z"         | induced_rule      | `pre_prompt_build`            | inject_context                              |
| schema-validate tool args (drop unknown kwargs)  | mechanism_layer   | `pre_tool_arg_validation`     | rewrite                                     |
| block tool call with missing required param      | reactive_guard    | `pre_tool_use`                | block                                       |
| retry hint on tool error                         | reactive_guard    | `on_tool_error`               | inject_context                              |
| length-recovery via sub-LLM with bigger budget   | reactive_guard    | `on_length_truncation`        | rewrite (via `ctx.chat(max_tokens=32768)`)  |
| regex-miss on XML tag → retry                    | reactive_guard    | `post_llm_response`           | inject_context (retry hint)                 |
| ensure XML tag exists at final emit              | reactive_guard    | `pre_final_emit`              | rewrite / block                             |
| title-case XML inner text                        | mechanism_layer   | `pre_final_emit`              | rewrite                                     |
| refuse termination without artifact              | reactive_guard    | `on_explicit_terminate`       | block                                       |

## Custom events (Tier 2/3)

A component can emit its own event name to coordinate with sibling components in the same task. Convention:

```
iter<N>_<slug>_<event>        e.g. iter9_tool_arg_schema_filter_dropped_unknown_kwarg
on_<thing>                    cross-iter failure-mode name
```

Always declare `emits=("iter<N>_<slug>_<event>", ...)` on the publisher. Subscribers declare `listens="iter<N>_<slug>_<event>"`. To admit non-`allow` decisions on a custom event, add a string-key entry to `ALLOWED[ComponentClass.X]` in `agent/component_runtime_sopbench/policy.py`. Grep `agent/components_sopbench_<domain>/*.py` for `emits=` to find taken names.

## What NOT to write

- Components that memorise train-set answers (encode `hazard_class = "C"` for product_id starting with `P_13`).
- Components that try to compute the SOP output deterministically without calling the LLM (defeats the SUT measurement).
- Components that touch `agent/llm.py` or the SopBenchAgent loop.
- Components that call an LLM with a different `model=` kwarg.
- Components that fire on every task (`matcher=None`, priority=0) — that's effectively a SOP rewrite, not a component.
