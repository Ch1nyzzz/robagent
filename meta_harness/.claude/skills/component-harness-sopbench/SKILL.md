---
name: component-harness-sopbench
description: Run ONE iteration of Amazon SOP-Bench harness evolution by proposing ONE workflow graph patch — a single Python file declaring `COMPONENT: Component` plus a patch op (add_node / replace_node / disable_node) against the frontier workflow at `meta_harness/workflows/sopbench_<domain>.yaml`. The component is typed by mount + class + state_scope, gated by a class×mount×decision permission matrix, and verified by a Trust block (evidence_anchor + out_of_evidence_probe). Sibling of component-harness-gaia/tau2; adds tool-use mounts (PRE_TOOL_USE / POST_TOOL_USE) and a per-turn mount (PRE_LLM_TURN) to match the SopBenchAgent function-calling loop.
---

# component-harness-sopbench

Run ONE iteration of agent evolution against one Amazon SOP-Bench domain by proposing ONE **workflow graph patch**. A patch is `add_node` / `replace_node` / `disable_node`, applied to the frontier workflow at `meta_harness/workflows/sopbench_<domain>.yaml`. The node added or replaced is one **component** — a single Python file in `agent/components_sopbench_<domain>/<name>.py` exporting `COMPONENT: Component`. **You do NOT run benchmarks.** You analyse prior baselines + traces, write one component, choose one patch op, write `pending_eval.json`, and exit. The outer loop applies the patch, scores it on the domain's train subset (30 rows), and admits or rejects.

## What SOP-Bench tasks look like

A task = one row from `third_party/SOP-Bench/src/amazon_sop_bench/benchmarks/data/<domain>/test_set_with_outputs.csv`. The agent receives:
  * `sop` (str): the natural-language SOP document
  * `task` (dict): structured inputs (per `metadata.json::input_columns`)
  * `tools` (ToolManager): function-callable utilities the SOP says to call

The agent runs a multi-turn function-calling loop, then emits XML like `<hazard_class>Hazard Class B</hazard_class>`. The expected output is one column per `metadata.json::output_columns`. Failure is normally one of:
  * **format / case mismatch** — model writes `hazard class b` (lowercase, no "Hazard"); parser extracts `hazard class b` but grader expects `Hazard Class B`.
  * **incorrect reasoning** — model picks the wrong branch of the SOP's decision tree.
  * **tool-arg hallucination** — model passes a parameter the tool spec doesn't accept (`assessmentFormId` etc.).
  * **missing tool call** — model answers without invoking the SOP-prescribed validation step.
  * **format guard** — model emits a long XML report containing the right value buried in extra tags the parser doesn't know about.

Your job is to pick ONE such mechanism present in ≥3 train failures and add ONE component that addresses it.

## First principles

1. **The LLM is the last resort.** Each iteration, find one place the LLM is doing work deterministic code could do — output normalisation, tool-arg validation, format extraction, retry on observed failure — and move it into Python.
2. **Decompose; don't defer.** A failure that looks "reasoning-bound" is rarely atomic. Split: what is the minimal judgment the LLM must make, and what computation / lookup / validation around it is fully deterministic? Build the deterministic part.
3. **Code earns its place by capturing stable structure, not by fitting recent failures.** Anything you encode must point at a fact OUTSIDE evidence — a tool's declared schema, a JSON Schema invariant, the SOP's quoted output format, an LLM API field (`finish_reason`). An IF/THEN induced from N failed train rows is a memorised map; it overfits.
4. **Per-domain specialization.** Components live in one shared dir but are namespaced by domain. Your work is for ONE domain only; do not optimize for cross-domain transfer.

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

### Mount enum (SOP-Bench-specific, FC loop)

| mount               | when it fires                                                  | typical use                                                       |
|---------------------|----------------------------------------------------------------|-------------------------------------------------------------------|
| `session_start`     | once per task, before any LLM call                             | static system-prompt injection (e.g. add an output-format hint)   |
| `pre_prompt_build`  | per task, after default system+user prompts built              | rewrite user prompt; inject SOP step checklist into system_prompt |
| `pre_llm_turn`      | per turn, before each chat() call                              | rewrite messages list (e.g., trim noisy history; add reminder)    |
| `post_llm_response` | per turn, after each chat() response                           | rewrite assistant content; trigger retry via inject_context        |
| `pre_tool_use`      | before each tool dispatch                                      | validate / rewrite tool args; block invalid calls                 |
| `post_tool_use`     | after each tool result, before adding to messages              | reformat result; insert validation flag                            |
| `pre_final_emit`    | after the loop, before returning final output                  | normalise XML casing; ensure required tag present                  |
| `session_end`       | bookkeeping only                                               | —                                                                 |

**Per-turn vs. per-call**: `pre_llm_turn` and `post_llm_response` fire EACH turn (potentially many times per task). `pre_tool_use` and `post_tool_use` fire EACH tool call within a turn (a turn can have 0..N tool calls). Be mindful of side-effects.

### ComponentClass enum

| class                 | the matcher tests…                                                                                    | risk                          |
|-----------------------|-------------------------------------------------------------------------------------------------------|-------------------------------|
| `mechanism_layer`     | a system field, a tool-declared JSON Schema, an LLM API field, a general algorithm                    | LOW                           |
| `reactive_guard`      | an observed failure event (tool error, malformed args, empty content, missing XML tag, retry-needed)  | LOW                           |
| `channel`             | task structure — `task_input` keys, SOP excerpt content, an external lookup                           | LOW                           |
| `induced_rule`        | a reading of policy/SOP text — IF/THEN compiled from N evidence rows                                  | HIGH (admitted ADVISORY-ONLY) |
| `predictive_heuristic`| raw prompt / response text — regex/keyword guess about what the model is about to do                  | REJECTED at load time         |

### Decision

| decision         | semantics (mount-dependent payload)                                                                             |
|------------------|-----------------------------------------------------------------------------------------------------------------|
| `allow`          | no-op                                                                                                           |
| `block`          | terminate task as blocked (final answer empty) — EXCEPT at `pre_tool_use` where it skips just this tool call    |
| `rewrite`        | replace the live payload at this mount (see semantics table below)                                              |
| `inject_context` | `session_start`/`pre_prompt_build` → append to `ctx.system_prompt`; `post_llm_response` → queue for next turn   |

REWRITE payload by mount:

| mount               | payload type                | replaces                              |
|---------------------|-----------------------------|---------------------------------------|
| `pre_prompt_build`  | `str`                       | `ctx.user_prompt`                     |
| `pre_llm_turn`      | `list[dict]`                | `ctx.messages`                        |
| `post_llm_response` | `str`                       | `ctx.raw_response`                    |
| `pre_tool_use`      | `dict`                      | `ctx.current_tool_args`               |
| `post_tool_use`     | `str`                       | `ctx.current_tool_result_str`         |
| `pre_final_emit`    | `str` (or `None` → blocked) | `ctx.final_output`                    |

### StateScope

`none` (default) / `session` (per-task scratchpad at `ctx.state[component_name]`) / `cross_session` (reserved).

### Trust

```python
@dataclass
class Trust:
    evidence_anchor: str        # REQUIRED — name the stable structure being targeted
    blast_radius: str           # REQUIRED: local | workflow | global
    rollback_when: str          # REQUIRED — concrete signal that says "stop using this"
    out_of_evidence_probe: str  # REQUIRED for INDUCED_RULE
    fallback: str               # OPTIONAL
```

### Capability

`none` / `read_file` / `http_get` / `llm_call` (sub-LLM for verifier/re-format/critic) / `tool_call` / `mutate_shared`.

## The class × mount × decision matrix

Load-time gate at `agent/component_runtime_sopbench/policy.py::ALLOWED`.

| class \ mount         | session_start | pre_prompt_build              | pre_llm_turn         | post_llm_response          | pre_tool_use         | post_tool_use   | pre_final_emit     |
|-----------------------|---------------|-------------------------------|----------------------|----------------------------|----------------------|-----------------|--------------------|
| `mechanism_layer`     | inject        | inject, rewrite, block        | rewrite, block       | rewrite, block, inject     | rewrite, block       | rewrite         | rewrite, block     |
| `reactive_guard`      | —             | —                             | —                    | rewrite, block, inject     | rewrite, block       | rewrite         | rewrite, block     |
| `channel`             | inject        | inject                        | —                    | —                          | —                    | —               | —                  |
| `induced_rule`        | —             | **inject (advisory)**         | —                    | —                          | —                    | —               | —                  |
| `predictive_heuristic`| rejected      | rejected                      | rejected             | rejected                   | rejected             | rejected        | rejected           |

A component whose (class, mount) is absent raises `ComponentPolicyError` at load. A handler that emits a non-admitted decision raises at fire time.

## The workflow graph

Per-domain frontier yaml: `meta_harness/workflows/sopbench_<domain>.yaml`

```yaml
nodes:
  - sopbench_dangerous_goods_final_xml_normalizer
edges: []
disabled: []
```

Plus a JSON snapshot `meta_harness/logs_components_sopbench_<domain>/frontier_workflow.json` written on accept.

### Patch ops

| op             | meaning                                                                                                   |
|----------------|-----------------------------------------------------------------------------------------------------------|
| `add_node`     | append a new node; `agent/components_sopbench_<domain>/<id>.py` must be newly written                              |
| `replace_node` | keep the existing node id; overwrite the file with new behavior (same `COMPONENT.name`)                    |
| `disable_node` | add the id to `disabled:`; file remains for the durability audit                                          |

## Hard rules

- Exactly ONE patch per invocation.
- **You do NOT run benchmarks.** No `run_sopbench_baseline.py`, no `evaluate()`. The outer loop scores.
- **No task-specific code.** No domain-specific entity names in matchers (no chemical IDs, partner IDs, PO numbers). No encoded gold answers.
- **The target inference model is LOCKED.** It is the System Under Test. Do NOT pass a `model=` kwarg to `chat()` and do NOT call any other model API. `agent.llm.chat()` enforces this at call time — passing any model other than `DEFAULT_MODEL` raises `RuntimeError`. (No flash/lite/cheaper-variant fallbacks. No second-opinion calls to a different model.)
- For `replace_node`, the **first shell action** MUST be:
  ```bash
  cp agent/components_sopbench_<domain>/<existing>.py agent/components_sopbench_<domain>/<existing>.py.bak_iter<N>
  ```
  The outer loop relies on the `.bak` to roll back on reject.
- READ-ONLY: `third_party/SOP-Bench/`, `amazon_sop_bench/`, `bench/`, `agent/sopbench_agent.py`, `agent/component_runtime_sopbench/`, `agent/llm.py`, `agent/events.py`, `agent/components/` (GAIA's), `meta_harness/scripts/run_sopbench_baseline.py`, `meta_harness/meta_harness_components_sopbench.py`. The outer loop reads your output; you do not modify the loop.
- Component file naming convention: `agent/components_sopbench_<domain>/component_sopbench_<domain>_iter<N>_<slug>.py`. The `COMPONENT.name` should also include the domain slug (e.g. `sopbench_dangerous_goods_final_xml_normalizer`). Per-domain dirs ensure other domains' components are not loaded by your runtime; do NOT read or edit files under any other domain's dir.
- Domain context: you are evolving for exactly ONE domain. The `pending_eval.json` is read by the outer loop which knows the domain; you don't pick the domain.

## Component file interface

```python
# agent/components_sopbench_<domain>/component_sopbench_<domain>_iter<N>_<slug>.py
from __future__ import annotations

from agent.component_runtime_sopbench import (
    Capability, Component, ComponentClass, ComponentContext,
    Decision, Mount, StateScope, Trust,
)


def _matches(ctx: ComponentContext) -> bool:
    # Read ctx.raw_response / ctx.final_output / ctx.current_tool_args /
    # ctx.current_tool_result_str / ctx.finish_reason / ctx.shared.
    # NEVER read ctx.task_id (treat it as opaque).
    # NEVER read ctx.task_input fields you are about to compare against —
    # that's just memorisation of the test set.
    ...


def _handler(ctx: ComponentContext) -> Decision:
    return Decision.rewrite(...)   # or inject_context / block / allow


COMPONENT = Component(
    name="sopbench_<domain>_<slug>",
    cls=ComponentClass.REACTIVE_GUARD,
    mount=Mount.PRE_FINAL_EMIT,
    matcher=_matches,
    handler=_handler,
    state_scope=StateScope.NONE,
    capabilities=(Capability.NONE,),
    priority=100,
    trust=Trust(
        evidence_anchor="...",      # what stable structure is this targeting?
        blast_radius="local",       # local | workflow | global
        rollback_when="...",        # how would you know to disable this?
        # out_of_evidence_probe="..." if INDUCED_RULE
    ),
)
```

## `pending_eval.json` schema

The outer loop reads this file after your run. Write it once, validated.

```json
{
  "candidate": {
    "name": "mh_iter<N>_<slug>",
    "hypothesis": "one-sentence claim of what the failure mode is",
    "changes": "one-sentence description of what your component does",
    "component": {
      "name": "<COMPONENT.name>",
      "cls": "reactive_guard",
      "mount": "pre_final_emit",
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
      "file": "agent/components_sopbench_<domain>/component_sopbench_<domain>_iter<N>_<slug>.py",
      "edges_in": [],
      "edges_out": []
    }
  }
}
```

For `disable_node`, omit `file` and the only `name` is the existing component's name. For `replace_node`, `name` reuses the existing `COMPONENT.name`; the `file` is the same path you overwrote (the runtime registry uses last-loaded-wins by name).

## How to investigate

1. **Read frontier_val.json's `per_task`** to find tasks the frontier scores 0 on. Each entry has `agent`, `score`, `predicted`.
2. **Read v0 baseline results.json** at `meta_harness/logs_components_sopbench_<domain>/v0__train__results.json`. Each task result has `task_id`, `success` (= correct), `predicted_output`, `expected_output`, `tool_calls`, `error`, `reasoning_trace`.
3. **Look for ≥3 failures sharing the same mechanism**. Examples:
   - All 3 outputs are correctly-named XML tags but lowercase (`hazard class b` vs `Hazard Class B`) → `pre_final_emit` REWRITE that title-cases the inner text.
   - All 3 failures call `calculate_X_score` with an extra `assessmentFormId` param the tool rejects → `pre_tool_use` REWRITE that strips unknown kwargs, OR `session_start` INJECT a note about valid params.
   - All 3 failures emit prose without any XML tag → `post_llm_response` REACTIVE_GUARD that on regex-miss INJECT_CONTEXT a retry hint reminding the model of the expected tag.
4. **Form ONE hypothesis** and tie it to a stable structure (the JSON Schema of the tool, the SOP's quoted Output section, an OpenAI-API field). State the structure in `trust.evidence_anchor`.
5. **Write ONE component** at the appropriate mount with the smallest possible matcher/handler. Resist embedding domain-specific entity names.
6. **Validate registration**:
   ```bash
   python -c "
   from agent.component_runtime_sopbench import load_components_from_dir
   comps = load_components_from_dir(only=['<COMPONENT.name>'])
   assert any(c.name == '<COMPONENT.name>' for cs in comps.values() for c in cs)
   print('ok')
   "
   ```
7. **Write `pending_eval.json`** and print `CANDIDATE: <name>`.

## What NOT to write

- Components that memorise train-set answers (encode `hazard_class = "C"` for product_id starting with `P_13`).
- Components that try to compute the SOP output deterministically without calling the LLM (defeats the SUT measurement).
- Components that touch agent/llm.py or the SopBenchAgent loop.
- Components that call an LLM with a different `model=` kwarg.
- Components that fire on every task (priority=0 with always-True matcher) — that's effectively a SOP rewrite, not a component.
