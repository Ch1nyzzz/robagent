---
name: component-harness-enterpriseops
description: Run ONE iteration of ServiceNow EnterpriseOps-Gym harness evolution by proposing ONE workflow graph patch — a single Python file declaring `COMPONENT: Component` plus a patch op (add_node / replace_node / disable_node) against the frontier workflow at `meta_harness/workflows/enterpriseops_<domain>.yaml`. Components subscribe to a named event via `listens=` and react via a Decision; admission is gated by a class×event×decision permission matrix and a Trust block (evidence_anchor + out_of_evidence_probe). EnterpriseOps targets the upstream React loop over MCP gym servers with SQL-verifier judging on the final DB state.
---

# component-harness-enterpriseops

Run ONE iteration of agent evolution against one EnterpriseOps-Gym domain by proposing ONE **workflow graph patch** — `add_node` / `replace_node` / `disable_node` applied to `meta_harness/workflows/enterpriseops_<domain>.yaml`. The node added or replaced is one **component**: a single Python file under `agent/components_enterpriseops_<domain>/<name>.py` exporting `COMPONENT: Component`. **You do NOT run benchmarks.** You analyse prior baselines + traces, write one component, choose one patch op, write `pending_eval.json`, and exit.

## What EnterpriseOps-Gym tasks look like

A task = one row from HuggingFace `ServiceNow-AI/EnterpriseOps-Gym` split `<domain>` config `oracle` (or `plus_5_tools` / `plus_10_tools` / `plus_15_tools` for tool-retrieval stress modes). The agent receives:
  * `system_prompt` (str) — agent role + domain policy
  * `user_prompt` (str) — natural-language task instruction
  * `selected_tools` (list[str]) — oracle tool set (or oracle + N distractors)
  * `gym_servers_config` (list[dict]) — one or more dockerized MCP servers
  * `verifiers` (list[dict]) — **OPAQUE to the agent**: SQL queries the judge runs over the final DB state. You may NOT read `verifiers` from a component.

The agent runs upstream's ReAct loop (`third_party/EnterpriseOps-Gym/orchestrators/react.py`) using LangChain + MCP tool dispatch. **Judging is outcome-based**: each task's verifiers run as SQL queries on the gym DB after the loop ends; a task passes only if ALL verifiers match. There is no "predicted output" text to score.

Common failure modes (calendar / itsm baselines):
  * **Tool-arg shape error** — `start: "2025-11-14T15:00"` vs expected `start_datetime` + separate `timezone`.
  * **Wrong tool selected** — `get_event` to search by name when `list_events` + filter is required.
  * **Field-format mismatch** — ACL with `scope_email: "carol"` vs verifier expecting `"carol.white@techcorp.com"`.
  * **Missing intermediate step** — event created but `insert_acl_rule` never called.
  * **Date/timezone confusion** — `start_timezone: 'America/New_York'` when verifier expects `'UTC'`.
  * **Loop exhaustion** — hits `max_iterations` mid-task.

## First principles

1. **The LLM is the last resort.** Move tool-arg validation, parameter normalisation (email casing, datetime→UTC), required-field enforcement, retry-on-failure into Python.
2. **Decompose; don't defer.** Split a "reasoning-bound" failure into the minimal LLM judgment + the deterministic computation around it.
3. **Code earns its place by capturing stable structure, not by fitting recent failures.** Anchor on an MCP tool's JSON Schema, a timezone library, the gym server's `list_tools` response, an LLM API field. IF/THEN induced from N failed train rows is a memorised map; it overfits.
4. **Per-domain specialization.** Components live in per-domain dirs and are namespaced by domain. Work for ONE domain only.
5. **Verifiers are opaque.** Components key off MCP tool schemas, user prompt structure, tool errors, `finish_reason` — NEVER the SQL queries in `verifiers`.

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
    state_scope: StateScope = StateScope.NONE
    capabilities: tuple[Capability, ...] = (Capability.NONE,)
    priority: int = 100
    emits: tuple[str, ...] = ()
```

### Events the EnterpriseOps runtime emits

Per-task setup:

| event                   | when it fires                                                          | typical use                                              |
|-------------------------|------------------------------------------------------------------------|----------------------------------------------------------|
| `task_received`         | top of `run_task`, after `ComponentContext` is built                   | lifecycle anchor                                         |
| `session_start`         | once per task, after MCP tools discovered                              | "always pass full email in ACL scope" style injection    |
| `pre_prompt_build`      | per task; default system+user prompts built                            | rewrite user prompt; inject domain policy reminder       |
| `pre_context_build`     | alias of `pre_prompt_build` (cross-sibling vocabulary)                 | same as above                                            |
| `pre_agent_construct`   | last hook before BenchmarkConfig is sealed                             | inference-hint injection                                 |

Per LLM turn (avg ~9 turns / task, max 34):

| event                       | when it fires                                                         | typical use                                                       |
|-----------------------------|-----------------------------------------------------------------------|-------------------------------------------------------------------|
| `pre_llm_turn`              | per turn, before `llm_client.invoke_with_tools()`                     | rewrite messages list (re-add user_info context)                  |
| `pre_llm_request`           | per turn (Tier-1 alias for `pre_llm_turn`)                            | sub-LLM verifier prep                                             |
| `post_llm_response`         | per turn, after LangChain response                                    | rewrite assistant content; queue retry hint                       |
| `post_llm_response_raw`     | alias of `post_llm_response`                                          | same                                                              |
| `on_length_truncation`      | **synthesised** when `finish_reason == "length"`                      | sub-LLM recovery with bigger budget                               |
| `on_empty_response`         | **synthesised** when raw_response is empty                            | reactive retry                                                    |
| `on_no_tool_call_emitted`   | **synthesised** when no tool_calls emitted                            | retry-with-tool reminder                                          |

Per MCP tool call (max ~11 / turn):

| event                       | when it fires                                                         | typical use                                                       |
|-----------------------------|-----------------------------------------------------------------------|-------------------------------------------------------------------|
| `pre_tool_arg_validation`   | per MCP tool call, narrow schema-check phase before `pre_tool_use`     | rewrite args (normalise email casing, fill UTC) / block           |
| `pre_tool_use`              | per MCP tool call, main pre-tool decision                              | validate / rewrite args; block invalid calls                      |
| `post_tool_use`             | per MCP tool call, after invocation                                    | reformat result; insert validation flag                           |
| `post_tool_result_raw`      | per MCP tool call, raw result anchor                                   | observe / inject                                                  |
| `on_tool_error`             | **synthesised** when `current_tool_success == False`                   | retry-hint via `inject_context`                                   |

Per-task exit:

| event                       | when it fires                                                         | typical use                                                       |
|-----------------------------|-----------------------------------------------------------------------|-------------------------------------------------------------------|
| `on_explicit_terminate`     | **synthesised** when assistant emits final answer w/o tool calls       | exit-gate; may BLOCK to refuse termination                        |
| `pre_final_emit`            | after loop exit, AFTER upstream SQL verifiers ran (observational)      | observational only — verifier outcome is fixed by now             |
| `session_end`               | bookkeeping                                                           | —                                                                 |

**Why `pre_final_emit` is weak**: judging is outcome-based on the final DB state, not on the agent's last text. Rewriting `ctx.final_output` does not change the verifier result — it only changes what downstream consumers see logged. Useful components target `pre_tool_arg_validation` / `pre_tool_use` (deterministic arg fix-up), `post_llm_response` / `on_*` (catch and retry), and `session_start` / `pre_prompt_build` (one-shot policy injection).

### Component classes

| class                  | what the matcher tests                                                                              | risk                          |
|------------------------|-----------------------------------------------------------------------------------------------------|-------------------------------|
| `mechanism_layer`      | an MCP tool JSON Schema, a system field, an LLM API field, a general algorithm                      | LOW                           |
| `reactive_guard`       | an observed failure event (tool 4xx/error, malformed args, empty content)                            | LOW                           |
| `channel`              | task structure — `user_info` fields, gym server names, user_prompt structure                         | LOW                           |
| `induced_rule`         | a reading of domain policy text — IF/THEN compiled from N evidence rows                              | HIGH (admitted ADVISORY-ONLY) |
| `predictive_heuristic` | raw prompt / response text — regex/keyword guess about what the model is about to do                 | REJECTED at load time         |

### Decision

| decision         | semantics                                                                                                       |
|------------------|-----------------------------------------------------------------------------------------------------------------|
| `allow`          | no-op                                                                                                           |
| `block`          | terminate task as blocked — EXCEPT at `pre_tool_use` / `pre_tool_arg_validation` where it skips just this tool call |
| `rewrite`        | replace the live payload at this event (see Event → payload below)                                              |
| `inject_context` | pre-LLM events → append to `ctx.system_prompt`; post-LLM / post-tool events → queue for next turn               |

Event → REWRITE payload:

| event                                          | payload type                | replaces                              |
|------------------------------------------------|-----------------------------|---------------------------------------|
| `pre_prompt_build` / `pre_context_build`       | `str`                       | `ctx.user_prompt`                     |
| `pre_llm_turn`                                 | `list[BaseMessage]`         | `ctx.messages`                        |
| `post_llm_response*` / `on_length_truncation` / `on_empty_response` / `on_no_tool_call_emitted` | `str` | `ctx.raw_response` |
| `pre_tool_use` / `pre_tool_arg_validation`     | `dict`                      | `ctx.current_tool_args`               |
| `post_tool_use` / `post_tool_result_raw` / `on_tool_error` | `str`           | `ctx.current_tool_result_str`         |
| `pre_final_emit`                               | `str` or `None`             | `ctx.final_output` (observational)    |

### StateScope

`none` (default) / `session` (per-task scratchpad at `ctx.state[component_name]`) / `cross_session` (reserved).

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

### Capability

| capability        | what it permits                                                       |
|-------------------|-----------------------------------------------------------------------|
| `none`            | pure function                                                         |
| `read_file`       | `open(path, "r")` on workspace paths                                  |
| `http_get`        | outbound HTTP GET                                                     |
| `llm_call`        | invoke `ctx.chat(...)` (locked SUT model via `agent.llm.chat`, NOT the upstream langchain client) |
| `tool_call`       | issue a sub-tool-call within the handler                              |
| `mutate_shared`   | write to `ctx.shared`                                                 |

`ctx.chat(messages, max_tokens=..., temperature=..., system_override=..., tools=...)` routes through `agent.llm.chat` (locked SUT model name, NOT upstream LangChain). `ctx.emit(custom_event_name, **fields)` re-enters the dispatcher (depth cap = 10). `ctx.emit_upstream(key, value)` writes to `ctx.upstream`.

## The class × event × decision matrix

Load-time gate at `agent/component_runtime_enterpriseops/policy.py::ALLOWED`.

Setup events: `mechanism_layer` admits `inject_context` (+ rewrite/block on `pre_prompt_build` / `pre_context_build`); `channel` admits `inject_context`; `induced_rule` admits `inject_context` (ADVISORY) on `pre_prompt_build` / `pre_context_build`; `reactive_guard` does not fire.

Per-turn / per-tool events: `mechanism_layer` and `reactive_guard` admit `rewrite` / `block` / `inject_context` per the cell (see policy.py); `channel` / `induced_rule` do not fire.

Exit events: `mechanism_layer` and `reactive_guard` admit `rewrite` + `block` on `pre_final_emit`, `block` on `on_explicit_terminate`, `allow` on `session_end`.

`predictive_heuristic` is rejected at every event.

## The workflow graph

Per-domain frontier YAML: `meta_harness/workflows/enterpriseops_<domain>.yaml`.

```yaml
nodes: []
edges: []
disabled: []
```

Plus a JSON snapshot `meta_harness/logs_components_enterpriseops_<domain>/frontier_workflow.json`.

### Patch ops

| op             | meaning                                                                                                              |
|----------------|----------------------------------------------------------------------------------------------------------------------|
| `add_node`     | append a new node; `agent/components_enterpriseops_<domain>/<id>.py` must be newly written                            |
| `replace_node` | keep the existing node id; overwrite the file (same `COMPONENT.name`)                                                 |
| `disable_node` | add the id to `disabled:`; file remains for the durability audit                                                      |

## Hard rules

- Exactly ONE patch per invocation.
- **You do NOT run benchmarks.** No `run_enterpriseops_baseline.py`, no `evaluate.py`. The outer loop scores.
- **No task-specific code.** No domain-specific entity names in matchers (no calendar names, no user names, no UUIDs). No encoded gold answers.
- **You may NOT read `ctx.verifiers`** to drive component logic — that is the test set's grading rubric. Components target structure (tool schemas, user_prompt patterns, finish_reason) — never the SQL queries.
- **The target inference model is LOCKED via the upstream LLM config.** Components do NOT call langchain / direct API for additional model calls. `ctx.chat()` IS permitted (goes through `agent.llm.chat`, same locked model name). Declare `Capability.LLM_CALL`.
- For `replace_node`, the **first shell action** MUST be:
  ```bash
  cp agent/components_enterpriseops_<domain>/<existing>.py agent/components_enterpriseops_<domain>/<existing>.py.bak_iter<N>
  ```
- READ-ONLY: `third_party/EnterpriseOps-Gym/`, `agent/enterpriseops_agent.py`, `agent/component_runtime_enterpriseops/`, `agent/llm.py`, `meta_harness/scripts/run_enterpriseops_baseline.py`, `meta_harness/scripts/enterpriseops_smoke.py`, `meta_harness/scripts/select_enterpriseops_split.py`.
- Component file naming: `agent/components_enterpriseops_<domain>/component_enterpriseops_<domain>_iter<N>_<slug>.py`. The `COMPONENT.name` should include the domain slug.
- Domain context: you are evolving for exactly ONE domain. The outer loop knows the domain; you don't pick it.

## Component file template

```python
# agent/components_enterpriseops_<domain>/component_enterpriseops_<domain>_iter<N>_<slug>.py
from __future__ import annotations

from agent.component_runtime_enterpriseops import (
    Capability, Component, ComponentClass, ComponentContext,
    Decision, StateScope, Trust,
)


def _matches(ctx: ComponentContext) -> bool:
    # Read ctx.raw_response / ctx.final_output / ctx.current_tool_args /
    # ctx.current_tool_result_str / ctx.finish_reason / ctx.shared /
    # ctx.user_info / ctx.tool_specs / ctx.current_tool_name.
    # ctx.event names the firing event for branchable handlers.
    # NEVER read ctx.verifiers (= grading rubric).
    # NEVER read ctx.task_id (treat as opaque).
    ...


def _handler(ctx: ComponentContext) -> Decision:
    return Decision.rewrite(...)        # or inject_context / block / allow


COMPONENT = Component(
    name="enterpriseops_<domain>_<slug>",
    cls=ComponentClass.MECHANISM_LAYER,
    listens="pre_tool_use",
    matcher=_matches,
    handler=_handler,
    state_scope=StateScope.NONE,
    capabilities=(Capability.NONE,),
    priority=100,
    emits=(),
    trust=Trust(
        evidence_anchor="...",
        blast_radius="local",
        rollback_when="...",
        out_of_evidence_probe="",
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
      "cls": "mechanism_layer",
      "listens": "pre_tool_use",
      "file": "agent/components_enterpriseops_<domain>/component_enterpriseops_<domain>_iter<N>_<slug>.py",
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
      "file": "agent/components_enterpriseops_<domain>/component_enterpriseops_<domain>_iter<N>_<slug>.py",
      "edges_in": [],
      "edges_out": []
    }
  }
}
```

For `disable_node`, omit `file` and the only `name` is the existing component's name. For `replace_node`, `name` reuses the existing `COMPONENT.name`.

## How to investigate

1. **Read `frontier_val.json`'s `per_task`** at `meta_harness/logs_components_enterpriseops_<domain>/frontier_val.json` to find tasks the frontier scores 0 on. Each entry has `agent`, `score` (0/1), `verifier_pass_rate` (partial credit; useful for picking "almost-correct" failures).
2. **Read baseline results** at `meta_harness/logs_components_enterpriseops_<domain>/baseline/<ts>/{train,test}/results/run_1/results_*.json`. Each file has `runs[0]` with `conversation_flow`, `tools_used`, `tool_results`, `verification_results` (per-verifier name → {passed, error, details}), `verification_summary`, `overall_success`.
3. **Look for ≥3 failures sharing the same mechanism**. Examples:
   - 3 failures call `insert_acl_rule` with `scope_email: "carol"` but verifier checks `scope_email = 'carol.white@techcorp.com'` → `pre_tool_arg_validation` MECHANISM_LAYER that looks up the full email in `ctx.user_prompt` / `ctx.user_info` and rewrites.
   - 3 failures emit `create_event` with `start_datetime: "2025-11-14T15:00"` and no timezone → `pre_tool_arg_validation` REWRITE that normalises any naive datetime to UTC and fills `start_timezone`.
   - 3 failures' assistant content shows a SQL-style WHERE clause → `post_llm_response` REACTIVE_GUARD that `inject_context`s "use the list_X tool with a filter argument".
4. **Form ONE hypothesis** and tie it to a stable structure (the MCP tool's JSON Schema, ISO 8601 invariant, RFC 5322 email regex). State the structure in `trust.evidence_anchor`.
5. **Write ONE component** with the smallest possible matcher/handler. Resist embedding specific entity names (no `"carol.white@techcorp.com"` literal — look it up from `ctx.user_info` / `ctx.user_prompt` at fire time).
6. **Validate**:
   ```bash
   python -c "
   from agent.component_runtime_enterpriseops import load_components_from_dir
   from pathlib import Path
   comps = load_components_from_dir(
       Path('agent/components_enterpriseops_<domain>'),
       only=['<COMPONENT.name>'],
   )
   assert any(c.name == '<COMPONENT.name>' for c in comps)
   print('ok')
   "
   ```
7. **Write `pending_eval.json`** and print `CANDIDATE: <name>`.

## Common patterns

| pattern                                              | class             | listens                       | decision                                    |
|------------------------------------------------------|-------------------|-------------------------------|---------------------------------------------|
| inject "always pass full email in ACL scope"         | mechanism_layer   | `session_start`               | inject_context                              |
| inject domain policy reminder                        | channel           | `pre_prompt_build`            | inject_context                              |
| advisory: policy paragraph paraphrase                | induced_rule      | `pre_context_build`           | inject_context                              |
| normalise bare-local-part email to full address      | mechanism_layer   | `pre_tool_arg_validation`     | rewrite                                     |
| fill missing `start_timezone='UTC'` on naive datetime | mechanism_layer  | `pre_tool_arg_validation`     | rewrite                                     |
| block tool call with missing required param          | reactive_guard    | `pre_tool_use`                | block                                       |
| retry hint on MCP 4xx                                | reactive_guard    | `on_tool_error`               | inject_context (queued for next turn)       |
| length-recovery via sub-LLM with bigger budget       | reactive_guard    | `on_length_truncation`        | rewrite (via `ctx.chat(max_tokens=32768)`)  |
| "use list_X with filter, don't write SQL"            | reactive_guard    | `post_llm_response`           | inject_context                              |
| refuse termination without expected DB write         | reactive_guard    | `on_explicit_terminate`       | block                                       |

## Custom events (Tier 2/3)

```
iter<N>_<slug>_<event>        e.g. iter9_acl_normalize_resolved_email
on_<thing>                    failure-mode style
```

Always declare `emits=(...)` on the publisher. Subscribers declare `listens="iter<N>_<slug>_<event>"`. To admit non-`allow` decisions on a custom event, add a string-key entry to `ALLOWED[ComponentClass.X]` in `agent/component_runtime_enterpriseops/policy.py`.

## What NOT to write

- Components that memorise train-set answers (encode `scope_email = "carol.white@techcorp.com"` for any task whose user_prompt contains "Carol").
- Components that read `ctx.verifiers` to figure out what value to write.
- Components that short-circuit the agent loop (construct the entire correct sequence of MCP calls in Python).
- Components that modify `agent/enterpriseops_agent.py` or any upstream file under `third_party/EnterpriseOps-Gym/`.
- Components that call langchain / direct API (use `ctx.chat()` if you need a sub-LLM).
- Components that fire on every task (`matcher=None`, priority=0) — that's effectively a system_prompt rewrite.
