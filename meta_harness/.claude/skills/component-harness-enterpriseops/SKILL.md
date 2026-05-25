---
name: component-harness-enterpriseops
description: Run ONE iteration of ServiceNow EnterpriseOps-Gym harness evolution by proposing ONE workflow graph patch — a single Python file declaring `COMPONENT: Component` plus a patch op (add_node / replace_node / disable_node) against the frontier workflow at `meta_harness/workflows/enterpriseops_<domain>.yaml`. The component is typed by mount + class + state_scope, gated by a class×mount×decision permission matrix, and verified by a Trust block (evidence_anchor + out_of_evidence_probe). Sibling of component-harness-sopbench / -gaia / -tau2; targets the upstream React loop over MCP gym servers with SQL-verifier judging on the final DB state.
---

# component-harness-enterpriseops

Run ONE iteration of agent evolution against one EnterpriseOps-Gym domain by proposing ONE **workflow graph patch**. A patch is `add_node` / `replace_node` / `disable_node`, applied to the frontier workflow at `meta_harness/workflows/enterpriseops_<domain>.yaml`. The node added or replaced is one **component** — a single Python file in `agent/components_enterpriseops_<domain>/<name>.py` exporting `COMPONENT: Component`. **You do NOT run benchmarks.** You analyse prior baselines + traces, write one component, choose one patch op, write `pending_eval.json`, and exit. The outer loop applies the patch, scores it on the domain's train subset, and admits or rejects.

## What EnterpriseOps-Gym tasks look like

A task = one row from HuggingFace `ServiceNow-AI/EnterpriseOps-Gym` split `<domain>` config `oracle` (or `plus_5_tools` / `plus_10_tools` / `plus_15_tools` for tool-retrieval stress modes). The agent receives:
  * `system_prompt` (str) — agent role + domain policy
  * `user_prompt` (str) — natural-language task instruction (e.g. "Create a calendar 'X', grant Carol edit access, schedule a 60-min Technical Kickoff on the first free day after Nov 14")
  * `selected_tools` (list[str]) — the oracle tool set (or oracle + N distractors)
  * `gym_servers_config` (list[dict]) — one or more dockerized MCP servers (`mcp_server_url`, `seed_database_file`, per-gym `context` headers)
  * `verifiers` (list[dict]) — opaque to the agent; SQL queries the judge runs over the final DB state, each with `name`, `validation_config.query`, `expected_value`, `comparison_type`

The agent runs upstream's multi-turn ReAct loop (`third_party/EnterpriseOps-Gym/orchestrators/react.py`) using LangChain + MCP tool dispatch. **Judging is outcome-based**: each task's verifiers are run as SQL queries on the gym DB after the loop ends; a task `PASS`es only if ALL verifiers return their expected value. There is no "predicted output" text to score — the model's final assistant message does not enter scoring.

Common failure modes (from baseline traces on `calendar` and `itsm`):
  * **Tool-arg shape error** — model passes `start: "2025-11-14T15:00"` when the tool expects `start_datetime` and a separate `timezone`; the tool 422s, model retries with wrong fix.
  * **Wrong tool selected** — model uses `get_event` to search by name when `list_events` + filter is required; gets `None`, gives up.
  * **Field-format mismatch downstream of policy** — model creates an ACL with `scope_email: "carol"` instead of the full email; the verifier SQL `WHERE scope_email = 'carol.white@techcorp.com'` returns 0.
  * **Missing intermediate step** — model creates the event but never calls `insert_acl_rule`; ACL verifier fails even though calendar/event ones pass.
  * **Date/timezone confusion** — model schedules at the right local time but wrong UTC (`start_timezone: 'America/New_York'` when verifier expects `'UTC'`).
  * **Loop exhaustion** — model hits `max_iterations` mid-task without finishing the sequence (long-horizon HR/CSM tasks).

Your job is to pick ONE such mechanism present in ≥3 train failures and add ONE component that addresses it.

## First principles

1. **The LLM is the last resort.** Each iteration, find one place the LLM is doing work deterministic code could do — tool-arg validation, parameter normalisation (email casing, datetime → UTC), required-field enforcement, retry on observed failure — and move it into Python.
2. **Decompose; don't defer.** A failure that looks "reasoning-bound" is rarely atomic. Split: what is the minimal judgment the LLM must make, and what computation / lookup / validation around it is fully deterministic? Build the deterministic part.
3. **Code earns its place by capturing stable structure, not by fitting recent failures.** Anything you encode must point at a fact OUTSIDE evidence — an MCP tool's JSON Schema, a timezone library, the gym server's `list_tools` response, an LLM API field (`finish_reason`). An IF/THEN induced from N failed train rows is a memorised map; it overfits.
4. **Per-domain specialization.** Components live in per-domain dirs and are namespaced by domain. Your work is for ONE domain only; do not optimize for cross-domain transfer.
5. **Verifiers are opaque.** You may NOT read the `verifiers` field to drive a component's logic — that is the test set's grading rubric. Components key off MCP tool schemas, user prompt structure, tool errors, finish_reason — never the SQL queries in `verifiers`.

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

### Mount enum (EnterpriseOps-Gym FC loop over MCP)

| mount               | when it fires                                                  | typical use                                                                |
|---------------------|----------------------------------------------------------------|----------------------------------------------------------------------------|
| `session_start`     | once per task, after MCP tools discovered, before any LLM call | static system-prompt injection (e.g. "always pass full email in ACL scope") |
| `pre_prompt_build`  | per task, after default system+user prompts built              | rewrite user prompt; inject domain policy reminder into system_prompt       |
| `pre_llm_turn`      | per turn, before each `llm_client.invoke_with_tools()`         | rewrite messages list (e.g., re-add omitted user_info context)              |
| `post_llm_response` | per turn, after the response                                   | rewrite assistant content; queue a retry hint via `inject_context`          |
| `pre_tool_use`      | before each MCP tool dispatch                                  | validate / rewrite tool args; block invalid calls (drop, don't error)       |
| `post_tool_use`     | after each MCP tool result, before adding ToolMessage          | reformat result; insert validation flag                                     |
| `pre_final_emit`    | after loop exit, AFTER upstream's SQL verifiers ran             | observational — verifier outcome already fixed; can rewrite final message    |
| `session_end`       | bookkeeping only                                                | —                                                                          |

**Per-turn vs. per-call**: `pre_llm_turn` and `post_llm_response` fire EACH turn (potentially many per task; avg ~9 in EnterpriseOps-Gym, max 34). `pre_tool_use` and `post_tool_use` fire EACH MCP tool call within a turn (a turn can have 0..N tool calls, max ~11). Be mindful of side-effects.

**Why `pre_final_emit` is weak for EnterpriseOps**: judging is outcome-based on final DB state, not on the agent's last text. Rewriting `ctx.final_output` does not change the verifier result — it only changes what downstream consumers see logged. Most useful components target `pre_tool_use` (deterministic arg fix-up), `post_llm_response` (catch and retry on schema/policy violations), and `session_start` (one-shot policy injection).

### ComponentClass enum

| class                 | the matcher tests…                                                                                          | risk                          |
|-----------------------|-------------------------------------------------------------------------------------------------------------|-------------------------------|
| `mechanism_layer`     | an MCP tool JSON Schema, a system field, an LLM API field, a general algorithm (date parsing, email regex)  | LOW                           |
| `reactive_guard`      | an observed failure event (tool 4xx/error, malformed args, empty content, missing required output)         | LOW                           |
| `channel`             | task structure — `user_info` fields, gym server names, user_prompt structure, an external lookup            | LOW                           |
| `induced_rule`        | a reading of domain policy text — IF/THEN compiled from N evidence rows                                      | HIGH (admitted ADVISORY-ONLY) |
| `predictive_heuristic`| raw prompt / response text — regex/keyword guess about what the model is about to do                       | REJECTED at load time         |

### Decision

| decision         | semantics (mount-dependent payload)                                                                             |
|------------------|-----------------------------------------------------------------------------------------------------------------|
| `allow`          | no-op                                                                                                           |
| `block`          | terminate task as blocked — EXCEPT at `pre_tool_use` where it skips just this tool call (drops it silently)      |
| `rewrite`        | replace the live payload at this mount (see semantics table below)                                              |
| `inject_context` | `session_start`/`pre_prompt_build` → append to `ctx.system_prompt`; `post_llm_response` → queue for next turn   |

REWRITE payload by mount:

| mount               | payload type                  | replaces                              |
|---------------------|-------------------------------|---------------------------------------|
| `pre_prompt_build`  | `str`                         | `ctx.user_prompt`                     |
| `pre_llm_turn`      | `list[BaseMessage]`           | `ctx.messages`                        |
| `post_llm_response` | `str`                         | `ctx.raw_response`                    |
| `pre_tool_use`      | `dict`                        | `ctx.current_tool_args`               |
| `post_tool_use`     | `str`                         | `ctx.current_tool_result_str`         |
| `pre_final_emit`    | `str` (or `None` → blocked)   | `ctx.final_output` (observational)    |

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

`none` / `read_file` / `http_get` / `llm_call` (sub-LLM for verifier/re-format/critic — discouraged for v1) / `tool_call` / `mutate_shared`.

## The class × mount × decision matrix

Load-time gate at `agent/component_runtime_enterpriseops/policy.py::ALLOWED`.

| class \ mount         | session_start | pre_prompt_build              | pre_llm_turn         | post_llm_response          | pre_tool_use         | post_tool_use   | pre_final_emit     |
|-----------------------|---------------|-------------------------------|----------------------|----------------------------|----------------------|-----------------|--------------------|
| `mechanism_layer`     | inject        | inject, rewrite, block        | rewrite, block       | rewrite, block, inject     | rewrite, block       | rewrite         | rewrite, block     |
| `reactive_guard`      | —             | —                             | —                    | rewrite, block, inject     | rewrite, block       | rewrite         | rewrite, block     |
| `channel`             | inject        | inject                        | —                    | —                          | —                    | —               | —                  |
| `induced_rule`        | —             | **inject (advisory)**         | —                    | —                          | —                    | —               | —                  |
| `predictive_heuristic`| rejected      | rejected                      | rejected             | rejected                   | rejected             | rejected        | rejected           |

A component whose (class, mount) is absent raises `ComponentPolicyError` at load. A handler that emits a non-admitted decision raises at fire time.

## The workflow graph

Per-domain frontier yaml: `meta_harness/workflows/enterpriseops_<domain>.yaml`

```yaml
nodes:
  - enterpriseops_calendar_iter1_acl_email_normalizer
edges: []
disabled: []
```

Plus a JSON snapshot `meta_harness/logs_components_enterpriseops_<domain>/frontier_workflow.json` written on accept.

### Patch ops

| op             | meaning                                                                                                              |
|----------------|----------------------------------------------------------------------------------------------------------------------|
| `add_node`     | append a new node; `agent/components_enterpriseops_<domain>/<id>.py` must be newly written                            |
| `replace_node` | keep the existing node id; overwrite the file with new behavior (same `COMPONENT.name`)                               |
| `disable_node` | add the id to `disabled:`; file remains for the durability audit                                                      |

## Hard rules

- Exactly ONE patch per invocation.
- **You do NOT run benchmarks.** No `run_enterpriseops_baseline.py`, no `evaluate.py`. The outer loop scores.
- **No task-specific code.** No domain-specific entity names in matchers (no calendar names, no user names, no UUIDs). No encoded gold answers.
- **You may NOT read the `verifiers` field to drive component logic.** Components target structure (tool schemas, user_prompt patterns, finish_reason) — never the grading rubric.
- **The target inference model is LOCKED via the upstream LLM config.** Components do NOT make additional LLM calls via langchain / direct API. No alternate model. No "second opinion" call.
- **`ctx.chat()` (Phase C+) is permitted for sub-LLM verifier / re-format patterns** — it goes through the SAME locked SUT model name (via `agent.llm.chat`, NOT the upstream langchain client) with mutable inference params: `ctx.chat(messages, max_tokens=8192, temperature=0.0, system_override=None, tools=None)`. The helper does NOT accept a `model=` kwarg. Declaring `capabilities=(Capability.LLM_CALL,)` is required.
- For `replace_node`, the **first shell action** MUST be:
  ```bash
  cp agent/components_enterpriseops_<domain>/<existing>.py agent/components_enterpriseops_<domain>/<existing>.py.bak_iter<N>
  ```
  The outer loop relies on the `.bak` to roll back on reject.
- READ-ONLY: `third_party/EnterpriseOps-Gym/`, `agent/enterpriseops_agent.py`, `agent/component_runtime_enterpriseops/`, `agent/llm.py`, `meta_harness/scripts/run_enterpriseops_baseline.py`, `meta_harness/scripts/enterpriseops_smoke.py`, `meta_harness/scripts/select_enterpriseops_split.py`, `meta_harness/scripts/recompute_eog_baseline.py`. The outer loop reads your output; you do not modify the loop.
- Component file naming convention: `agent/components_enterpriseops_<domain>/component_enterpriseops_<domain>_iter<N>_<slug>.py`. The `COMPONENT.name` should also include the domain slug (e.g. `enterpriseops_calendar_iter1_acl_email_normalizer`). Per-domain dirs ensure other domains' components are not loaded by your runtime; do NOT read or edit files under any other domain's dir.
- Domain context: you are evolving for exactly ONE domain. The `pending_eval.json` is read by the outer loop which knows the domain; you don't pick the domain.

## Component file interface

```python
# agent/components_enterpriseops_<domain>/component_enterpriseops_<domain>_iter<N>_<slug>.py
from __future__ import annotations

from agent.component_runtime_enterpriseops import (
    Capability, Component, ComponentClass, ComponentContext,
    Decision, Mount, StateScope, Trust,
)


def _matches(ctx: ComponentContext) -> bool:
    # Read ctx.raw_response / ctx.final_output / ctx.current_tool_args /
    # ctx.current_tool_result_str / ctx.finish_reason / ctx.shared /
    # ctx.user_info / ctx.tool_specs / ctx.current_tool_name.
    # NEVER read ctx.verifiers (= grading rubric).
    # NEVER read ctx.task_id (treat as opaque).
    ...


def _handler(ctx: ComponentContext) -> Decision:
    return Decision.rewrite(...)   # or inject_context / block / allow


COMPONENT = Component(
    name="enterpriseops_<domain>_<slug>",
    cls=ComponentClass.MECHANISM_LAYER,
    mount=Mount.PRE_TOOL_USE,
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
    "name": "candidate_iter<N>_<slug>",
    "hypothesis": "one-sentence claim of what the failure mode is",
    "changes": "one-sentence description of what your component does",
    "component": {
      "name": "<COMPONENT.name>",
      "cls": "mechanism_layer",
      "mount": "pre_tool_use",
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

For `disable_node`, omit `file` and the only `name` is the existing component's name. For `replace_node`, `name` reuses the existing `COMPONENT.name`; the `file` is the same path you overwrote (the runtime registry uses last-loaded-wins by name).

## How to investigate

1. **Read `frontier_val.json`'s `per_task`** at `meta_harness/logs_components_enterpriseops_<domain>/frontier_val.json` to find tasks the frontier scores 0 on. Each entry has `agent`, `score` (0/1, all-verifiers-pass), `verifier_pass_rate` (partial credit, useful for picking "almost-correct" failures).
2. **Read baseline results** at `meta_harness/logs_components_enterpriseops_<domain>/baseline/<ts>/{train,test}/results/run_1/results_*.json`. Each file has `runs[0]` with `conversation_flow`, `tools_used`, `tool_results`, `verification_results` (per-verifier name → {passed, error, details}), `verification_summary` (total / passed / pass_rate), `overall_success`.
3. **Look for ≥3 failures sharing the same mechanism**. Examples (calendar baseline observed):
   - All 3 failures call `insert_acl_rule` with `scope_email: "carol"` but verifier checks `scope_email = 'carol.white@techcorp.com'` → `pre_tool_use` MECHANISM_LAYER that, when scope_email is a bare local-part, looks up the full email in `ctx.user_prompt` + `ctx.user_info` and rewrites.
   - All 3 failures emit `create_event` with `start_datetime: "2025-11-14T15:00"` and no timezone, verifier expects `start_timezone='UTC'` → `pre_tool_use` REWRITE that normalises any naive datetime to UTC and fills `start_timezone`.
   - All 3 failures' assistant content shows a SQL-style WHERE clause in the prose ("WHERE name = ...") then no tool call → `post_llm_response` REACTIVE_GUARD that inject_context's "use the list_X tool with a filter argument; do not write SQL".
4. **Form ONE hypothesis** and tie it to a stable structure (the MCP tool's JSON Schema, an ISO 8601 invariant, an RFC 5322 email regex, the gym server's discovered tool list). State the structure in `trust.evidence_anchor`.
5. **Write ONE component** at the appropriate mount with the smallest possible matcher/handler. Resist embedding specific entity names (no `"carol.white@techcorp.com"` literal — look it up from user_info / user_prompt or the gym server's user list at fire time).
6. **Validate registration**:
   ```bash
   python -c "
   from agent.component_runtime_enterpriseops import load_components_from_dir
   from pathlib import Path
   comps = load_components_from_dir(
       Path('agent/components_enterpriseops_<domain>'),
       only=['<COMPONENT.name>'],
   )
   assert any(c.name == '<COMPONENT.name>' for cs in comps.values() for c in cs)
   print('ok')
   "
   ```
7. **Write `pending_eval.json`** at `meta_harness/logs_components_enterpriseops_<domain>/pending_eval.json` and print `CANDIDATE: <name>`.

## Event runtime additions (Phase D)

The mount-based dispatch above remains the canonical entry — every existing
component fires via `listens = mount.value` by default (auto-set in
`Component.__post_init__`). Phase D adds a parallel **Tier-1 event** namespace
that surfaces finer per-turn / per-tool failure-mode signals the runtime sees.

### Tier-1 events emitted by `EnterpriseOpsReactOrchestrator.execute()` + `run_task`

All 15 Tier-1 events fire in enterpriseops:

| event                       | when it fires                                                         | mount alignment            |
|-----------------------------|-----------------------------------------------------------------------|----------------------------|
| `task_received`             | right after `ComponentContext` is built (top of `run_task`)            | (new — lifecycle anchor)   |
| `session_start`             | once per task, after MCP tools discovered                              | = `Mount.SESSION_START`    |
| `pre_prompt_build`          | per task, default prompts built                                        | = `Mount.PRE_PROMPT_BUILD` |
| `pre_context_build`         | paired with `pre_prompt_build` (cross-sibling alias)                   | aliases `pre_prompt_build` |
| `pre_agent_construct`       | last hook before `BenchmarkConfig` is sealed                           | (new)                      |
| `pre_llm_turn`              | per turn, before `llm_client.invoke_with_tools()`                      | = `Mount.PRE_LLM_TURN`     |
| `pre_llm_request`           | paired with `pre_llm_turn` (cross-sibling alias)                       | (new alias)                |
| `post_llm_response`         | per turn, after LangChain response                                     | = `Mount.POST_LLM_RESPONSE`|
| `post_llm_response_raw`     | paired with `post_llm_response`                                        | aliases `post_llm_response`|
| `on_length_truncation`      | **synthesised** when `finish_reason == "length"`                       | (new)                      |
| `on_empty_response`         | **synthesised** when `raw_response.strip() == ""`                      | (new)                      |
| `on_no_tool_call_emitted`   | **synthesised** when assistant emits no tool_calls                     | (new)                      |
| `pre_tool_arg_validation`   | per MCP tool call, **before** `pre_tool_use`                           | (new)                      |
| `pre_tool_use`              | per MCP tool call, main pre-tool decision                              | = `Mount.PRE_TOOL_USE`     |
| `post_tool_use`             | per MCP tool call, after invocation                                    | = `Mount.POST_TOOL_USE`    |
| `post_tool_result_raw`      | per MCP tool call, raw result anchor                                   | (new)                      |
| `on_tool_error`             | **synthesised** when `current_tool_success == False`                   | (new)                      |
| `on_explicit_terminate`     | **synthesised** when assistant emits final answer w/o tool calls       | (new)                      |
| `pre_final_emit`            | after loop exit, AFTER upstream SQL verifiers ran (observational)      | = `Mount.PRE_FINAL_EMIT`   |
| `session_end`               | bookkeeping                                                            | = `Mount.SESSION_END`      |

### `Component.listens` and `emits` (Phase B)

```python
COMPONENT = Component(
    ...,
    listens="on_tool_error",          # default = mount.value; existing
                                       # mount-only components migrate transparently.
    emits=("iter12_acl_email_resolved",),  # self-doc of custom events the
                                       # component raises via ctx.emit(...).
)
```

Pick `listens="<tier-1 name>"` for narrow failure-mode targeting (e.g.
`on_tool_error` rather than `post_tool_use` + matcher). The `mount` field is
still required for policy validation — set it to the closest Mount enum.

### `ctx.chat()` / `ctx.emit()` / `ctx.emit_upstream()` (Phase C)

`ComponentContext` now inherits from `EventContext`:

| method                                | semantics                                                                                          |
|---------------------------------------|----------------------------------------------------------------------------------------------------|
| `ctx.chat(messages, *, max_tokens=8192, temperature=0.0, system_override=None, tools=None)` | Sub-LLM call bound to the locked SUT model name via `agent.llm.chat`. No `model=` kwarg.            |
| `ctx.emit(custom_event_name, **fields)` | Synchronously fire a Tier-2/3 custom event. Depth cap = 10. Fields dropped in v1; use `ctx.shared` / `ctx.upstream`. |
| `ctx.emit_upstream(key, value)`       | `ctx.upstream[key] = value` sugar.                                                                  |
| `ctx.fetch` / `ctx.read_file`         | None-wired in enterpriseops v1.                                                                     |

Declare `capabilities=(Capability.LLM_CALL,)` for components that use `ctx.chat()`.

### Custom events (Tier 2/3)

```
iter<N>_<slug>_<event>        e.g. iter9_acl_normalize_resolved_email
on_<thing>                    failure-mode style
```

Always declare `emits=(...)` on the publisher; grep
`agent/components_enterpriseops_<domain>/*.py` for taken names. To admit
non-ALLOW decisions on a custom event, add a string-key entry to
`ALLOWED[ComponentClass.X]` in
`agent/component_runtime_enterpriseops/policy.py`.

### Event-based patterns

| pattern                                              | class           | listens                    | decision                                    |
|------------------------------------------------------|-----------------|----------------------------|---------------------------------------------|
| schema-validate MCP tool args narrowly               | mechanism_layer | `pre_tool_arg_validation`  | rewrite (normalise email casing, fill UTC)  |
| retry-hint after MCP 4xx                              | reactive_guard  | `on_tool_error`            | inject_context (queued for next turn)       |
| sub-LLM verify final answer before SQL judging       | mechanism_layer | `on_explicit_terminate`    | inject_context (retry) / block (rare)        |
| publisher/subscriber dataflow within one task        | mechanism_layer | (A emits `iter<N>_<slug>_X`) → (B `listens="iter<N>_<slug>_X"`) | inject_context (consume `ctx.upstream`)      |

## What NOT to write

- Components that memorise train-set answers (encode `scope_email = "carol.white@techcorp.com"` for any task whose user_prompt contains "Carol").
- Components that read `ctx.verifiers` to figure out what value to write.
- Components that try to short-circuit the agent loop (e.g. construct the entire correct sequence of MCP calls in Python without the LLM).
- Components that modify `agent/enterpriseops_agent.py` or any upstream file under `third_party/EnterpriseOps-Gym/`.
- Components that call an LLM (no `chat()`, no langchain, no direct API).
- Components that fire on every task (priority=0 with always-True matcher) — that's effectively a system_prompt rewrite, not a component.
