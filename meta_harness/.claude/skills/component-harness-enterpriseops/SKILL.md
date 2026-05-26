---
name: component-harness-enterpriseops
description: Propose ONE hook (a single Python file declaring `COMPONENT: Component`) that stabilizes the EnterpriseOps-Gym main agent on a recurring failure mode for ONE domain, plus a workflow patch (add / replace / disable) against `meta_harness/workflows/enterpriseops_<domain>.yaml`. The main agent runs the upstream React loop over dockerized MCP gym servers (LangChain + MCP tool dispatch); judging is SQL-verifier on the final DB state. Hooks subscribe to lifecycle events via `listens=` and react via a Decision (allow / block / rewrite / inject_context). Admission gated by a class×event×decision matrix and a Trust block (evidence_anchor + out_of_evidence_probe).
---

# component-harness-enterpriseops

The EnterpriseOps-Gym main agent is the protagonist. For each domain (`calendar`, `itsm`, `hybrid`, …) it runs upstream's React loop (`third_party/EnterpriseOps-Gym/orchestrators/react.py`) over dockerized MCP gym servers using LangChain + MCP tool dispatch, with the locked SUT model. Judging is **outcome-based**: each task's verifiers run as SQL queries on the gym DB after the loop ends; a task passes only if ALL verifiers match — there is no "predicted output" text to score. **Your job is to stabilize it**, not replace it. You propose ONE hook — a single Python file under `agent/components_enterpriseops_<domain>/<name>.py` exporting `COMPONENT: Component` — plus a workflow patch (`add` / `replace` / `disable`). **You do NOT run benchmarks.** The outer loop applies the patch, scores on the domain's train subset, and admits or rejects.

## What EnterpriseOps-Gym tasks look like

A task = one row from HuggingFace `ServiceNow-AI/EnterpriseOps-Gym` split `<domain>` config `oracle` (or `plus_5_tools` / `plus_10_tools` / `plus_15_tools` for tool-retrieval stress). The agent receives:
  * `system_prompt` (str) — agent role + domain policy
  * `user_prompt` (str) — NL task instruction
  * `selected_tools` (list[str]) — oracle tool set (or oracle + N distractors)
  * `gym_servers_config` (list[dict]) — one or more dockerized MCP servers
  * `verifiers` (list[dict]) — **OPAQUE to the agent**: SQL queries the judge runs over the final DB state. You may NOT read `ctx.verifiers` from a hook.

Common failure modes (calendar / itsm baselines):
  * **Tool-arg shape error** — `start: "2025-11-14T15:00"` vs expected `start_datetime` + separate `timezone`.
  * **Wrong tool selected** — `get_event` to search by name when `list_events` + filter is required.
  * **Field-format mismatch** — ACL with `scope_email: "carol"` vs verifier expecting `"carol.white@techcorp.com"`.
  * **Missing intermediate step** — event created but `insert_acl_rule` never called.
  * **Date/timezone confusion** — `start_timezone: 'America/New_York'` when verifier expects `'UTC'`.
  * **Loop exhaustion** — hits `max_iterations` mid-task.

## First principles

0. **Main agent is the protagonist.** Hooks stabilize MCP-tool arg shapes / retries / one-shot policy nudges; they don't reconstruct the action sequence in Python. Before writing a hook, ask: "would the main agent still be the same agent without this — just less prone to failing on X?" If "no, it'd be doing fundamentally different work" (e.g. you've built the entire correct MCP-call sequence in the handler), the hook is too heavy.
1. **Capability vs Stabilization — keep them separate.** If a domain is failing because the agent literally can't reach a needed tool / server, that's a capability gap — file it as an MCP server / tool registration, not a hook that injects the would-be tool result.
2. **The LLM is the last resort within stabilization.** Move tool-arg validation, parameter normalisation (email casing, datetime→UTC), required-field enforcement, retry-on-failure into Python.
3. **Code earns its place by capturing stable structure, not by fitting recent failures.** Anchor on an MCP tool's JSON Schema, a timezone library, the gym server's `list_tools` response, an LLM API field, RFC 5322. IF/THEN induced from N failed train rows is a memorised map; it overfits.
4. **Per-domain specialization.** Components live in per-domain dirs (`agent/components_enterpriseops_<domain>/`) and are namespaced by domain. Work for ONE domain only.
5. **Verifiers are opaque.** Hooks key off MCP tool schemas, user_prompt structure, tool errors, `finish_reason` — NEVER the SQL queries in `verifiers`.

## Toolbox (pick what fits — none of these are exclusive)

| When you'd reach for it | What you change |
|---|---|
| Main agent can't reach a tool / server it needs | Register a new MCP server or tool (Capability expansion — separate workflow, **not this skill**) |
| Main agent calls an MCP tool with the wrong arg shape | Write a `pre_tool_arg_validation` `mechanism_layer` (this skill) |
| Tool returns error and the LLM loops on identical args | Write an `on_tool_error` `reactive_guard` (this skill) |
| Agent ends loop without making the DB write the verifier checks | Write an `on_explicit_terminate` `reactive_guard` (this skill — refuses termination) |
| Static framework nudge ("always pass full email in ACL scope") | Write a `mechanism_layer` on `session_start` (this skill) |
| Multiple hooks need to coordinate | Custom event: A `ctx.emit("iter<N>_<slug>_X")`, B `listens="iter<N>_<slug>_X"` |

## The component model

```python
@dataclass(frozen=True, kw_only=True)
class Component:
    name: str                              # stable id; reuse for `replace`
    cls: ComponentClass                    # mechanism_layer | reactive_guard | induced_rule
    listens: str                           # event name the dispatcher routes on
    matcher: Optional[Callable[[Ctx], bool]]
    handler: Callable[[Ctx], Decision]
    trust: Trust                           # required verification block
    priority: int = 100                    # smaller fires first within an event bucket
    emits: tuple[str, ...] = ()            # self-doc of custom Tier-2/3 events this raises
```

### Lifecycle events the EnterpriseOps runtime emits

Per-task setup:

| event | when it fires | typical use |
|---|---|---|
| `task_received` | top of `run_task` | lifecycle anchor |
| `session_start` | once per task, after MCP tools discovered | "always pass full email in ACL scope" style injection |
| `pre_prompt_build` | per task; default system+user prompts built | rewrite user prompt; inject domain policy reminder |
| `pre_context_build` | alias of `pre_prompt_build` | same as above |
| `pre_agent_construct` | last hook before BenchmarkConfig sealed | inference-hint injection |

Per LLM turn (avg ~9 turns / task, max 34):

| event | when it fires | typical use |
|---|---|---|
| `pre_llm_turn` | per turn, before `llm_client.invoke_with_tools()` | rewrite messages list (re-add user_info context) |
| `pre_llm_request` | per turn (alias for `pre_llm_turn`) | sub-LLM verifier prep |
| `post_llm_response` / `post_llm_response_raw` | per turn, after LangChain response | rewrite assistant content; queue retry hint |
| `on_length_truncation` | **synthesised** when `finish_reason=="length"` | sub-LLM recovery |
| `on_empty_response` | **synthesised** when raw_response empty | reactive retry |
| `on_no_tool_call_emitted` | **synthesised** when no tool_calls | retry-with-tool reminder |

Per MCP tool call (max ~11 / turn):

| event | when it fires | typical use |
|---|---|---|
| `pre_tool_arg_validation` | narrow schema-check phase before `pre_tool_use` | rewrite args (normalise email casing, fill UTC) / block |
| `pre_tool_use` | main pre-tool decision | validate / rewrite args; block invalid calls |
| `post_tool_use` | after invocation | reformat result; insert validation flag |
| `post_tool_result_raw` | raw result anchor | observe / inject |
| `on_tool_error` | **synthesised** when `current_tool_success==False` | retry-hint via `inject_context` |

Per-task exit:

| event | when it fires | typical use |
|---|---|---|
| `on_explicit_terminate` | **synthesised** when assistant emits final answer w/o tool calls | exit-gate; may BLOCK to refuse termination |
| `pre_final_emit` | after loop exit, AFTER upstream SQL verifiers ran (observational) | observational only — verifier outcome is fixed by now |
| `session_end` | bookkeeping | — |

**Why `pre_final_emit` is weak**: judging is outcome-based on the final DB state, not on the agent's last text. Rewriting `ctx.final_output` does not change the verifier result — it only changes what downstream consumers see logged. Useful hooks target `pre_tool_arg_validation` / `pre_tool_use` (deterministic arg fix-up), `post_llm_response` / `on_*` (catch and retry), and `session_start` / `pre_prompt_build` (one-shot policy injection), plus `on_explicit_terminate` for artifact gates.

Helpers: `matcher_for_tool("create_event")` and `matcher_for_server("calendar")` from `agent.component_runtime_enterpriseops.types` scope matchers cleanly to one tool or one gym server (useful in multi-gym hybrid tasks).

### Component classes

| class | what the matcher tests | risk |
|---|---|---|
| `mechanism_layer` | MCP tool JSON Schema / system field / LLM API field / general algorithm (timezone normalize, RFC 5322 email) | LOW |
| `reactive_guard` | observed failure event (tool error string, malformed args, empty content) | LOW |
| `induced_rule` | reading of domain policy text — IF/THEN compiled from N evidence rows | HIGH (advisory `inject_context` only) |
| `predictive_heuristic` | raw prompt / response text via regex / keywords | REJECTED at load time |

### Decision

| decision | semantics |
|---|---|
| `allow` | no-op |
| `block` | terminate task as blocked — EXCEPT at `pre_tool_use` / `pre_tool_arg_validation` where it skips just this tool call |
| `rewrite` | replace the live payload at this event (see Event → payload below) |
| `inject_context` | setup events → append to `ctx.system_prompt`; per-turn / per-tool events → queue for next turn |

Event → REWRITE payload:

| event | payload type | replaces |
|---|---|---|
| `pre_prompt_build` / `pre_context_build` | `str` | `ctx.user_prompt` |
| `pre_llm_turn` | `list[BaseMessage]` | `ctx.messages` |
| `post_llm_response[_raw]` / `on_length_truncation` / `on_empty_response` / `on_no_tool_call_emitted` | `str` | `ctx.raw_response` |
| `pre_tool_use` / `pre_tool_arg_validation` | `dict` | `ctx.current_tool_args` |
| `post_tool_use` / `post_tool_result_raw` / `on_tool_error` | `str` | `ctx.current_tool_result_str` |
| `pre_final_emit` | `str` or `None` | `ctx.final_output` (observational) |

### Trust

Required: `evidence_anchor` / `blast_radius` (local|workflow|global) / `rollback_when`. `out_of_evidence_probe` required for `induced_rule`. Optional: `fallback`.

- `evidence_anchor`: name a stable structure OUTSIDE evidence — an MCP tool JSON Schema field, RFC 5322, ISO 8601, the gym server's `list_tools` response, an OpenAI/DeepSeek API field. If your anchor is "I observed row_017/023/041 all do X", you're anchored INSIDE evidence — pick a different class or don't write the hook.
- `out_of_evidence_probe` (induced_rule only): name one concrete case NOT in evidence rows where the matcher fires and what handler returns on it.

### Handler helpers on `ctx`

| helper | what it does |
|---|---|
| `ctx.chat(messages, max_tokens=..., temperature=..., system_override=..., tools=...)` | sub-LLM via the locked SUT model (NOT upstream LangChain) |
| `ctx.shared` | per-task dict, free to read/write |
| `ctx.state[component_name]` | per-task per-component scratchpad |
| `ctx.emit("custom_event_name", **fields)` | fire custom event; re-enters dispatcher (depth cap 10) |
| `ctx.emit_upstream(key, value)` | shorthand for `ctx.upstream[key] = value` |

Per-event fields: `ctx.benchmark` (domain slug), `ctx.user_info` (`{user_id, name, email, timezone}`), `ctx.gym_servers`, `ctx.tool_specs` (each `{name, description, input_schema, _mcp_server_name, ...}`), `ctx.selected_tools`, `ctx.extras`, `ctx.system_prompt`, `ctx.user_prompt`, `ctx.messages`, `ctx.raw_response`, `ctx.tool_calls`, `ctx.turn_index`, `ctx.finish_reason`, `ctx.current_tool_name / _args / _result / _result_str / _call_id / _success / _error / _server`, `ctx.final_output`. **NEVER read `ctx.verifiers`** (grading rubric). NEVER read `ctx.task_id`.

## The class × event × decision matrix

Load-time gate at `agent/component_runtime_enterpriseops/policy.py::ALLOWED`.

Setup events: `mechanism_layer` admits `inject_context` (+ `rewrite` / `block` on `pre_prompt_build` / `pre_context_build`); `induced_rule` admits `inject_context` (ADVISORY) on `pre_prompt_build` / `pre_context_build`; `reactive_guard` does not fire.

Per-turn / per-tool events: `mechanism_layer` and `reactive_guard` admit `rewrite` / `block` / `inject_context` per the cell (see policy.py).

Exit events: `mechanism_layer` and `reactive_guard` admit `rewrite` + `block` on `pre_final_emit`, `block` on `on_explicit_terminate`, `allow` on `session_end`.

`predictive_heuristic` is rejected at every event.

## The workflow file

Per-domain frontier YAML: `meta_harness/workflows/enterpriseops_<domain>.yaml`.

```yaml
nodes: []
disabled: []
```

Plus a JSON snapshot `meta_harness/logs_components_enterpriseops_<domain>/frontier_workflow.json`.

### Patch ops

| op | meaning |
|---|---|
| `add` | append a new node; `agent/components_enterpriseops_<domain>/<id>.py` must be newly written |
| `replace` | keep the existing id; overwrite the file (same `COMPONENT.name`) |
| `disable` | add the id to `disabled:`; file remains for durability audit |

For `replace`, the **first shell action MUST be** `cp agent/components_enterpriseops_<domain>/<existing>.py agent/components_enterpriseops_<domain>/<existing>.py.bak_iter<N>`.

## Hard rules

- Exactly ONE patch per invocation.
- **You do NOT run benchmarks.** No `run_enterpriseops_baseline.py`, no `evaluate.py`. The outer loop scores.
- **No task-specific code.** No domain-specific entity literals (no calendar names, user names, UUIDs). No encoded gold answers.
- **You may NOT read `ctx.verifiers`** to drive hook logic — that is the test set's grading rubric. Hooks target structure (tool schemas, user_prompt patterns, `finish_reason`) — never the SQL queries.
- **The target inference model is LOCKED via the upstream LLM config.** Hooks do NOT call langchain / direct API for additional model calls. `ctx.chat()` IS permitted (goes through `agent.llm.chat`, same locked SUT model name).
- **Capability gaps are not for hooks.** If a domain genuinely needs a new MCP server / tool, file a separate request — do not invent a hook that injects the would-be tool result.
- For `replace`, the **first shell action** MUST be the `cp ... .bak_iter<N>` command above.
- READ-ONLY: `third_party/EnterpriseOps-Gym/`, `agent/enterpriseops_agent.py`, `agent/component_runtime_enterpriseops/`, `agent/llm.py`, all other domains' `agent/components_enterpriseops_*/`, `meta_harness/scripts/run_enterpriseops_baseline.py`, `meta_harness/scripts/enterpriseops_smoke.py`, `meta_harness/scripts/select_enterpriseops_split.py`.
- Component file naming: `agent/components_enterpriseops_<domain>/component_enterpriseops_<domain>_iter<N>_<slug>.py`. The `COMPONENT.name` SHOULD include the domain slug.

## Component file template

```python
# agent/components_enterpriseops_<domain>/component_enterpriseops_<domain>_iter<N>_<slug>.py
from __future__ import annotations

from agent.component_runtime_enterpriseops import (
    Component, ComponentClass, ComponentContext, Decision, Trust,
)


def _matches(ctx: ComponentContext) -> bool:
    # Read ctx.raw_response / ctx.final_output / ctx.current_tool_name /
    # ctx.current_tool_args / ctx.current_tool_result_str / ctx.finish_reason /
    # ctx.user_info / ctx.tool_specs / ctx.current_tool_server / ctx.shared.
    # ctx.event names the firing event for branchable handlers.
    # NEVER read ctx.verifiers (= grading rubric).
    # NEVER read ctx.task_id.
    ...


def _handler(ctx: ComponentContext) -> Decision:
    return Decision.rewrite(...)        # or inject_context / block / allow


COMPONENT = Component(
    name="enterpriseops_<domain>_<slug>",
    cls=ComponentClass.MECHANISM_LAYER,
    listens="pre_tool_arg_validation",
    matcher=_matches,
    handler=_handler,
    priority=100,
    emits=(),
    trust=Trust(
        evidence_anchor="<MCP tool JSON Schema / RFC 5322 / ISO 8601 / API field>",
        blast_radius="local",
        rollback_when="<observable rollback signal>",
        out_of_evidence_probe="<required for induced_rule>",
        fallback="<matcher-false / handler-allow semantics>",
    ),
)
```

## `pending_eval.json` schema

```json
{
  "iteration": <N>,
  "candidate": {
    "name": "candidate_iter<N>_<slug>",
    "hypothesis": "<one-sentence falsifiable claim of the failure mode>",
    "mechanism": "<failure mode this targets>",
    "evidence_task_ids": ["<rid1>", "<rid2>", "<rid3>"],
    "changes": "<plain-English diff vs prior frontier>",
    "expected_delta": "<expected pass-rate change on train>",
    "component": {
      "id": "<COMPONENT.name>",
      "cls": "mechanism_layer | reactive_guard | induced_rule",
      "listens": "<event_name>",
      "file": "agent/components_enterpriseops_<domain>/component_enterpriseops_<domain>_iter<N>_<slug>.py",
      "trust": {
        "evidence_anchor": "...",
        "blast_radius": "local | workflow | global",
        "rollback_when": "...",
        "out_of_evidence_probe": "<required for induced_rule>",
        "fallback": "..."
      }
    },
    "workflow_patch": {
      "op": "add | replace | disable",
      "name": "<COMPONENT.name; existing id for disable>",
      "file": "agent/components_enterpriseops_<domain>/component_enterpriseops_<domain>_iter<N>_<slug>.py"
    }
  }
}
```

For `disable`, omit `component` and `workflow_patch.file`; only `workflow_patch.name` matters.

## How to investigate

1. **Read `frontier_val.json`'s `per_task`** at `meta_harness/logs_components_enterpriseops_<domain>/frontier_val.json` to find tasks the frontier scores 0 on. Each entry has `agent`, `score` (0/1), `verifier_pass_rate` (partial credit; useful for picking "almost-correct" failures).
2. **Read baseline results** at `meta_harness/logs_components_enterpriseops_<domain>/baseline/<ts>/{train,test}/results/run_1/results_*.json`. Each file has `runs[0]` with `conversation_flow`, `tools_used`, `tool_results`, `verification_results` (per-verifier name → `{passed, error, details}`), `verification_summary`, `overall_success`.
3. **Look for ≥3 failures sharing the same mechanism.** Examples:
   - 3 failures call `insert_acl_rule` with `scope_email: "carol"` but verifier checks `scope_email = 'carol.white@techcorp.com'` → `pre_tool_arg_validation` MECHANISM_LAYER that looks up the full email in `ctx.user_prompt` / `ctx.user_info` and rewrites. Anchor: RFC 5322 + the tool's `scope_email` JSON Schema field.
   - 3 failures emit `create_event` with `start_datetime: "2025-11-14T15:00"` and no timezone → `pre_tool_arg_validation` REWRITE that normalises naive datetimes to UTC and fills `start_timezone`. Anchor: ISO 8601 + the tool's schema.
   - 3 failures' assistant content shows a SQL-style WHERE clause → `post_llm_response` REACTIVE_GUARD that `inject_context`s "use the list_X tool with a filter argument".
4. **Form ONE hypothesis** tied to a stable structure. State it in `trust.evidence_anchor`.
5. **Write ONE hook** with the smallest possible matcher/handler. Resist embedding entity literals (no `"carol.white@techcorp.com"` literal — look it up from `ctx.user_info` / `ctx.user_prompt` at fire time).
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

## Common stabilization patterns

| pattern | class | listens | decision |
|---|---|---|---|
| inject "always pass full email in ACL scope" | mechanism_layer | `session_start` | inject_context |
| inject domain policy reminder | mechanism_layer | `pre_prompt_build` | inject_context |
| advisory: policy paragraph paraphrase | induced_rule | `pre_context_build` | inject_context |
| normalise bare-local-part email to full address (RFC 5322) | mechanism_layer | `pre_tool_arg_validation` | rewrite |
| fill missing `start_timezone='UTC'` on naive datetime (ISO 8601) | mechanism_layer | `pre_tool_arg_validation` | rewrite |
| block tool call with missing required param | reactive_guard | `pre_tool_use` | block (skips this tool call only) |
| retry hint on MCP 4xx | reactive_guard | `on_tool_error` | inject_context |
| length-recovery via sub-LLM with bigger budget | reactive_guard | `on_length_truncation` | rewrite via `ctx.chat(max_tokens=32768)` |
| "use list_X with filter, don't write SQL" reminder | reactive_guard | `post_llm_response` | inject_context |
| refuse termination without expected DB write | reactive_guard | `on_explicit_terminate` | block |

## Custom events (Tier 2/3)

```
iter<N>_<slug>_<event>        e.g. iter9_acl_normalize_resolved_email
on_<thing>                    failure-mode style
```

Declare `emits=(...)` on the publisher; subscribers declare `listens="iter<N>_<slug>_<event>"`. To admit non-`allow` decisions on a custom event, add a string-key entry to `ALLOWED[ComponentClass.X]` in `agent/component_runtime_enterpriseops/policy.py`.

## What this skill does NOT do

- Run benchmarks (the outer loop does).
- Register new MCP servers / tools (capability expansion — a separate workflow).
- Modify `agent/enterpriseops_agent.py`, `agent/component_runtime_enterpriseops/`, `third_party/EnterpriseOps-Gym/`, or other domains' component dirs.
- Memorise train-set answers (encode `scope_email = "carol.white@techcorp.com"` for any task whose user_prompt mentions "Carol").
- Read `ctx.verifiers` to figure out what value to write.
- Short-circuit the agent loop (construct the entire correct sequence of MCP calls in Python).
- Call langchain / direct API for sub-LLM (use `ctx.chat()` instead).
- Fire on every task (`matcher=None`, priority=0) — that's effectively a system_prompt rewrite.
- Loop or propose multiple patches in one invocation.
- Encode a domain policy interpretation as `mechanism_layer` override (`rewrite` / `block`). Route to `induced_rule` + `inject_context` only.
- Encode a prompt-shape regex as `predictive_heuristic` — rejected at load.
