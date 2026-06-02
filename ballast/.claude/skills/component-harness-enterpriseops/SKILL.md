---
name: component-harness-enterpriseops
description: Propose ONE iteration on the EnterpriseOps-Gym wrapper agent for ONE domain (calendar | itsm). The frontier IS a directory (`agent/enterpriseops/v_<domain>_<N>/`); the outer loop clones it for you and you edit anything inside. Two paths are first-class: capability (edit agent.py — SYSTEM_PROMPT, tool registration, orchestrator) and stabilization (add/edit/delete files in components_<domain>/). The main agent runs upstream's React loop over dockerized MCP gym servers; judging is SQL-verifier on the final DB state. Components subscribe to lifecycle events via `listens=` and react via a Decision (allow / block / rewrite / inject_context); the class×event×decision matrix and a Trust block (evidence_anchor + out_of_evidence_probe) gate them.
---

# component-harness-enterpriseops

## What you propose

The outer loop has cloned the prior frontier directory into a fresh
`agent/enterpriseops/v_<domain>_<N>/` for you. **That directory is your
candidate.** If accepted, the symlink `agent/enterpriseops/current_<domain>`
moves to point at it; if rejected, the entire directory is deleted.

You may edit ANY file inside the candidate dir. Two paths are first-class:

| Path | What you edit | When |
|---|---|---|
| **Capability** | `agent.py` (and/or new files alongside it) — SYSTEM_PROMPT, tool registration in `available_tools`, orchestrator subclass, max_iterations, retry policy | The agent is missing knowledge / a tool / a structural ability the task class genuinely needs |
| **Stabilization** | `components_<domain>/*.py` — add a hook with a `Component` declaration | The agent already has the capability but a recurring small structural error costs verifier passes |

A single iter MAY combine both if the hypothesis requires it. **You still
propose ONE coherent change** — one hypothesis, one trust block — even if
that change touches multiple files.

You do NOT run benchmarks. The outer loop scores on the domain's train
subset and admits or rejects.

## What EnterpriseOps-Gym tasks look like

A task = one row from HuggingFace `ServiceNow-AI/EnterpriseOps-Gym` split
`<domain>` config `oracle` (or `plus_N_tools`). The agent receives:
  * `system_prompt` (str) — agent role + domain policy
  * `user_prompt` (str) — NL task instruction
  * `selected_tools` (list[str]) — oracle tool set (or oracle + N distractors)
  * `gym_servers_config` (list[dict]) — dockerized MCP server(s)
  * `verifiers` (list[dict]) — **OPAQUE**: SQL queries the judge runs over the
    final DB state. You may NOT read `ctx.verifiers` from a component.

Common failure modes (calendar / itsm baselines):
  * **Tool-arg shape error** — `start: "2025-11-14T15:00"` vs expected `start_datetime` + separate `timezone`.
  * **Wrong tool selected** — `update_calendar` vs `update_calendar_in_list` (different resources).
  * **Field-format mismatch** — `scope_email: "carol"` vs verifier expecting `"carol.white@techcorp.com"`.
  * **Missing API field** — `visibility='private'` when verifier expects `'confidential'`.
  * **Date/timezone confusion** — `start_timezone='UTC'` when verifier expects `'Asia/Singapore'`.
  * **Loop exhaustion** — hits `max_iterations` mid-task.

## First principles

0. **Main agent is the protagonist.** Hooks stabilize MCP-tool arg shapes /
   retries / one-shot policy nudges; capability edits extend what the agent
   can reach. Neither should reconstruct the action sequence in Python.

1. **Capability vs Stabilization — keep them separate at the file level.**
   Edits to `agent.py` are capability moves. Files in `components_<domain>/`
   are stabilization. Don't conflate (e.g. don't write a giant component
   that injects an API cheatsheet on every task — put that into `SYSTEM_PROMPT`
   in agent.py instead).

2. **Code earns its place by capturing stable structure, not by fitting
   recent failures.** Anchor on a tool's JSON Schema, RFC 5322, ISO 8601,
   the gym server's `list_tools` response, a documented API field. "I
   observed N failures all do X" is anchoring inside evidence.

3. **The LLM is the last resort within stabilization.** Move tool-arg
   normalisation, datetime→UTC fills, required-field enforcement, retry-on-failure
   into deterministic Python.

4. **Per-domain specialization.** components live in domain-scoped
   `components_<domain>/`. A calendar iter does not touch `components_itsm/`
   and vice versa. The agent.py CAN have generic edits that benefit both
   domains (e.g. a tool-schema validator), but be honest about it in trust.

5. **Verifiers are opaque.** Components key off MCP tool schemas, user_prompt
   structure, tool errors, `finish_reason` — NEVER the SQL queries in `verifiers`.

## Working directory layout

The outer loop tells you the candidate dir. It looks like:

```
agent/enterpriseops/v_<domain>_<N>/
    agent.py                          # the wrapper agent — SYSTEM is fine to edit
    runtime/                          # dispatcher / policy / types
        __init__.py
        base.py
        policy.py                     # class × event × decision matrix
        registry.py
        types.py
    components_<domain>/              # YOUR stabilization-component dir
        __init__.py
        <existing files from prior frontier, if any>
```

Activation rule: every `*.py` in `components_<domain>/` (except `_*.py`) is
loaded. **Delete** a file to disable. **Overwrite** a file to replace.
**Add** a file with a unique `COMPONENT.name` to extend.

There is no workflow.yaml. There is no Patch op. The file system is the state.

## The component model (unchanged)

```python
@dataclass(frozen=True, kw_only=True)
class Component:
    name: str                              # stable id; reuse to replace
    cls: ComponentClass                    # mechanism_layer | reactive_guard | induced_rule
    listens: str                           # event name the dispatcher routes on
    matcher: Optional[Callable[[Ctx], bool]]
    handler: Callable[[Ctx], Decision]
    trust: Trust                           # required verification block
    priority: int = 100                    # smaller fires first within an event bucket
    emits: tuple[str, ...] = ()            # custom Tier-2/3 events this raises
```

Components in this v_N use **relative imports** so `cp -r` carries them forward:

```python
from ..runtime import (
    Component, ComponentClass, ComponentContext, Decision, Trust,
)
```

## Lifecycle events the EnterpriseOps runtime emits

Per-task setup:

| event | when it fires | typical use |
|---|---|---|
| `task_received` | top of `run_task` | lifecycle anchor |
| `session_start` | once per task, after MCP tools discovered | one-shot policy injection |
| `pre_prompt_build` | per task; default system+user prompts built | rewrite user prompt; inject domain policy |
| `pre_context_build` | alias of `pre_prompt_build` | same |
| `pre_agent_construct` | last hook before BenchmarkConfig sealed | inference-hint injection |

Per LLM turn (avg ~9 turns / task, max 34):

| event | when it fires | typical use |
|---|---|---|
| `pre_llm_turn` | per turn, before `llm_client.invoke_with_tools()` | rewrite messages list |
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

**Why `pre_final_emit` is weak**: judging is outcome-based on the final
DB state, not on the agent's last text. Rewriting `ctx.final_output` does
not change verifier outcomes. Useful events: `pre_tool_arg_validation` /
`pre_tool_use` (deterministic arg fix-up), `post_llm_response` / `on_*`
(catch and retry), and `session_start` / `pre_prompt_build` (one-shot
policy injection), plus `on_explicit_terminate` for artifact gates.

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

Required: `evidence_anchor` / `blast_radius` / `rollback_when`.
`out_of_evidence_probe` required for `induced_rule`. Optional: `fallback`.

`blast_radius`:
| value | when |
|---|---|
| `local` | a single component file with a narrow matcher |
| `workflow` | multiple components or a coordinated multi-file stabilization change |
| `agent` | you edited `agent.py` (capability path) — call it out |

- `evidence_anchor`: name a stable structure OUTSIDE evidence — an MCP tool
  JSON Schema field, RFC 5322, ISO 8601, the gym server's `list_tools`
  response, an OpenAI/DeepSeek API field. If your anchor is "I observed
  row_017/023/041 all do X", you're anchored INSIDE evidence.
- `out_of_evidence_probe` (induced_rule only): name one concrete case NOT
  in evidence rows where the matcher fires and what handler returns.

### Handler helpers on `ctx`

| helper | what it does |
|---|---|
| `ctx.chat(messages, max_tokens=..., temperature=..., system_override=..., tools=...)` | sub-LLM via the locked SUT model (NOT upstream LangChain) |
| `ctx.shared` | per-task dict, free to read/write |
| `ctx.state[component_name]` | per-task per-component scratchpad |
| `ctx.emit("custom_event_name", **fields)` | fire custom event; re-enters dispatcher (depth cap 10) |
| `ctx.emit_upstream(key, value)` | shorthand for `ctx.upstream[key] = value` |

Per-event fields: `ctx.benchmark` (domain slug), `ctx.user_info`
(`{user_id, name, email, timezone}`), `ctx.gym_servers`, `ctx.tool_specs`
(each `{name, description, input_schema, _mcp_server_name, ...}`),
`ctx.selected_tools`, `ctx.extras`, `ctx.system_prompt`, `ctx.user_prompt`,
`ctx.messages`, `ctx.raw_response`, `ctx.tool_calls`, `ctx.turn_index`,
`ctx.finish_reason`, `ctx.current_tool_name / _args / _result /
_result_str / _call_id / _success / _error / _server`, `ctx.final_output`.
**NEVER read `ctx.verifiers`** (grading rubric). NEVER read `ctx.task_id`.

## The class × event × decision matrix

Load-time gate at `runtime/policy.py::ALLOWED`. See that file for the
authoritative truth.

- Setup events: `mechanism_layer` admits `inject_context` (+ `rewrite` /
  `block` on `pre_prompt_build` / `pre_context_build`); `induced_rule`
  admits `inject_context` (ADVISORY) on `pre_prompt_build` /
  `pre_context_build`; `reactive_guard` does not fire.
- Per-turn / per-tool events: `mechanism_layer` and `reactive_guard`
  admit `rewrite` / `block` / `inject_context` per the cell.
- Exit events: `mechanism_layer` and `reactive_guard` admit `rewrite` +
  `block` on `pre_final_emit`, `block` on `on_explicit_terminate`,
  `allow` on `session_end`.
- `predictive_heuristic` is rejected at every event.

## Hard rules

- **ONE coherent change per iteration.** Multiple files OK if they realize
  one hypothesis; multiple unrelated hypotheses are NOT.
- **You do NOT run benchmarks.** The outer loop scores.
- **No task-specific code.** No entity literals (no calendar names, user
  names, UUIDs, gold answers).
- **You may NOT read `ctx.verifiers`** to drive component logic.
- **The target inference model is LOCKED.** Components do NOT call
  langchain / direct API for additional calls. `ctx.chat()` IS permitted
  (goes through `agent.llm.chat`, same locked SUT model).
- **Stay inside the candidate dir** the outer loop gave you. Do NOT modify
  the frozen `agent/enterpriseops/v0/` or other v_N dirs.
- **Cross-domain isolation**: A calendar iter does NOT modify
  `components_itsm/` and vice versa. The agent.py is shared inside YOUR
  candidate dir but if your edits there benefit only one domain, be honest
  in the hypothesis.
- READ-ONLY (outside the candidate dir): `third_party/EnterpriseOps-Gym/`,
  `agent/enterpriseops/v0/` (frozen control), all other v_*_N dirs,
  `ballast/scripts/run_enterpriseops_baseline.py`,
  `ballast/evolve_enterpriseops.py`,
  `agent/llm.py`.
- Component file naming: `components_<domain>/<descriptive_slug>.py`.
  `COMPONENT.name` SHOULD encode `<domain>_<slug>` to keep namespaces clear
  across v_N forks.

## Capability path: editing `agent.py`

Common edits:

| Edit | What it accomplishes |
|---|---|
| Extend `SYSTEM_PROMPT` (a constant set inside `run_task` or alongside it) with an API cheatsheet | Teach baseline the API quirks (e.g. Google Calendar `defaultReminders` lives on calendarList) instead of stamping a session_start inject_context |
| Add a new tool in the wrapper agent | Genuine capability extension (rare; usually a gym-server change is needed) |
| Wrap MCP tool args with a schema validator before the LLM ever sees them | Cross-cutting deterministic guard (alt: pre_tool_arg_validation component) |
| Change `max_iterations` or per-turn `max_tokens` | Address loop exhaustion / length truncation systemically |
| Add a verifier sub-LLM step inside the orchestrator subclass | Cross-cutting verification (alt: post_llm_response component) |

If your change is a one-liner add to SYSTEM_PROMPT, prefer that over a
session_start component — it's the canonical capability path and avoids
hook-noise across unrelated tasks.

## Stabilization path: writing a component

```python
# components_<domain>/<slug>.py
from __future__ import annotations

from ..runtime import (
    Component, ComponentClass, ComponentContext, Decision, Trust,
)


def _matches(ctx: ComponentContext) -> bool:
    # Read ctx.raw_response / ctx.final_output / ctx.current_tool_name /
    # ctx.current_tool_args / ctx.current_tool_result_str / ctx.finish_reason /
    # ctx.user_info / ctx.tool_specs / ctx.current_tool_server / ctx.shared.
    # ctx.event names the firing event for branchable handlers.
    # NEVER read ctx.verifiers. NEVER read ctx.task_id.
    ...


def _handler(ctx: ComponentContext) -> Decision:
    return Decision.rewrite(...)   # or inject_context / block / allow


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
  "candidate": {
    "name": "candidate_iter<N>_<slug>",
    "hypothesis": "<one-sentence falsifiable claim of the failure mode>",
    "changes": "<plain-English summary of what was edited and why>",
    "edited_files": [
      "agent/enterpriseops/v_<domain>_<N>/agent.py",
      "agent/enterpriseops/v_<domain>_<N>/components_<domain>/<slug>.py"
    ],
    "trust": {
      "evidence_anchor": "...",
      "blast_radius": "local | workflow | agent",
      "rollback_when": "...",
      "out_of_evidence_probe": "<required for induced_rule>",
      "fallback": "..."
    }
  }
}
```

`edited_files` is mandatory and lists every file you touched inside the
candidate dir. The outer loop reads this for the audit trail.

## How to investigate

1. **Read `frontier_val.json`'s `per_task`** at
   `ballast/logs_components_enterpriseops_<domain>/frontier_val.json`
   to find tasks the frontier scores 0 on. Each entry has `agent`, `score`
   (0/1), `verifier_pass_rate` (partial credit; useful for picking
   "almost-correct" failures).

2. **Read v0 per-task traces** at
   `ballast/logs_components_enterpriseops_<domain>/v0__train__traces/<task_id>.json`.
   Each file's `result.runs[0]` has `conversation_flow`, `tools_used`,
   `tool_results`, `verification_results` (per-verifier name →
   `{passed, error, details}`), `verification_summary`, `overall_success`.

3. **Read the prior iteration's component fire trace** at
   `.component-state-enterpriseops/<run_tag>/fired.jsonl` (absent on the
   first iter past v0). Confirms whether the prior frontier's components
   are actually firing.

4. **Look for ≥3 failures sharing the same mechanism.** Examples:
   - 3 failures call `insert_acl_rule` with `scope_email: "carol"` but
     verifier checks `scope_email = 'carol.white@techcorp.com'` →
     `pre_tool_arg_validation` MECHANISM_LAYER that looks up the full
     email in `ctx.user_prompt` / `ctx.user_info` and rewrites. Anchor:
     RFC 5322 + tool's `scope_email` JSON Schema field.
   - 3 failures emit `create_event` with naive datetime and no timezone →
     `pre_tool_arg_validation` REWRITE that fills `start_timezone='UTC'`.
     Anchor: ISO 8601 + tool's schema.
   - 4 failures span different API quirks (visibility enum, defaultReminders
     routing, freebusy items scope) → capability path: extend SYSTEM_PROMPT
     in agent.py with one focused API cheatsheet covering all the relevant
     gotchas. Anchor: documented API resource schemas.

5. **Form ONE hypothesis** tied to a stable structure. State it in
   `trust.evidence_anchor`.

6. **Pick the path:**
   - **One specific arg-shape error** → component on `pre_tool_arg_validation`.
   - **Multiple API-knowledge gaps** → capability edit on `SYSTEM_PROMPT` in agent.py.
   - **Schema-derivable validation** → component on `pre_tool_use` that reads
     `ctx.tool_specs` and blocks with a hint; or do the schema check inline
     in the orchestrator.

7. **Validate** that the candidate dir still imports:
   ```bash
   python -c "
   import importlib
   m = importlib.import_module('agent.enterpriseops.v_<domain>_<N>.agent')
   r = importlib.import_module('agent.enterpriseops.v_<domain>_<N>.runtime')
   print('agent OK; components:',
         [c.name for c in r.load_components_from_dir(
             'agent/enterpriseops/v_<domain>_<N>/components_<domain>')])
   "
   ```

8. **Write `pending_eval.json`** with the schema above and print
   `CANDIDATE: <name>`.

## Common stabilization patterns

| pattern | class | listens | decision |
|---|---|---|---|
| inject "always pass full email in ACL scope" | mechanism_layer | `session_start` | inject_context |
| inject domain policy reminder | mechanism_layer | `pre_prompt_build` | inject_context |
| advisory: policy paragraph paraphrase | induced_rule | `pre_context_build` | inject_context |
| normalise bare-local-part email (RFC 5322) | mechanism_layer | `pre_tool_arg_validation` | rewrite |
| fill `start_timezone='UTC'` on naive datetime (ISO 8601) | mechanism_layer | `pre_tool_arg_validation` | rewrite |
| schema validator: block `update_calendar(defaultReminders=…)` and hint to use `update_calendar_in_list` | mechanism_layer | `pre_tool_arg_validation` | block |
| retry hint on MCP 4xx | reactive_guard | `on_tool_error` | inject_context |
| length-recovery via sub-LLM with bigger budget | reactive_guard | `on_length_truncation` | rewrite via `ctx.chat(max_tokens=32768)` |
| "use list_X with filter" reminder | reactive_guard | `post_llm_response` | inject_context |
| refuse termination without expected DB write | reactive_guard | `on_explicit_terminate` | block |

## Custom events (Tier 2/3)

```
iter<N>_<slug>_<event>        e.g. iter9_acl_normalize_resolved_email
on_<thing>                    failure-mode style
```

Declare `emits=(...)` on the publisher; subscribers declare
`listens="iter<N>_<slug>_<event>"`. To admit non-`allow` decisions on a
custom event, add a string-key entry to `ALLOWED[ComponentClass.X]` in
`runtime/policy.py` inside your candidate dir (since runtime is part of
the v_N tree, edits propagate via fork).

## What this skill does NOT do

- Run benchmarks (the outer loop does).
- Modify anything outside the candidate dir (frozen v0, other v_N dirs,
  ballast, third_party).
- Memorise train-set answers.
- Read `ctx.verifiers` to figure out what value to write.
- Short-circuit the agent loop (construct the entire correct MCP-call
  sequence in Python).
- Call langchain / direct API for sub-LLM (use `ctx.chat()` instead).
- Fire a component on every task (`matcher=None`, priority=0) — that's
  effectively a system_prompt rewrite; put it in `SYSTEM_PROMPT` instead.
- Loop or propose multiple unrelated hypotheses in one invocation.
- Encode a domain policy interpretation as `mechanism_layer` override
  (`rewrite` / `block`). Route to `induced_rule` + `inject_context` only.
- Encode a prompt-shape regex as `predictive_heuristic` — rejected at load.
