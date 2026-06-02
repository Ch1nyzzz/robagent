---
name: component-harness-toolathlon
description: Propose ONE hook (a single Python file declaring `COMPONENT: Component`) that stabilizes the Toolathlon main agent on a recurring failure mode, plus a workflow patch (add / replace / disable) against `ballast/workflows/toolathlon_main.yaml`. The main agent is the OpenAI Agents SDK Runner over an MCP gateway (filesystem / GitHub / Notion / GCP / arxiv-local / scholarly / browser); every MCP tool is wrapped as a FunctionTool so the dispatcher sees REAL args + result. Hooks subscribe to lifecycle events via `listens=` and react via a Decision (allow / block / rewrite_tool_args / inject_context). Some SDK-internal events are declared but not yet emitted in v1.
---

# component-harness-toolathlon

The Toolathlon main agent is the protagonist. It runs an OpenAI Agents SDK `Runner.run(...)` loop over an MCP gateway with the locked SUT model (deepseek-v4-pro via Together AI) and a user simulator (for multi-turn tasks). Toolathlon mandates v2 tool wrapping: every MCP tool is wrapped as an SDK FunctionTool BEFORE the Agent sees it, so the dispatcher can intercept BEFORE and AFTER the real tool invocation with actual args and actual result strings. **Your job is to stabilize it**, not replace it. You propose ONE hook — a single Python file under `agent_toolathlon/components/<name>.py` exporting `COMPONENT: Component` — plus a workflow patch (`add` / `replace` / `disable`). **You do NOT run benchmarks.** The outer loop applies the patch, scores on the 28-task train subset, and admits or rejects.

## What Toolathlon tasks look like

A task = one directory under `Toolathlon-src/tasks/finalpool/<task_id>/`. The agent receives:
  * a natural-language user instruction (the `task_str`)
  * a set of MCP servers exposing tools (filesystem, GitHub, Notion, GCP, arxiv-local, scholarly, browser, …)
  * a Python-based per-task verifier that runs after the agent finishes

The verifier is **binary** (pass / fail). Failure mechanisms vary widely:
  * **wrong final emission** — passable answer in wrong format (Markdown when plain text asked).
  * **missing required tool call** — answered from prior knowledge without invoking the prescribed lookup.
  * **tool-arg hallucination** — `paper_id="2505.20286v1"` (with version) when the corpus uses versionless ids, or invented filesystem paths outside `/workspace/dumps/workspace`.
  * **prompt-instruction omission** — ignored part of a multi-part request.
  * **scoping mistake** — touched the wrong Notion page / GitHub repo because the verifier's allowlist wasn't read.
  * **runaway tool loop** — retried the same failing tool dozens of times.

## SDK-internal event gap (read this carefully)

Some events live INSIDE the SDK Runner's `Runner.run(...)` call and are not surfaced to the dispatcher. These are **declared in policy.py** (so a forward-compatible YAML can mention them) but **NOT emitted in v1**:

  * `pre_llm_request` — mid-turn pre-inference hook
  * `pre_tool_arg_validation` — mid-turn arg-validation phase
  * `post_tool_result_raw` — mid-turn raw-result anchor
  * `on_tool_error` — mid-turn tool-error event
  * `on_no_tool_call_emitted` — mid-turn no-tool-call event

Subscribers to these events load but never fire. Use the post-Runner subset for those failure modes (`post_llm_response_raw` + `on_length_truncation` + `on_empty_response`); these fire AFTER `Runner.run` returns and see `result.raw_responses`. `DEFER` is also v2.5+ — rejected at load.

## First principles

0. **Main agent is the protagonist.** Hooks stabilize tool args / output format / loop budget; they don't replace the SDK Runner's tool-selection reasoning. Before writing a hook, ask: "would the main agent still be the same agent without this — just less prone to failing on X?" If "no, it'd be doing fundamentally different work" (e.g. you've embedded the correct tool sequence in Python), the hook is too heavy.
1. **Capability vs Stabilization — keep them separate.** If a task is failing because an MCP server isn't registered, that's a capability gap — register it. Hooks are for stabilizing what the FunctionTool wrapper already sees: strip an arxiv version suffix from `paper_id`, refuse a write to a path outside `/workspace/dumps/workspace`, append a strict-format reminder.
2. **The LLM is the last resort within stabilization.** Move system-prompt scaffolding, output-format constraints, tool-arg shape validation, retry-on-loop heuristics into Python.
3. **Code earns its place by capturing stable structure, not by fitting recent failures.** Anchor on a tool's declared JSON Schema, an SDK API field, an MCP server's documented behavior, the user-instruction grammar ("return X, Y, Z in this exact format"). IF/THEN induced from N failed train rows is a memorised map; it overfits.
4. **Prefer events that work in single-turn mode.** ~90% of Toolathlon tasks run with `single_turn_mode=True`; the user simulator never re-prompts. Setup events (`pre_context_build` / `session_start` / `user_prompt_submit`) and per-tool events (`pre_tool_use` / `post_tool_use`) reach the LLM in every task. Post-Runner events (`post_llm_response_raw`, `on_explicit_terminate`, `session_end`) fire after the SDK Runner returns.

## Toolbox (pick what fits — none of these are exclusive)

| When you'd reach for it | What you change |
|---|---|
| Main agent can't reach a needed MCP server / tool | Register the MCP / FunctionTool (Capability expansion — separate workflow, **not this skill**) |
| Main agent calls the right tool with a malformed arg | Write a `pre_tool_use` `mechanism_layer` (REWRITE_TOOL_ARGS) (this skill) |
| Main agent tries to write outside `/workspace/dumps/workspace` | Write a `pre_tool_use` `reactive_guard` (BLOCK) (this skill) |
| Main agent ends without producing the expected artifact | Write an `on_explicit_terminate` `reactive_guard` (BLOCK refuses termination) (this skill) |
| Tool result needs a wrapper note ("the LLM should also check field X") | Write a `post_tool_use` `mechanism_layer` (INJECT_CONTEXT — concatenated into the tool result string) (this skill) |
| Static framework reminder ("return plain text, no markdown") | Write a `mechanism_layer` on `pre_context_build` or `session_start` (this skill) |
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

### Lifecycle events the toolathlon runtime emits

Per-task setup (in `setup_agent`, before SDK Agent is built):

| event | when it fires | typical use |
|---|---|---|
| `task_received` | top of `run_interaction_loop` | lifecycle anchor |
| `pre_context_build` | in `setup_agent`, paired with the instructions-build phase | inject system-prompt scaffolding |
| `session_start` | in `setup_agent`, after `pre_context_build` | inject framework constants |
| `pre_agent_construct` | last hook in `setup_agent` before `Agent(...)` is built | inference-hint injection |
| `user_prompt_submit` | each outer user turn, after user_query obtained | append text to user message before it lands in logs |

Per MCP tool call (via FunctionTool wrapper; sees REAL args + result):

| event | when it fires | typical use |
|---|---|---|
| `pre_tool_use` | inside FunctionTool wrapper, before MCP `call_tool` | ALLOW / REWRITE_TOOL_ARGS / true BLOCK (composes across hooks) |
| `post_tool_use` | inside FunctionTool wrapper, after MCP `call_tool` | INJECT_CONTEXT is **concatenated INTO the tool result string** so the LLM sees it on its next inference |

Post-Runner (after `Runner.run` returns):

| event | when it fires | typical use |
|---|---|---|
| `post_llm_response_raw` | post-`Runner.run`, parses `result.raw_responses` | observe / inject for next turn |
| `on_length_truncation` | **synthesised** when `result.raw_responses[-1].finish_reason=="length"` | flag the limit |
| `on_empty_response` | **synthesised** when `result.final_output.strip()==""` | retry hint |
| `on_explicit_terminate` | after `termination_checker(...)` returns True | **artifact gate**: BLOCK refuses termination and the loop continues |
| `session_end` | top of `save_results` | bookkeeping |

SDK-internal events (`pre_llm_request`, `pre_tool_arg_validation`, `post_tool_result_raw`, `on_tool_error`, `on_no_tool_call_emitted`) and `stop` are declared in policy but NOT emitted in v1.

`matcher_for_tool("tool_name")` from `agent_toolathlon.component_runtime.types` scopes per-tool matchers cleanly.

### Component classes

| class | what the matcher tests | risk |
|---|---|---|
| `mechanism_layer` | system field / MCP tool JSON Schema / SDK lifecycle property / general algorithm (URL parse, arxiv version strip) | LOW |
| `reactive_guard` | observed failure event (tool error string contains "denied", missing required substring in final response, repeated identical call) | LOW |
| `induced_rule` | reading of policy / instruction text — IF/THEN compiled from N evidence rows | HIGH (advisory `inject_context` only) |
| `predictive_heuristic` | raw prompt / response text guessing what the model is about to do | REJECTED at load time |

### Decision

| decision | semantics |
|---|---|
| `allow` | no-op |
| `block` | `pre_tool_use`: the MCP `call_tool` is SKIPPED; the tool result handed to the LLM is a `<component_block …>` string. `on_explicit_terminate`: refuses termination (loop continues). Other events: terminate task |
| `rewrite_tool_args` | `pre_tool_use`: the dict you return REPLACES the LLM-emitted args before the MCP call. Multiple hooks compose left→right |
| `inject_context` | setup events → spliced into `Agent.instructions`. `user_prompt_submit` → appended to user_query before it lands in logs. `post_tool_use` → CONCATENATED into the tool result string (LLM sees next inference). Post-Runner events → queued as note |
| `defer` | **REJECTED at load time** (replay queue is v2.5+) |

`ctx.tool_call` at `pre_tool_use` is a **dict** `{"name": str, "arguments": dict}` (not a tau2-style ToolCall object). Hooks REWRITE_TOOL_ARGS return `Decision.rewrite_tool_args(new_args_dict)`; the apply layer rebuilds the dict.

### Trust

Required: `evidence_anchor` / `blast_radius` (local|workflow|global) / `rollback_when`. `out_of_evidence_probe` required for `induced_rule`. Optional: `fallback`.

- `evidence_anchor`: name a stable structure OUTSIDE evidence — an MCP tool JSON Schema field, an SDK API field (`finish_reason`, `result.final_output`), the user-instruction grammar, an arxiv id grammar, an OpenAI API field. If your anchor is "I observed task_017/023/041 all do X", you're anchored INSIDE evidence — pick a different class or don't write the hook.
- `out_of_evidence_probe` (induced_rule only): name one concrete case NOT in evidence rows where the matcher fires and what handler returns on it.

### Handler helpers on `ctx`

| helper | what it does |
|---|---|
| `ctx.chat(messages, max_tokens=..., temperature=..., system_override=..., tools=...)` | sub-LLM via the locked SUT model (NOT the SDK ModelProvider) |
| `ctx.shared` | per-task dict, free to read/write |
| `ctx.state[component_name]` | per-task per-component scratchpad |
| `ctx.emit("custom_event_name", **fields)` | fire custom event; re-enters dispatcher (depth cap 10) |
| `ctx.emit_upstream(key, value)` | shorthand for `ctx.upstream[key] = value` |

Per-event fields: `ctx.tool_name` and `ctx.tool_args` (on `pre_tool_use` — REAL args via wrapper), `ctx.incoming_message` (`post_tool_use` — dict `{tool_name, output}`), `ctx.history` (snapshot of `task_agent.logs`), `ctx.tool_names` (full tool list), `ctx.proposed_system_prompt` (during setup events), `ctx.domain_policy`. NEVER read or compare against task_id-specific data.

## The class × event × decision matrix

Load-time gate at `agent_toolathlon/component_runtime/policy.py::ALLOWED`.

Setup events (`task_received`, `pre_context_build`, `session_start`, `pre_agent_construct`, `user_prompt_submit`):

| class | admissible decisions |
|---|---|
| `mechanism_layer` | inject_context |
| `induced_rule` | inject_context (`pre_context_build` / `user_prompt_submit` only — ADVISORY) |
| `reactive_guard` | inject_context (`user_prompt_submit` only) |

Per-tool events (`pre_tool_use`, `post_tool_use`):

| class | admissible decisions |
|---|---|
| `mechanism_layer` | `pre_tool_use`: allow / block / rewrite_tool_args. `post_tool_use`: allow / inject_context |
| `reactive_guard` | same |

Post-Runner events (`post_llm_response_raw`, `on_length_truncation`, `on_empty_response`):

| class | admissible decisions |
|---|---|
| `mechanism_layer` | inject_context (+ block on synthesised failure events) |
| `reactive_guard` | same |

Termination events (`on_explicit_terminate`, `stop`):

| class | admissible decisions |
|---|---|
| `mechanism_layer` | block |
| `reactive_guard` | block |

`predictive_heuristic` is rejected at every event. `defer` is rejected at every event.

## The workflow file

Single frontier YAML: `ballast/workflows/toolathlon_main.yaml`.

```yaml
nodes:
  - some_component_name
disabled: []
```

Plus a JSON snapshot `ballast/logs_components_toolathlon/frontier_workflow.json` written on accept.

### Patch ops

| op | meaning |
|---|---|
| `add` | append a new node; `agent_toolathlon/components/<id>.py` must be newly written |
| `replace` | keep the existing id; overwrite the file (same `COMPONENT.name`) |
| `disable` | add the id to `disabled:`; file remains for durability audit |

For `replace`, the **first shell action MUST be** `cp agent_toolathlon/components/<existing>.py agent_toolathlon/components/<existing>.py.bak_iter<N>`.

## Hard rules

- Exactly ONE patch per invocation.
- **You do NOT run benchmarks.** No `toolathlon_runner.py`. The outer loop scores.
- **No task-specific code.** No train task_ids in matchers. No hardcoded entity strings (paper IDs, GitHub repo names, Notion page slugs). No encoded gold answers.
- **The target inference model (deepseek-v4-pro via Together AI) is LOCKED.** Hooks do NOT call any other LLM API. `ctx.chat()` IS permitted (locked SUT model via `agent.llm.chat`, NOT the SDK ModelProvider).
- **Capability gaps are not for hooks.** If a task needs an MCP server not currently registered, file a separate request — do not invent a hook that injects the would-be result.
- **Don't design around SDK-internal events.** Hooks declaring `listens=` one of the 5 SDK-internal events or `stop` load but NEVER fire in v1 — pick a different design.
- **DEFER is rejected at load.** No `Decision.defer(...)` in v1.
- For `replace`, the **first shell action** MUST be the `cp ... .bak_iter<N>` command above.
- READ-ONLY: `Toolathlon-src/`, `bench/toolathlon/`, `toolathlon_runner.py`, `agent_toolathlon/v0/`, `agent_toolathlon/component_runtime/`, `ballast/evolve_toolathlon.py`, `ballast/toolathlon_*.txt`.
- Component file naming: `agent_toolathlon/components/component_iter<N>_<slug>.py`. The `COMPONENT.name` SHOULD follow the same pattern.

## Component file template

```python
# agent_toolathlon/components/component_iter<N>_<slug>.py
from __future__ import annotations

from agent_toolathlon.component_runtime.types import (
    Component, ComponentClass, ComponentContext, Decision, Trust,
)


def _matches(ctx: ComponentContext) -> bool:
    # Read ctx.tool_name (pre/post_tool_use), ctx.tool_args (REAL args via wrapper),
    # ctx.incoming_message (post_tool_use: dict with tool_name + output),
    # ctx.history (snapshot of self.logs), ctx.tool_names (full tool list),
    # ctx.shared, ctx.proposed_system_prompt (during setup events).
    # ctx.event names the firing event for branchable handlers.
    # NEVER read or compare against task_id-specific data.
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
        evidence_anchor="<MCP tool JSON Schema / SDK API field / general algorithm / instruction grammar>",
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
    "name": "candidate_toolathlon_iter<N>_<slug>",
    "hypothesis": "<one-sentence falsifiable claim of the failure mode>",
    "mechanism": "<failure mode this targets>",
    "evidence_task_ids": ["<tid1>", "<tid2>", "<tid3>"],
    "changes": "<plain-English diff vs prior frontier>",
    "expected_delta": "<expected pass-rate change on train-28>",
    "component": {
      "id": "<COMPONENT.name>",
      "cls": "mechanism_layer | reactive_guard | induced_rule",
      "listens": "<event_name>",
      "file": "agent_toolathlon/components/component_iter<N>_<slug>.py",
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
      "file": "agent_toolathlon/components/component_iter<N>_<slug>.py"
    }
  }
}
```

For `disable`, omit `component` and `workflow_patch.file`; only `workflow_patch.name` matters.

## How to investigate

State files and trace locations:

```
ballast/workflows/toolathlon_main.yaml                            active workflow
ballast/logs_components_toolathlon/frontier_workflow.json         frontier snapshot
ballast/logs_components_toolathlon/frontier_val.json              per-task best
ballast/logs_components_toolathlon/evolution_summary.jsonl        one row per iter (incl. rejected)
.component-state-toolathlon/toolathlon_iter<K>/fired.jsonl             which hooks fired in iter K
```

Per-task trace directories (preserved across iters):

```
Toolathlon-runs/v0/finalpool/<tid>/                          v0 baseline / iter 0
Toolathlon-runs/cr/iter<K>/finalpool/<tid>/                  iter K's candidate run
Toolathlon-runs/cr/final_test/finalpool/<tid>/               held-out test (only after evolution finishes)
```

Each `<tid>/` directory holds `traj_log.json`, `eval_res.json`, `host_loop.log`. The `evolution_summary.jsonl` row for iter K already has `per_task[*].dump_dir` filled.

1. **Read `frontier_val.json`'s `per_task` map.** Each entry has `passed`, `tier`, `dump_dir`, `error`. Start here.
2. **If `evolution_summary.jsonl` has rows with iter ≥ 1**, also read them. Each names a `candidate.hypothesis` + `train_score` + `accepted` + `per_task`. Avoid re-proposing a mechanism a prior `hypothesis` already covered; check which tasks a prior candidate broke.
3. **For each failing task_id (`passed=false`)**, read the three trace files:
   * `traj_log.json` — full message log, `config.single_turn_mode`, `key_stats`, `status`.
   * `eval_res.json` — `{pass: bool, details: str}` from the per-task verifier (often the most informative single file).
   * `host_loop.log` — pretty-printed TOOL_CALL / TOOL_OUT / SUMMARY events in chronological order.
4. **Look for ≥3 failures sharing the same mechanism.** Examples:
   - 3 failures end with assistant returning Markdown when instruction asked for "plain text without markdown" → `pre_context_build` INJECT a strict-format reminder. Anchor: the instruction grammar.
   - 3 failures call `gw-arxiv_local-read_paper` with `paper_id="2505.20286v1"` when corpus only has versionless ids → `pre_tool_use` REWRITE_TOOL_ARGS that strips the version suffix. Anchor: arxiv id grammar.
   - 3 failures retry the same failing tool >5 times → `session_start` INJECT a "if a tool fails the same way 3 times, switch strategy" instruction.
5. **Form ONE hypothesis** tied to a stable structure. State it in `trust.evidence_anchor`.
6. **Write ONE hook** with the smallest possible matcher/handler. Resist embedding task-specific entity names.
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

## Common stabilization patterns

| pattern | class | listens | decision |
|---|---|---|---|
| strict-format reminder in instructions | mechanism_layer | `pre_context_build` | inject_context |
| inject "always strip arxiv version suffix" framework note | mechanism_layer | `session_start` | inject_context |
| advisory: policy paragraph | induced_rule | `pre_context_build` | inject_context |
| reactive note appended to user query | reactive_guard | `user_prompt_submit` | inject_context |
| strip arxiv version suffix from tool args | mechanism_layer | `pre_tool_use` | rewrite_tool_args |
| block tool call outside workspace dir | reactive_guard | `pre_tool_use` | block (skips this call only) |
| reformat tool output before LLM sees it | mechanism_layer | `post_tool_use` | inject_context (concatenated into tool result string) |
| artifact gate before termination | reactive_guard | `on_explicit_terminate` | block (refuses termination; loop continues) |
| length-recovery hint | reactive_guard | `on_length_truncation` | inject_context |
| two-hook coordination | mechanism_layer | A `listens="post_llm_response_raw"` emits `iter<N>_<slug>_progress` → B `listens="iter<N>_<slug>_progress"` | allow / inject_context |

## Custom events (Tier 2/3)

```
iter<N>_<slug>_<event>        e.g. iter9_workspace_artifact_present
on_<thing>                    failure-mode style
```

Declare `emits=(...)` on the publisher; subscribers declare `listens="iter<N>_<slug>_<event>"`. To admit non-`allow` decisions on a custom event, add a string-key entry to `ALLOWED[ComponentClass.X]` in `agent_toolathlon/component_runtime/policy.py`. Grep `agent_toolathlon/components/*.py` for `emits=` to find taken names.

## What this skill does NOT do

- Run benchmarks (the outer loop does).
- Register new MCP servers / FunctionTools (capability expansion — a separate workflow).
- Modify `toolathlon_runner.py`, `Toolathlon-src/`, `agent_toolathlon/v0/`, or `agent_toolathlon/component_runtime/`.
- Memorise train-set answers (encode "if task instruction contains 'Alita', return paper_id=2505.20286").
- Compute the final answer deterministically without calling the LLM (defeats the SUT measurement).
- Use the 5 SDK-internal events or `stop` (declared but not emitted in v1) — pick a different design.
- Use `Decision.defer(...)` (rejected at load in v1).
- Fire on every task (`matcher=None`, priority=0) — that's effectively a prompt rewrite, not a hook.
- Loop or propose multiple patches in one invocation.
- Encode a policy interpretation as `mechanism_layer` override on tool args / termination. Route to `induced_rule` + `inject_context` only.
- Encode a prompt-shape regex as `predictive_heuristic` — rejected at load.
